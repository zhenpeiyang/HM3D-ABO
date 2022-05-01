# The following code is modified from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
import time
from math import radians
import argparse, sys, os, math, re
import bpy, bmesh
from mathutils import Vector, Matrix
from collections import defaultdict
from glob import glob
import numpy as np
from bpy_extras.io_utils import axis_conversion

sys.path.insert(0, '.')
from utils import get_local_split
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument("--process_id", type=int, default=0)
parser.add_argument("--lens", type=int, default=18)
parser.add_argument("--n_proc", type=int, default=0)
parser.add_argument('--views',
                    type=int,
                    default=30,
                    help='number of views to be rendered')
parser.add_argument('--obj',
                    type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--scene',
                    type=str,
                    help='Path to the scene file to be rendered.')
parser.add_argument('--loc',
                    type=str,
                    help='Path to the scene file to be rendered.')
parser.add_argument('--output_folder',
                    type=str,
                    default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument(
    '--scale',
    type=float,
    default=1,
    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles',
                    type=bool,
                    default=False,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split',
                    type=bool,
                    default=False,
                    help='Adds edge split filter.')
parser.add_argument(
    '--depth_scale',
    type=float,
    default=1.4,
    help=
    'Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.'
)
parser.add_argument(
    '--color_depth',
    type=str,
    default='16',
    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format',
                    type=str,
                    default='OPEN_EXR',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--width',
                    type=int,
                    default=640,
                    help='Resolution of the images.')
parser.add_argument('--height',
                    type=int,
                    default=480,
                    help='Resolution of the images.')
parser.add_argument(
    '--engine',
    type=str,
    default='BLENDER_EEVEE',
    help=
    'Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')


# Build intrinsic camera parameters from Blender camera data
#
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width,
                                        camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit, scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(((s_u, skew, u_0), (0, s_v, v_0), (0, 0, 1)))
    return K


# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0], ),
                 R_world2cv[1][:] + (T_world2cv[1], ),
                 R_world2cv[2][:] + (T_world2cv[2], )))
    return RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


def setup_light():
    # Make light just directional, disable shadows.
    h = 1.5
    for i, location in enumerate([(0, -h, 1), (0, -h, -1), (1, -h, 0),
                                  (-1, -h, 0)]):
        # Create light datablock
        light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
        light_data.energy = 30
        # Create new object, pass the light data
        light_object = bpy.data.objects.new(name="my-light%d" % i,
                                            object_data=light_data)
        # Link object to collection in context
        bpy.context.collection.objects.link(light_object)
        # Change light position
        light_object.location = location

    light = bpy.data.lights['Light']
    light.type = 'POINT'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 0


def setup_cam(cam, camera_pose):
    cam.location = (camera_pose[0, 3], camera_pose[1, 3], camera_pose[2, 3])
    bpy.context.view_layer.update()
    for k1 in range(3):
        for k2 in range(3):
            cam.matrix_world[k1][k2] = camera_pose[k1, k2]


def delete_hierarchy(obj):
    names = set([obj.name])

    # recursion
    def get_child_names(obj):
        for child in obj.children:
            names.add(child.name)
            if child.children:
                get_child_names(child)

    get_child_names(obj)

    print(names)
    objects = bpy.data.objects
    for n in names:
        objects[n].select_set(True)

    bpy.ops.object.delete()


def delete_all_objects_except_camera_light():
    objNames = [object.name for object in bpy.data.objects]
    for object in bpy.data.objects:
        if 'camera' not in object.name.lower(
        ) and 'light' not in object.name.lower():
            object.select_set(True)
    bpy.ops.object.delete()


argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render
cam = scene.objects['Camera']

bpy.data.scenes['Scene'].render.use_persistent_data = True
render.engine = args.engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = args.color_depth  # ('8', '16')
render.image_settings.file_format = 'PNG'  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
render.resolution_x = args.width
render.resolution_y = args.height
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create depth output nodes
depth_file_output = nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
depth_file_output.base_path = ''
depth_file_output.format.file_format = args.format
links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = 'PNG'
id_file_output.format.color_depth = args.color_depth

if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    links.new(render_layers.outputs['IndexOB'], id_file_output.inputs[0])

else:
    id_file_output.format.color_mode = 'BW'

    divide_node = nodes.new(type='CompositorNodeMath')
    divide_node.operation = 'DIVIDE'
    divide_node.use_clamp = False
    divide_node.inputs[1].default_value = 2**int(args.color_depth)

    links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
    links.new(divide_node.outputs[0], id_file_output.inputs[0])

# Delete default cube
print('delete default cube')
context.active_object.select_set(True)
bpy.ops.object.delete()
print('finish delete default cube')

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

DATA_DIR = './data/'
SCENE_DIR = f"{DATA_DIR}/hm3d-train-glb"
MODEL_DIR = f"{DATA_DIR}/3dmodels/original"
CAMERA_DIR = f"{DATA_DIR}/camera_pose_configs"


def get_scene_fn(sceneID):
    sceneFn = glob(f"{SCENE_DIR}/*-{sceneID}/*glb")[0]
    return sceneFn


def get_object_fn(objID):
    objFn = glob(f"{MODEL_DIR}/*/{objID}.glb")[0]
    return objFn


def load_pose_config(sceneID, objID):
    poseFn = f"{CAMERA_DIR}/{sceneID}_{objID}.npy"
    configure = np.load(poseFn, allow_pickle=True).item()
    camera_poses = configure['camera_pose']
    if args.debug:
        camera_poses = configure['camera_pose'][3:4]
    obj_pose = configure['obj_pose']
    angleY = -configure['angle']  # negate as habitat y -> blender -y
    return camera_poses, obj_pose, angleY


files = glob(f"{CAMERA_DIR}/*")
files.sort()
files = [x.split('/')[-1].split('.')[0] for x in files]
configs = defaultdict(list)
scenes = set()
for x in files:
    sceneID, modelID = x.split('_')
    configs[sceneID].append(modelID)
    scenes.add(sceneID)
scenes = sorted(list(scenes))

if args.n_proc > 0:
    scenes = get_local_split(scenes, args.n_proc, args.process_id, seed=666)

cam.data.lens = args.lens

## setup GPU
if args.gpu:
    bpy.context.preferences.addons[
        "cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    print(bpy.context.preferences.addons["cycles"].preferences.
          compute_device_type)
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1  # Using all devices, include GPU and CPU
        print(d["name"], d["use"])

FPS = []
for i_scene, sceneID in enumerate(scenes):
    print(f"progress {i_scene}/{len(scenes)}")
    sceneFn = get_scene_fn(sceneID)

    print('import scene mesh...')
    delete_all_objects_except_camera_light()
    bpy.ops.import_scene.gltf(filepath=sceneFn)
    print('finish import scene mesh')

    ## setup light
    setup_light()
    objects = configs[sceneID]
    old_objs = set(
        context.scene.objects)  #These are scene + camera + lighthing
    for i_object, objID in enumerate(objects):
        print(f" progress: {i_scene}/{len(scenes)}, {i_object}/{len(objects)}")
        objFn = get_object_fn(objID)
        ## load configuration
        camera_poses, obj_pose, angleY = load_pose_config(sceneID, objID)
        ## remove inserted object if exists
        for object in bpy.data.objects:
            if object not in old_objs and object.parent is None:
                delete_hierarchy(object)
                break
        assert (len(set(context.scene.objects) - old_objs) == 0)

        ## insert object
        bpy.ops.import_scene.gltf(filepath=objFn)

        ## infer object root Node
        imported_objs = set(context.scene.objects) - old_objs
        num_root = 0
        for obj in imported_objs:
            if obj.parent is None:
                obj_name = obj.name
                object_node = obj
                num_root += 1
        assert (num_root == 1)
        ## change object center to bounding box center
        object_node.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        object_node.select_set(False)
        ## set object location
        location = obj_pose[:3, 3]
        rotation = obj_pose[:3, :3]

        bpy.context.scene.objects[obj_name].location = Vector(
            (location[0], location[1], location[2]))
        bpy.context.scene.objects[obj_name].rotation_mode = 'XYZ'
        bpy.context.scene.objects[obj_name].rotation_euler = (radians(90),
                                                              angleY, 0)
        ## set object semantic id
        object_node.pass_index = 1
        ## change light location acoording to object
        h = -1.0
        light_locations = [(-1, h, 0), (1, h, 0), (0, h, 1), (0, h, -1)]
        for k in range(4):
            light = bpy.context.scene.objects[f"my-light%d" % k]
            light.location = location + light_locations[k]

        counter = 0
        for object in bpy.context.scene.objects:
            if object.name in ['Camera', 'Lamp', 'root', 'Light']:
                continue
            if object.data is None:
                continue
            if "scene" not in object.name:
                context.view_layer.objects.active = object
            # Possibly disable specular shading

        model_identifier = f"{sceneID}_{objID}"
        fp = os.path.join(os.path.abspath(args.output_folder),
                          model_identifier, model_identifier)

        if not os.path.exists(
                os.path.join(os.path.abspath(args.output_folder),
                             model_identifier)):
            os.mkdir(
                os.path.join(os.path.abspath(args.output_folder),
                             model_identifier))

        ## write intrinsic
        _, K, _ = get_3x4_P_matrix_from_blender(cam)
        with open(fp + '_K.txt', 'w') as f:
            for i in range(3):
                f.write('%f %f %f\n' % (K[i][0], K[i][1], K[i][2]))

        for i in range(len(camera_poses)):
            print(f"frame {i}/{len(camera_poses)}")
            st = time.time()

            setup_cam(cam, camera_poses[i])

            render_file_path = fp + '_r_{0:03d}'.format(int(i))

            scene.render.filepath = render_file_path
            depth_file_output.file_slots[0].path = render_file_path + "_depth"
            id_file_output.file_slots[0].path = render_file_path + "_id"
            pose = render_file_path + "_pose.txt"  # % (stepsize * i)
            pose_obj = render_file_path + "_poseobj.txt"  # % (stepsize * i)
            bpy.context.view_layer.update()
            bpy.ops.render.render(write_still=True)  # render still

            ## camera pose w.r.t world
            P, K, RT = get_3x4_P_matrix_from_blender(cam)
            ## camera pose w.r.t. object
            with open(pose, 'w') as f:
                for k in range(3):
                    f.write('%f %f %f %f\n' %
                            (RT[k][0], RT[k][1], RT[k][2], RT[k][3]))
                f.write('0 0 0 1\n')

            FPS.append(1 / (time.time() - st))
            print('fps %.3f, num examples %d' % (np.mean(FPS), len(FPS)))
        if args.debug:
            bpy.ops.wm.save_as_mainfile(filepath='./debug.blend')

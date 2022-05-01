import math
import glob
import cv2
import json
import os
import shutil
import random
import sys
import time
import magnum as mn
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import habitat_sim
from habitat_sim.utils import common as ut
from habitat_sim.utils import viz_utils as vut
from habitat_sim.gfx import LightInfo, LightPositionModel
from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from habitat_sim import geo
import argparse
from utils import get_local_split
from habitat_utils import *
from collections import Counter
import scipy
from scipy.spatial.transform import Rotation as R


def make_simulator_from_settings(sim_settings, objects_dir):
    cfg = make_cfg(sim_settings)
    global sim
    global obj_attr_mgr
    global prim_attr_mgr
    global stage_attr_mgr
    global rigid_obj_mgr
    # clean-up the current simulator instance if it exists
    if sim != None:
        sim.close()
    # initialize the simulator
    cfg.sim_cfg.scene_light_setup = habitat_sim.gfx.NO_LIGHT_KEY
    sim = habitat_sim.Simulator(cfg)
    # Managers of various Attributes templates
    obj_attr_mgr = sim.get_object_template_manager()
    obj_attr_mgr.load_configs(objects_dir)
    #obj_attr_mgr.load_configs('/task_runtime/I-SLAM/data')
    prim_attr_mgr = sim.get_asset_template_manager()
    stage_attr_mgr = sim.get_stage_template_manager()
    # Manager providing access to rigid objects
    rigid_obj_mgr = sim.get_rigid_object_manager()

def object_place_filter(sim, loc, vec_front, size_x, size_z):
    """
    Place the object such that there is an walkable space in front of it.
    """
    THRESH = 0.8
    MAX_Y_DELTA = 0.5
    N = 10
    space_hori = size_x * 2
    space_front = size_z * 2
    xs = np.linspace(-0.5, 0.5, N)
    zs = np.linspace(0, 1, N)
    vec_right = np.array([1., 0., 0.]) 
    cnt = sum(sim.pathfinder.is_navigable(loc + space_front * vec_front * zs[i] + space_hori * vec_right * xs[j], MAX_Y_DELTA)  for i in range(N) for j in range(N))
    return cnt / (N**2) > THRESH

def sample_object_state(sim,
                          obj,
                          from_navmesh=True,
                          maintain_object_up=True,
                          max_tries=100,
                          bb=None):
    """
    sample a random valid state for the object from the scene bounding box or navmesh. 
    """
    if not sim.pathfinder.is_loaded:
        print("sample_object_state : no pathfinder, aborting.")
        return False
    tries = 0
    valid_placement = False
    # Note: following assumes sim was not reconfigured without close
    scene_collision_margin = stage_attr_mgr.get_template_by_id(0).margin
    scene_collision_margin = 0.0
    vec_front = np.array([0, 0, 1])
    vec_left = np.array([1, 0, 0])

    xform_bb = habitat_sim.geo.get_transformed_bb(
        obj.root_scene_node.cumulative_bb, obj.root_scene_node.transformation)
    size_x, size_y, size_z = xform_bb.size()

    rotation0 = obj.rotation
    # random rotation only on the Y axis
    theta = mn.Rad(random.random() * 2 * math.pi)

    while not valid_placement and tries < max_tries:
        tries += 1
        sample_location = sim.pathfinder.get_random_navigable_point()
        obj.translation = sample_location
        if maintain_object_up:
            y_rotation = mn.Quaternion.rotation(theta, mn.Vector3(0, 1.0, 0))
            obj.rotation = y_rotation * rotation0
        else:
            # unconstrained random rotation
            obj.rotation = ut.random_quaternion()
        vec_front_cur = np.array(
            obj.rotation.transform_vector(
                mn.Vector3(vec_front[0], vec_front[1], vec_front[2])))
        vec_left_cur = np.array(
            obj.rotation.transform_vector(
                mn.Vector3(vec_left[0], vec_left[1], vec_left[2])))
        if not object_place_filter(sim, sample_location, vec_front_cur, size_x, size_z):
            continue

        # raise object such that lowest bounding box corner is above the navmesh sample point.
        obj_node = obj.root_scene_node
        xform_bb = habitat_sim.geo.get_transformed_bb(obj_node.cumulative_bb,
                                                      obj_node.transformation)
        # also account for collision margin of the scene
        obj.translation += mn.Vector3(
            0,
            xform_bb.size_y() / 2.0 + scene_collision_margin, 0)
        # test for penetration with the environment
        if not sim.contact_test(obj.object_id):
            valid_placement = True

    if not valid_placement:
        return False
    obj.semantic_id = 1
    return True


def sample_camera_locations(obj, n_cams):
    obj_loc = np.array(obj.translation)
    obj_node = obj.root_scene_node
    xform_bb = habitat_sim.geo.get_transformed_bb(obj_node.cumulative_bb,
                                                  obj_node.transformation)
    diag = np.linalg.norm(xform_bb.size())
    r = min(diag / 2 + np.random.uniform(0.5, 1.5), diag)
    translations, rotations = [], []
    for i in range(n_cams):
        theta = np.random.uniform(0, np.pi * 2)
        phi = np.random.uniform(0, np.pi / 6)
        delta = np.array([
            r * np.sin(theta) * np.cos(phi), r * np.sin(phi),
            r * np.cos(theta) * np.cos(phi)
        ])
        delta[1] = max(min(delta[1], 1.5 - xform_bb.size_y() / 2),
                       -xform_bb.size_y() / 2)  # max obs height 1.0m
        cam_loc = obj_loc + delta
        nav = sim.pathfinder.is_navigable(cam_loc, max_y_delta=3.0)
        if not nav: continue
        offset = np.random.uniform(-diag * 0.1, diag * 0.1, [3])
        look_at = obj_loc - cam_loc + offset
        up = np.array([0, 1.0, 0])
        look_at /= np.linalg.norm(look_at)
        right = np.cross(look_at, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, look_at)
        up /= np.linalg.norm(up)
        rotation = mn.Quaternion.from_matrix(mn.Matrix3x3(right, up, -look_at))
        translations.append(cam_loc)
        rotations.append(rotation)

    return tuple(zip(translations, rotations))


def setup_object(obj_attr_mgr, objectID):
    file_obj_handles = obj_attr_mgr.get_file_template_handles()
    idx = np.where([objectID in x for x in file_obj_handles])[0][0]
    sel_file_obj_handle = file_obj_handles[idx]
    object_scale = 1
    sel_obj_template_cpy = obj_attr_mgr.get_template_by_handle(
        sel_file_obj_handle)
    sel_obj_template_cpy.scale = mn.Vector3(object_scale)
    obj_attr_mgr.register_template(sel_obj_template_cpy, "scaled_sel_obj")
    obj_1 = rigid_obj_mgr.add_object_by_template_handle("scaled_sel_obj")

    objectID = sel_file_obj_handle.split('/')[-1].split('.')[0]
    return obj_1, objectID, sel_file_obj_handle


def filter_crit(camera_cfg):
    translation, rotation = camera_cfg
    sim.agents[0].scene_node.translation = mn.Vector3(translation)
    sim.agents[0].scene_node.rotation = rotation
    observation = sim.get_sensor_observations()
    rgb_img = Image.fromarray(observation["color_sensor_1st_person"],
                              mode="RGBA")
    mask = observation['semantic_sensor_1st_person']
    x, y, w, h = cv2.boundingRect(np.stack(np.where(mask), 1)[:, ::-1])
    ratio = (w * h) / float(mask.shape[0] * mask.shape[1])
    return ratio > args.FILTER_CRIT_OBJECT_RATIO


def save_cfgs(fn, obj, camera_cfgs):
    translations = [x[0] for x in camera_cfgs]
    rotations = [x[1] for x in camera_cfgs]
    T_obj = np.eye(4)
    T_obj[:3, 3] = obj.translation
    T_obj[:3, :3] = obj.rotation.to_matrix()

    T = np.tile(np.eye(4)[None, :, :], [len(translations), 1, 1])
    T[:, :3, 3] = np.array(translations)
    for k in range(len(rotations)):
        T[k, :3, :3] = rotations[k].to_matrix()
    # habitat x,y,z -> blender x,-y,-z (https://github.com/facebookresearch/habitat-sim/issues/543)
    T[:, 1:3, :] *= -1
    T_obj[1:3, :] *= -1

    angleY = float(obj.rotation.angle())
    R = scipy.spatial.transform.Rotation.from_euler(
        'xyz', [90, np.rad2deg(angleY), 0], degrees=True).as_matrix()
    T_objv2 = np.eye(4).astype('float32')
    T_objv2[:3, :3] = R
    T_objv2[:3, 3] = T_obj[:3, 3]

    np.save(
        fn,
        {
            'camera_pose': T,
            'obj_pose':
            T_obj,  # This is formated for blender's coorindate system.
            'obj_posev2': T_objv2,  # This is used during network training.
            'angle': float(obj.rotation.angle()),
            'axis': np.array(obj.rotation.axis())
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_object_data", action="store_true")
    parser.add_argument("--camera_mode", type=str, default='v0')
    parser.add_argument(
        "--scene",
        type=str,
        default="./data/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
    )
    parser.add_argument("--process_id", type=int, default=0)
    parser.add_argument("--n_proc", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--remove_margin", action="store_true")
    parser.add_argument("--num_scene_sample_per_object", type=int, default=30)
    parser.add_argument("--num_gpu", type=int, default=4)
    parser.add_argument("--assets_dir", type=str, default='/mnt/data')
    parser.add_argument("--outputs_dir", type=str, default='/mnt/data')
    parser.add_argument("--physical_simulation", type=int, default=1)
    parser.add_argument("--MIN_CAMERA_PER_SAMPLE", type=int, default=10)
    parser.add_argument("--FILTER_CRIT_OBJECT_RATIO", type=float, default=0.2)
    parser.add_argument("--NUM_CAM_TRIAL", type=int, default=100)

    args = parser.parse_args()

    # define some globals the first time we run.
    if "sim" not in globals():
        global sim
        sim = None
        global obj_attr_mgr
        obj_attr_mgr = None
        global prim_attr_mgr
        obj_attr_mgr = None
        global stage_attr_mgr
        stage_attr_mgr = None
        global rigid_obj_mgr
        rigid_obj_mgr = None

    seed = 2
    random.seed(seed)
    np.random.seed(seed)

    DATA_DIR_OBJECT = f"{args.assets_dir}/3dmodels/original/"
    DATA_DIR_SCENE = f"{args.assets_dir}/hm3d-train-habitat/"
    OUTPUT_DIR = args.outputs_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TEMP_DIR_HABITAT_SIM = f"./temp/habitat"
    os.makedirs(TEMP_DIR_HABITAT_SIM, exist_ok=True)

    ## Find all object models
    objects = glob.glob(f"{DATA_DIR_OBJECT}/*/*glb")
    objects.sort()

    ## Use our filtered model list (chair+sofa)
    with open(f"../../data/abo_list.txt", 'r') as f:
        filtered_models = f.readlines()
        filtered_models = [x.strip() for x in filtered_models]

    ## Distribute jobs among process
    if args.n_proc > 0:
        filtered_models = get_local_split(filtered_models,
                                          args.n_proc,
                                          args.process_id,
                                          seed=666)
    objects = list(
        filter(lambda x: x.split('/')[-1].split('.')[0] in filtered_models,
               objects))

    ## Create object data file for habitat-sim
    for i, object in enumerate(objects):
        objectID = object.split('/')[-1].split('.')[0]
        if os.path.exists(
                f"{TEMP_DIR_HABITAT_SIM}/{objectID}.object_config.json"):
            continue
        data = {}
        data['render_asset'] = f"{objectID}.glb"
        data['requires_lighting'] = True
        data['join_collision_mesh'] = False
        data['mass'] = 2.0
        data['friction_coefficient'] = 0.5
        data['restitution_coefficient'] = 0.2
        with open(f"{TEMP_DIR_HABITAT_SIM}/{objectID}.object_config.json",
                  'w') as f:
            json.dump(data, f)
        cmd = f"ln {object} {TEMP_DIR_HABITAT_SIM}/"
        os.system(cmd)

    ## Find every scene assets
    scenes = glob.glob(f"{DATA_DIR_SCENE}/*/*glb")
    scenes.sort()

    for _obj, object in enumerate(objects):

        objectID = object.split('/')[-1].split('.')[0]

        ## For each object, we random sample N (scene + placement + cam trajectory)
        for _scene in range(args.num_scene_sample_per_object):
            ## Sample a scene
            np.random.seed(int(time.time()))
            scene = np.random.choice(scenes, 1)[0]
            sceneID = scene.split('/')[-1].split('.')[0]
            sim_settings = make_default_settings(args, scene=scene)
            make_simulator_from_settings(sim_settings, TEMP_DIR_HABITAT_SIM)

            print(
                f"worker: {args.process_id}/{args.n_proc}, progress: {_obj}/{len(objects)}, scenes: {_scene}/{args.num_scene_sample_per_object}"
            )

            ## Setup the object
            seed_value = random.randrange(1e9)
            sim.seed(seed_value)
            rigid_obj_mgr.remove_all_objects()
            obj, objectID, sel_file_obj_handle = setup_object(
                obj_attr_mgr, objectID)

            ## Sample object placement
            success = sample_object_state(sim,
                                            obj,
                                            from_navmesh=True,
                                            maintain_object_up=True,
                                            max_tries=1000)

            if not success:
                continue

            ## Sometimes the object are not touching the ground. Run a few step of physical simulation to let the object stand on the ground.
            if args.physical_simulation:
                ## Sample an observation locations for debugging purpose
                camera_cfgs = sample_camera_locations(obj, 5)
                if not len(camera_cfgs):
                    continue
                sim.agents[0].scene_node.translation = mn.Vector3(
                    camera_cfgs[0][0])
                sim.agents[0].scene_node.rotation = camera_cfgs[0][1]
                observations = simulate(sim, dt=1.0)
                # make_video_cv2(observations, prefix="test")

            ## Move object a little to the ground so that it does not floating over air
            if args.remove_margin:
                obj.translation = obj.translation + mn.Vector3(0, -0.04, 0)

            ## Sample camera configs
            camera_cfgs = sample_camera_locations(obj, args.NUM_CAM_TRIAL)

            ## Filter bad configurations.
            camera_cfgs = list(filter(filter_crit, camera_cfgs))

            ## Only keep the sample if the remaining configurations exceed a certain threshold.
            if len(camera_cfgs) > args.MIN_CAMERA_PER_SAMPLE:
                save_cfgs(f"{OUTPUT_DIR}/{sceneID}_{objectID}.npy", obj,
                          camera_cfgs)

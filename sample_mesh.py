import argparse
import igl
import trimesh
import numpy as np
import os
import glob
import sys
from multiprocessing import Pool
from functools import partial
if 0:
    # TODO: do this better
    sys.path.insert(0, '../occupancy_networks/')
    from im2mesh.utils import binvox_rw, voxels
    from im2mesh.utils.libmesh import check_mesh_contains
import open3d as o3d
from utils import write_ply
import copy

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('--input_folder',
                    type=str,
                    default='dense_view',
                    help='Path to input watertight meshes.')
parser.add_argument('--output_folder',
                    type=str,
                    help='Output path for voxelization.')
parser.add_argument('--n_proc',
                    type=int,
                    default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize',
                    action='store_true',
                    help='When active, resizes the mesh to bounding box.')

parser.add_argument('--rotate_xz',
                    type=float,
                    default=0.,
                    help='Angle to rotate around y axis.')

parser.add_argument('--bbox_padding',
                    type=float,
                    default=0.,
                    help='Padding for bounding box')
parser.add_argument('--bbox_in_folder',
                    type=str,
                    help='Path to other input folder to extract'
                    'bounding boxes.')

parser.add_argument('--pointcloud_size',
                    type=int,
                    default=100000,
                    help='Size of point cloud.')

parser.add_argument('--voxels_res',
                    type=int,
                    default=32,
                    help='Resolution for voxelization.')

parser.add_argument('--points_size',
                    type=int,
                    default=100000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio',
                    type=float,
                    default=1.,
                    help='Ratio of points to sample uniformly'
                    'in bounding box.')
parser.add_argument('--points_sigma',
                    type=float,
                    default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                    'samples on the surfaces.')
parser.add_argument('--points_padding',
                    type=float,
                    default=0.1,
                    help='Additional padding applied to the uniformly'
                    'sampled points on both sides (in total).')

parser.add_argument('--overwrite',
                    action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16',
                    action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits',
                    action='store_true',
                    help='Whether to save truth values as bit array.')


def main(args):
    input_files = glob.glob(f"{args.input_folder}/*/*.off")
    input_files.sort()
    os.makedirs(args.output_folder, exist_ok=True)

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for i, p in enumerate(input_files):
            try:
                print(i, len(input_files))
                process_path(p, args)
            except Exception as e:
                print(i, e)
                pass

def process_path(in_path, args):
    in_file = os.path.basename(in_path)

    modelID = in_path.split('/')[-2].split('.')[0]
    try:
        mesh = trimesh.load(in_path, process=False)
    except:
        print("cannot load off")
        return

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, modelname + '.off')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        if args.rotate_xz != 0:
            angle = args.rotate_xz / 180 * np.pi
            R = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
            mesh.apply_transform(R)

    # Expert various modalities
    out_fn = f"{args.output_folder}/{modelID}.sdf.npz"
    export_sdf(mesh, out_fn, loc, scale, args)
    out_fn = f"{args.output_folder}/{modelID}.point.npz"
    export_pointcloud(mesh, out_fn, loc, scale, args)


def export_pointcloud(mesh, filename, loc, scale, args):

    points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
    normals = mesh.face_normals[face_idx]

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)
    normals = normals.astype(dtype)

    print('Writing pointcloud: %s' % filename)
    np.savez(filename, points=points, normals=normals, loc=loc, scale=scale)


def export_sdf(mesh, filename, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % filename)
        return

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    sample_point = []
    points_surface = mesh.sample(args.points_size // 4)
    SURFACE_SAMPLE_STD = 0.1
    SURFACE_SAMPLE_STD2 = 0.01
    disp = np.random.randn(len(points_surface), 1, 3) * SURFACE_SAMPLE_STD
    sample_point.append((points_surface[:, None, :] + disp).reshape(-1, 3))
    disp = np.random.randn(len(points_surface), 1, 3) * SURFACE_SAMPLE_STD2
    sample_point.append((points_surface[:, None, :] + disp).reshape(-1, 3))
    sample_point.append(
        np.random.uniform(-0.5, 0.5, [len(points_surface) * 2, 3]))
    sample_point = np.concatenate(sample_point)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    S, I, C = igl.signed_distance(sample_point, vertices, faces)

    normal = (sample_point - C)
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal * ((S[:, None] > 0) - 0.5) * 2

    sdf_val = S
    np.savez(filename,
             points=sample_point.astype('float32'),
             sdf=sdf_val.astype('float32'),
             loc=loc,
             scale=scale,
             normal=normal)



if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

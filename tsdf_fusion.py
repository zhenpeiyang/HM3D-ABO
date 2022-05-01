import math
import collections
import glob
import cv2
import numpy as np
import os
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator as rgi
import argparse
import ntpath

import libmcubes
from multiprocessing import Pool

import libfusiongpu as libfusion
from libfusiongpu import tsdf_gpu as compute_tsdf
from utils import *



class Fusion:
    """
    Performs TSDF fusion.
    """
    def __init__(self):
        """
        Constructor.
        """

        parser = self.get_parser()
        self.options = parser.parse_args()

        self.render_intrinsics = np.array([
            self.options.focal_length_x, self.options.focal_length_y,
            self.options.principal_point_x, self.options.principal_point_x
        ],
                                          dtype=float)
        # Essentially the same as above, just a slightly different format.
        self.fusion_intrinsics = np.array(
            [711.111084, 0, 256, 0, 711.111084, 256, 0, 0, 1])  # amazon object
        self.image_size = np.array([
            self.options.image_height,
            self.options.image_width,
        ],
                                   dtype=np.int32)
        self.znf = np.array([1 - 0.75, 1 + 0.75], dtype=float)
        # Derive voxel size from resolution.
        self.voxel_size = 1. / self.options.resolution
        self.truncation = self.options.truncation_factor * self.voxel_size

    def get_parser(self):
        """
        Get parser of tool.

        :return: parser
        """
        parser = argparse.ArgumentParser(
            description='Scale a set of meshes stored as OFF files.')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--in_dir',
                            type=str,
                            default='dense_view',
                            help='Path to input directory.')
        parser.add_argument(
            '--out_dir',
            type=str,
            help='Path to output directory; files within are overwritten!')
        parser.add_argument('--t_dir',
                            type=str,
                            help='Path to transformation directory.')
        parser.add_argument('--n_proc',
                            type=int,
                            default=0,
                            help='Number of processes to run in parallel'
                            '(0 means sequential execution).')
        parser.add_argument('--overwrite',
                            action='store_true',
                            help='Overwrites existing files if true.')

        parser.add_argument('--n_points',
                            type=int,
                            default=100000,
                            help='Number of points to sample per model.')
        parser.add_argument('--n_views',
                            type=int,
                            default=100,
                            help='Number of views per model.')
        parser.add_argument('--image_height',
                            type=int,
                            default=640,
                            help='Depth image height.')
        parser.add_argument('--image_width',
                            type=int,
                            default=640,
                            help='Depth image width.')
        parser.add_argument('--focal_length_x',
                            type=float,
                            default=640,
                            help='Focal length in x direction.')
        parser.add_argument('--focal_length_y',
                            type=float,
                            default=640,
                            help='Focal length in y direction.')
        parser.add_argument('--principal_point_x',
                            type=float,
                            default=320,
                            help='Principal point location in x direction.')
        parser.add_argument('--principal_point_y',
                            type=float,
                            default=320,
                            help='Principal point location in y direction.')
        parser.add_argument('--sample_weighted',
                            action='store_true',
                            help='Whether to use weighted sampling.')
        parser.add_argument('--sample_scale',
                            type=float,
                            default=0.2,
                            help='Scale for weighted sampling.')
        parser.add_argument(
            '--depth_offset_factor',
            type=float,
            default=1.5,
            help=
            'The depth maps are offsetted using depth_offset_factor*voxel_size.'
        )
        parser.add_argument('--resolution',
                            type=float,
                            default=256,
                            help='Resolution for fusion.')
        parser.add_argument(
            '--truncation_factor',
            type=float,
            default=10,
            help=
            'Truncation for fusion is derived as truncation_factor*voxel_size.'
        )

        parser.add_argument('--depth_folder',
                            type=str,
                            default=None,
                            help='The path the output will be dumped to.')
        return parser

    def get_outpath(self, filepath):
        filename = os.path.basename(filepath)
        modelname = os.path.splitext(os.path.splitext(filename)[0])[0]
        outpath = os.path.join(self.options.out_dir, modelname + '.off')
        return outpath

    def fusion(self,
               depthmaps,
               Rs,
               Ts,
               min_x=-0.5,
               min_y=-0.5,
               min_z=-0.5,
               voxel_size=None,
               truncation=None):
        """
        Fuse the rendered depth maps.

        :param depthmaps: depth maps
        :type depthmaps: numpy.ndarray
        :param Rs: rotation matrices corresponding to views
        :type Rs: [numpy.ndarray]
        :return: (T)SDF
        :rtype: numpy.ndarray
        """

        Ks = self.fusion_intrinsics.reshape((1, 3, 3))
        Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

        views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)

        # Note that this is an alias defined as libfusiongpu.tsdf_gpu or libfusioncpu.tsdf_cpu!
        tsdf = compute_tsdf(views, self.options.resolution,
                            self.options.resolution, self.options.resolution,
                            min_x, min_y, min_z, voxel_size, truncation,
                            True)  # important to set this to True

        tsdf = np.transpose(tsdf[0], [2, 1, 0])
        return tsdf

    def run(self):
        """
        Run the tool.
        """
        files = glob.glob(f"{self.options.depth_folder}/*")
        method = self.run_fuse
        if self.options.n_proc == 0:
            for filepath in files:
                method(filepath)
        else:
            with Pool(self.options.n_proc) as p:
                p.map(method, files)

    def run_fuse(self, filepath):
        """
        Run fusion.
        """
        try:
            off_file = filepath + '/model.off'
            if os.path.exists(off_file):
                return
            timer = Timer()
            images = glob.glob(f"{filepath}/*depth*exr")
            images.sort()
            if len(images) != 100:
                return
            pose_files = [x[:-14] + '_pose.txt' for x in images]
            depths = []
            Rs, Ts = [], []
            pc_all = []
            poses = []
            for i in range(len(images)):
                poses.append(np.loadtxt(pose_files[i]))
                depth = cv2.imread(images[i], cv2.IMREAD_ANYCOLOR
                                   | cv2.IMREAD_ANYDEPTH)[:, :, 0]
                depth[depth == np.inf] = 0
                depth[depth > 100] = 0
                depths.append(depth)
                T = np.loadtxt(pose_files[i])
                xyz = transform4x4(
                    backproject_depth(depth,
                                      np.linalg.inv(
                                          self.fusion_intrinsics.reshape(3,
                                                                         3)),
                                      mask=True), np.linalg.inv(T))
                pc_all.append(xyz[np.random.choice(len(xyz), 3000)])
                Rs.append(T[:3, :3])
                Ts.append(T[:3, 3])
            depths = np.stack(depths).astype(np.float32)
            depths[depths == 0] = -1
            poses = np.stack(poses).astype(np.float32)
            Rs = np.stack(Rs).astype(np.float32)
            Ts = np.stack(Ts).astype(np.float32)
            pc_all = np.concatenate(pc_all)
            pc_all = pc_all[np.random.choice(len(pc_all), 100000)]
            if self.options.debug:
                write_ply('test.ply', pc_all)

                visCamera(np.linalg.inv(poses),
                          color=np.array([1, 0, 0]),
                          fn='./pose.ply')
            timer.reset()
            bb_min = pc_all.min(0)
            bb_max = pc_all.max(0)
            voxel_size = np.max(bb_max - bb_min) / self.options.resolution
            truncation = self.options.truncation_factor * voxel_size
            tsdf = self.fusion(depths, Rs, Ts, bb_min[0], bb_min[1], bb_min[2],
                               voxel_size, truncation)
            # To ensure that the final mesh is indeed watertight
            tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
            vertices, triangles = libmcubes.marching_cubes(-tsdf, 0)
            # Remove padding offset
            vertices -= 1
            # Normalize to [-0.5, 0.5]^3 cube
            vertices = vertices * voxel_size + bb_min[None, :]

            modelname = os.path.splitext(
                os.path.splitext(os.path.basename(filepath))[0])[0]

            libmcubes.export_off(vertices, triangles, off_file)
            print('[Data] wrote %s (%f seconds)' % (off_file, timer.elapsed()))
        except Exception as e:
            print(e)
            return


if __name__ == '__main__':
    app = Fusion()
    app.run()

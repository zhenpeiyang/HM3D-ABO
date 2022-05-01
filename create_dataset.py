import numpy as np
import os
import cv2
import glob
import argparse
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser(
    description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--n_proc', type=int, default=0)
parser.add_argument('--temp_folder', type=str, default=None)
parser.add_argument('--output_folder', type=str, default=None)
parser.add_argument('--configs_folder', type=str, default=None)
args = parser.parse_args()

DATA_DIR = args.temp_folder
OUT_DIR = args.output_folder
CAMERA_DIR = args.configs_folder
os.makedirs(OUT_DIR, exist_ok=True)

scenes = glob.glob(f"{DATA_DIR}/*")


def image_filter(image_mask_files):
    """
    filter based on object mask
    """
    idx = np.zeros([len(image_mask_files)])
    for j, fn in enumerate(image_mask_files):
        mask = cv2.imread(fn)[:, :, 0] == 255
        x, y, w, h = cv2.boundingRect(np.stack(np.where(mask), 1)[:, ::-1])
        ratio = (w * h) / float(mask.shape[0] * mask.shape[1])
        if ratio > 0.2:
            idx[j] = 1
    return idx


def process(scene):
    try:
        print(scene)
        sceneID = scene.split('/')[-1]
        odir = os.path.join(OUT_DIR, sceneID)

        ## Find paths
        depths = glob.glob(f"{scene}/*depth*")
        depths.sort()
        rgbs = [x.replace('_depth0001.exr', '.png') for x in depths]
        poses = [x.replace('_depth0001.exr', '_pose.txt') for x in depths]
        ids = [x.replace('_depth0001.exr', '_id0001.png') for x in depths]
        if not np.all([os.path.exists(x) for x in rgbs]) or not np.all(
            [os.path.exists(x)
             for x in ids]) or not np.all([os.path.exists(x) for x in poses]):
            return

        ## Filter out bad images
        idx = image_filter(ids)
        if not idx.sum():
            return

        os.makedirs(odir, exist_ok=True)
        os.makedirs(odir + '/depth', exist_ok=True)
        os.makedirs(odir + '/rgb', exist_ok=True)
        os.makedirs(odir + '/pose', exist_ok=True)
        os.makedirs(odir + '/mask', exist_ok=True)

        sceneID = scene.split('/')[-1]
        K = f"{scene}/{sceneID}_K.txt"
        cmd = f"cp {K} {odir}/intrinsic.txt"
        os.system(cmd)
        counter = 0
        ## Copy data to destination
        for j, depth in enumerate(depths):
            if not idx[j]:
                continue
            cmd = f"cp {depths[j]} {odir}/depth/{counter:06d}.exr"
            os.system(cmd)
            rgb = cv2.imread(rgbs[j])
            cv2.imwrite(f"{odir}/rgb/{counter:06d}.jpg", rgb)
            cmd = f"cp {poses[j]} {odir}/pose/{counter:06d}.txt"
            os.system(cmd)
            mask = cv2.imread(ids[j])
            cmd = f"cp {ids[j]} {odir}/mask/{counter:06d}.png"
            os.system(cmd)
            counter += 1

        ## Write object pose
        config = np.load(f"{CAMERA_DIR}/{sceneID}.npy",
                         allow_pickle=True).item()
        angleY = -config['angle']  # negate as habitat y -> blender -y
        R = scipy.spatial.transform.Rotation.from_euler(
            'xyz', [90, np.rad2deg(angleY), 0], degrees=True).as_matrix()
        T_obj_to_world = np.eye(4).astype('float32')
        T_obj_to_world[:3, :3] = R
        T_obj_to_world[:3, 3] = config['obj_pose'][:3, 3]
        np.savetxt(f"{odir}/obj_pose.txt", T_obj_to_world)

    except:
        return


if args.n_proc == 0:
    for i, scene in enumerate(scenes):
        print(i, len(scenes))
        process(scene)
else:
    with Pool(args.n_proc) as p:
        p.map(process, scenes)

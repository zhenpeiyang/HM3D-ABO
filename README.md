<div align="center">
    
# HM3D-ABO Dataset
    
**HM3D-ABO is a photo-realistic object-centric multi-view dataset.**

    
---
    
<p align="center">
<img src="./data/demo.gif" alt="drawing" width="800"/>
</p>
    
</div>

## ⚙️ Dataset Attributes
It contains 3196 object-scene configurations which involve 1966 objects and 500 indoor scenes. For each configurations, we generate following multi-view captures and corresponding G.T. object models.
- Images
- Depths
- Object masks
- Camera Poses
- 3D Model (pointcloud, SDF, water-tight mesh)


## ⚙️ Installation

```
conda create --name fvor python=3.7
conda activate fvor

```

- Install [blender 2.93.1](https://www.blender.org/download/lts/2-93/)
```
mkdir -p third_party && cd third_party && wget https://mirror.clarkson.edu/blender/release/Blender2.93/blender-2.93.1-linux-x64.tar.xz && tar -xvf blender-2.93.1-linux-x64.tar.xz && rm blender-2.93.1-linux-x64.tar.xz && cd ..
```

- Install habitat-sim. Clone [habitat-sim](https://github.com/facebookresearch/habitat-sim) repo using following commands and install it following their instructions.
```
conda install -y habitat-sim withbullet headless -c conda-forge -c aihabitat
mkdir -p third_party && cd third_party && git clone https://github.com/facebookresearch/habitat-sim.git && cd ../
```

- Install other package dependency
- (Optional) Setup TSDF fusion tools. Only necessary if you want to process the ABO assets yourself.
```
pip install cython
pip install open3d-python
pip install trimesh
pip install opencv-python==4.5.3.56
conda install -y -c conda-forge igl

cd libfusiongpu
mkdir build
cd build
cmake ..
make
cd ..
python setup.py build_ext --inplace
cd ..

cd libmcubes
python setup.py build_ext --inplace
cd ..
```

## ⚙️ Data Assets Download
- [Amazon-Berkeley Object Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) Download *abo-3dmodels.tar*, put the abo-3dmodels.tar file in the data folder. Then uncompress with command
```
tar -xvf abo-3dmodels.tar
```
- [Habitat - Matterport 3D Research Dataset](https://matterport.com/partners/facebook) Download the *hm3d-train-glb.tar* and *hm3d-train-habitat.tar*, put the hm3d-train-habitat.tar file and hm3d-train-glb.tar in the data folder. Then uncompress with commands
```
mkdir hm3d-train-glb && tar -xvf hm3d-train-glb.tar -C hm3d-train-glb
mkdir hm3d-train-habitat && tar -xvf hm3d-train-habitat.tar -C hm3d-train-habitat
```
- [Our Provided Configurations](https://drive.google.com/file/d/1uKSjfEjqUTt44ImB5qYvL9Tv0Hy55eZT/view?usp=sharing)
Includes object placement and camera pose configurations. 
- [Sample Data](https://drive.google.com/file/d/1lr8hcYX7RQw1CDThPG37U9IjeP95rfy0/view?usp=sharing) contains one data object-scene configuration. 


You should have a structure like these:
```
./data/
├── 3dmodels
│   ├── metadata
│   └── original
├── camera_pose_configs
│   └── ZxkSUELrWtQ_B07PC15YLQ.npy
├── hm3d-train-glb
│   └── 00459-Ze6tkhg7Wvc
└── hm3d-train-habitat
    └── 00459-Ze6tkhg7Wvc

```

## ⚙️ Render Images Using Our Configurations

We provide configurations for object placement as well as camera poses. After download our configuration files, run following command to render images using Blender. 
```
bash render.sh
```
Tunable parameters:
- The *lens* argument in render.sh could be adjusted to produce different FoV(field of view). The default *lens*(30) produce 62&deg; horizontal FoV.
- The *n_jobs_per_gpu* could be lowered to accomodate your GPU memory. 

Finally, your dataset structure should be looks like following:
```
data
└── HM3D_ABO
    ├── abo_assets
    │   ├── B075X4YDCM.point.npz
    │   └── B075X4YDCM.sdf.npz
    └── scenes
        └── 1S7LAXRdDqK_B075X4YDCM
            ├── rgb
            ├── pose
            ├── mask
            ├── depth
            ├── intrinsic.txt
            └── obj_pose.txt
```
## ⚙️ Generate Your Own Configurations.
We've included our code for generating object and camera configurations. The basic idea is simple: we find a walkable locations of the scene, and try to place an object into that location and sample some camera around it. This section instruct you how to run our pipeline. 
Before we start, make sure that your have downloaded Amazon-Berkeley Object Dataset and Habitat - Matterport 3D Research Dataset. Make sure you've correctly setup the HM3D and ABO assets as instructed above. 

Then, run following command to sample configurations. 
```
bash sample.sh
```
The results are a set of *{scene_id}_{model_id}.npy* files inside *camera_pose_configs* folder. 

Tunable parameters:
- configure *camera_mode* argument in sample.py for different type of camera poses. 
- configure *OUTPUT_DIR* argument in sample.py. 

## ⚙️ Preprocessing Code for ABO datasets.
Please first follow the installation instructions for Setup TSDF fusion tools.
We generate two files for each ABO models. *model_id*.point.npz contains the surface point cloud and normals. *model_id*.sdf.npz contains sampled signed distance function.

To generate the signed distance function, we follow a similar procedure as [Occupancy Network](https://github.com/autonomousvision/occupancy_networks). We first render the depth map from a dense array of cameras surrounding the object. Then we use TSDF fusion to get the watertight mesh, which was used to calculate the 
signed distance function for sampled points.

```
bash process_abo.sh
```

## Acknowledgement
- [Occupancy Network](https://github.com/autonomousvision/occupancy_networks)
- [stanford-shapenet-renderer](https://github.com/panmari/stanford-shapenet-renderer)

## Citations
If you use the our dataset in your research, please cite the following papers:
```bibtex
@inproceedings{ramakrishnan2021hm3d,
  title={Habitat-Matterport 3D Dataset ({HM}3D): 1000 Large-scale 3D Environments for Embodied {AI}},
  author={Santhosh Kumar Ramakrishnan and Aaron Gokaslan and Erik Wijmans and Oleksandr Maksymets and Alexander Clegg and John M Turner and Eric Undersander and Wojciech Galuba and Andrew Westbury and Angel X Chang and Manolis Savva and Yili Zhao and Dhruv Batra},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=-v4OuqNs5P}
}
@article{collins2021abo,
  title={ABO: Dataset and Benchmarks for Real-World 3D Object Understanding},
  author={Collins, Jasmine and Goel, Shubham and Luthra, Achleshwar and
          Xu, Leon and Deng, Kenan and Zhang, Xi and Yago Vicente, Tomas F and
          Arora, Himanshu and Dideriksen, Thomas and Guillaumin, Matthieu and
          Malik, Jitendra},
  journal={arXiv preprint arXiv:2110.06199},
  year={2021}
}
```


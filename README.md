# SLNR
**A Super Lightweight Neural Representation for Large-scale 3D Mapping**
<br>Chenhui Shi, Fulin Tang, Hao Wei, Yihong Wu<br>

Abstract: *We propose a new and ultra-lightweight neural representation with outstanding performance for large-scale 3D mapping. The representation defines a global signed distance function (SDF) in near-surface space based on a set of band-limited local SDFs anchored at support points sampled from point clouds. These SDFs are parameterized only by a tiny multi-layer perceptron (MLP) with no latent features, and the state of each SDF is modulated by three learnable geometric properties: position, rotation, and scaling, which make the representation adapt to complex geometries. Then, we develop a novel parallel algorithm tailored for this unordered representation to efficiently detect local SDFs where each sampled point is located, allowing for real-time updates of local SDF states during training. Additionally, a prune-and-expand strategy is introduced to enhance adaptability further. The synergy of our low-parameter model and its adaptive capabilities results in an extremely compact representation with excellent expressiveness.*

![SLNR](assets/overview.png)

## Introduce

This repository provides four core components: 
- A local SDF representation without relying on latent features;
- A parallel local SDF detection algorithm for real-time optimization;
- A prune-expand strategy to enhance the adaptability;
- Parallel point sampling along rays in a spatial hash grid.

## Install 

The code has been tested on an RTX 4090 GPU with CUDA-11.8 equipped. First, install the Python environment:

```bash
conda create --name slnr python=3.8
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install open3d pypose opencv-python scikit-image
```

Then, build the custom modules:

```bash
# Related to the spatial hash grid
cd ./modules/sparse_hash && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=your_env/lib/python3.8/site-packages/torch  ..
make -j$(nproc)
# Related to the local sdf detection and optimization
cd ./modules/gaussian_search && mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=your_env/lib/python3.8/site-packages/torch  ..
make -j$(nproc)
```

Note that in ``` ./Thirdparty/skimage``` , we provide a modified version of scikit-image. Specifically, changes have been made at lines 161 and 187 of ``` measure/_marching_cubes_lewiner.py```  to ignore regions without surfaces during the marching cubes process.

## Run

First, download the [Oxford Spires Example Data](https://drive.google.com/file/d/1y8QIgbFzWQBxyzx9anUB9N9AyKXV3XfQ/view?usp=sharing). Please refer to the [official website](https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/) for more detialed information and observe the license. Then, put the data to `./data` folder. Finally, execute the following command:

```bash
python run.py --conf=configs/example.yaml
```
The real-time optimization for basis SDF and geometric parameters is presented as below. Each support point is colored by the z-axis orientation of the local coordinate system of the defined local SDF.

![SLNR](assets/optimize.gif)

Some results are shown as below:

![SLNR](assets/results.png)

## Acknowledgement

This work was inspired by the B-spline basis function used in [PoissonRecon](https://github.com/mkazhdan/PoissonRecon), the neural point representation in [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM) and the splatting algorithm in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). We hope that this work will inspire interested parties to develop more lightweight and efficient mapping methods for robotic spatial intelligence.

## Citation
```bibtex
@article{shi2026slnmapping,
  title={SLNMapping: Super Lightweight Neural Mapping in Large-Scale Scenes},
  author={Shi, Chenhui and Tang, Fulin and Wei, Hao and Wu, Yihong},
  journal={International Journal of Computer Vision},
  volume={134},
  number={2},
  pages={68},
  year={2026},
  publisher={Springer}
}
```
```bibtex
@inproceedings{shi20253d,
  title={3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping},
  author={Shi, Chenhui and Tang, Fulin and An, Ning and Wu, Yihong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={27233--27242},
  year={2025}
}
```
If using the component of parallel point sampling along rays in a spatial hash grid, please cite:
```bibtex
@article{shi2023accurate,
  title={Accurate implicit neural mapping with more compact representation in large-scale scenes using ranging data},
  author={Shi, Chenhui and Tang, Fulin and Wu, Yihong and Jin, Xin and Ma, Gang},
  journal={IEEE Robotics and Automation Letters},
  volume={8},
  number={10},
  pages={6683--6690},
  year={2023},
  publisher={IEEE}
}
```

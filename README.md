## Installation
Clone this repository into any place you want.
```
git clone https://github.com/MillyMaMa/SSPC-MVA.git
cd SSPC-MVA
```
### Dependencies
set the conda environment.
```
conda env create --name SPC-MVA
conda activate SPC-MVA
```
Try to set the environment manually.
* Python 3.8.5
* PyTorch 1.10.1
* numpy
* h5py
* numba
* scikit-learn
* open3d
* torchsummary
* pytorch3d
* KNN-CUDA
* pykdtree
* torch_scatter
#### Pretrained model
Download from [link](https://pan.baidu.com/s/15GUblt7htrs4b_sb3TTPTA)  提取码: f5pj 

Train or test our model:
```
CUDA_VISIBLE_DEVICES=0 python main.py --experiment_id {experiment id} --dataset_name {dataset} --class_name {plane/car/chair/table} --batch_size 16
```

### Supplementary material
More experimental results of our SSPC-MVA submitted to ICASSP2025 can be found in supplementary_material_ICASSP2025_Jingjing-Lu_SSPC-MVA.pdf

### Installation
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
* PyTorch
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
### Datasets
We use the same datasets with ACL-SPC[link](https://github.com/Sangminhong/ACL-SPC_PyTorch)

### Pretrained model
come soon...

### Train or test our model
```
CUDA_VISIBLE_DEVICES=0 python main.py --experiment_id {experiment id} --dataset_name {dataset} --class_name {plane/car/chair/table}
```

# DIS-A100-4090

## Updated the environment from CUDA 10.2 to CUDA 11.8 !
Adapt to **4090** and **A100** ！

Using this environment, ISNet can run on GPUs with Ampere architecture and earlier, such as the 30 series cards, 40 series cards, A100, A10, etc. Except for the H100, which requires a CUDA 12+ environment, CUDA 11.8 currently supports the vast majority of Nvidia graphics cards.

## CUDA 11.8 Environment Configuration
### (1) Clone this repo
```
git clone https://github.com/HUANGYming/DIS-A100-4090.git
```
Go to the DIS/ISNet folder
1. Installing a Conda Environment Using a YAML File.
```
conda env create -f environment_cu118.yaml
```
2. Installing a Conda Environment Using a TXT File.
```
pip install requirements_cu118.txt
```


### (2) Only Download Configuration files
Go to the DIS/ISNet folder

Download ```environment_cu118.yaml``` or ```requirements_cu118.txt```, then install by Conda or pip.


### (3) Creating a Conda Environment from a Compressed Package.
In addition to installation from pip and conda sources, I have provided a conda environment compressed package. You can directly unzip it in the conda environment for use.
1. Download compressed package from onedrive
```
conda env create -f environment_cu118.yaml
```
2. Find the conda environment directory
```
conda info --envs
``
3. Enter into the conda environment directory
```
cd /path/to/file/envs/
```
3. Unzip the environment compressed package
```
unzip xxx
```

## Contact
If you have any questions or suggestions, please contact me at the following email: huangym2@connect.hku.hk

## Acknowledgements
Special thanks to the members of the ISNet project team, whose hard work and innovative thinking laid the groundwork for the success of this project.

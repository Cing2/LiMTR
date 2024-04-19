# LiMTR code base

This is the official repo of "LiMTR: Using local LiDAR features for Road Users Motion Prediction" paper.

This is a fork of the [MTR](https://github.com/sshaoshuai/MTR) code base.

The code base use the structure from [link](https://github.com/ashleve/lightning-hydra-template) for their pytorch ligthning and hydra setup, this allows for easy experimenting

We are unable to provide the pre-trained model weights due to [Waymo Dataset License Agreement](https://waymo.com/open/terms/). 

## Installation

Install requirements.

```bash
pip install --upgrade pip --user

# install waymo package
pip install waymo-open-dataset-tf-2-11-0==1.6.0
# install torch and cuda
pip install torch --index-url https://download.pytorch.org/whl/cu118
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# optional also gcc for compiling the source code
conda install -c conda-forge gxx=11

pip install -r requirements.txt --user
```

## Data download

Use the following script to download Waymo dataset.
This requires gsutil to be installed.

```bash
# for help
python src/tools/download_data.py -h
# limit number of .tfrecords
python src/tools/download_data.py /path/to/download/to -l 1
# to download lidar files
python src/tools/download_data.py /path/to/download/to  --lidar
```

## Compiling MTR

Compile the customized CUDA codes in MTR codebase

```bash
export CXX=g++-11
# new install with pip
cd src
pip install -e .

# optional select architecture to compile to
export TORCH_CUDA_ARCH_LIST="Ampere" # architecture of A100
export TORCH_CUDA_ARCH_LIST="Volta" # architecture of V100
export FORCE_CUDA=1
```

## Data preprocessing

Use the following script to preprocess the data:

```bash
python src/mtr/datasets/waymo/data_preprocess_tar.py /path/data/data/waymo/scenario/  /path/data/data/waymo  -l 1 -n 18
```

# Training the model

Entry point of program is src/trainLight.py.

To train the model run:

```bash
python src/trainLight.py

# you can set arguments in command line, like
python src/trainLight.py debug=default data.NUM_FILES.train=200

# to run experiment
python src/trainLight.py experiment=lidar

# To run on the server you can use the scripts, and use same argument option
sbatch scripts/run_slurm_1gpu.sh experiment=lidar
sbatch scripts/run_slurm_1n.sh debug=1epoch
```

We also have a batch size finder, that progressively finds a batch size that fits on the gpu.

```bash
sbatch scripts/run_slurm_1gpu.sh +batch_size_finder=true trainer=default

# or for other experiment setup
sbatch scripts/run_slurm_1gpu.sh +batch_size_finder=true trainer=default experiment=lidar

```

To run the Learning rate finder, run:

```bash
sbatch scripts/run_slurm_1gpu.sh +lr_finder=true trainer=default

# or for other experiment setup
sbatch scripts/run_slurm_1gpu.sh +lr_finder=true  experiment=lidar
```


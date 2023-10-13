# Welcome to OG-MARL
Off-The-Grid MARL is a research framework Cooperative Offline Multi-Agent Reinforcement Learning (MARL). Our code includes implementations of utilities for generating your own datasets, popular offline MARL algorithms, and tools for evaluating you algorithms performance. To get started, follow the instructions in this README which walk you through the instalation and how to run the quickstart tutorials.

# Using Conda
Because we support many different environments, each with their own set of dependencies which are often conflicting, you will need to follow slightly different instalation instruction for each environment. 

To manage the different dependencies, we reccomend using `miniconda` as a python virtual environment manager. Follow these instructions to install `conda`. 

* https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

We have tested these instalation instructions on Ubuntu. Unfortunatly they are unlikely to work on Windows and Mac because of our dependency on DeepMind's `reverb` and `launchpad` packages. In future we hope to relax these requirements.

In future we will also be releasing Dockerfiles.

# Overview
In the `examples/` directory we include a quickstart tutorial which includes 3 files:
* `examples/quickstart_double_cartpole.py`
* `examples/quickstart_generate_dataset.py`
* `examples/quickstart_train_offline_algo.py`

We also include scripts for replicating our benchmarking results:
* `examples/benchmark_mamujoco.py`
* `examples/benchmark_smac.py`


# Installation
You will need to follow slightly different instructions for the `quickstart`, `smac benchmark` and `mamujoco benchmark`. We reccomend creating a different `conda` environment for each and then switching between them as is neccessary by using the comand

`conda activate <conda_env_name>`

## Quickstart Instructions
First, create a conda environment for the quickstart tutorial.

`conda create --name og-marl-quickstart python=3.8`

Activate the conda environment.

`conda activate og-marl-quickstart`

Now install the core requirements.

`pip install -r requirements.txt`

Finally, install OG-MARL.

`pip install -e .`

You are now ready to run through the quickstart tutorial. Open the file `examples/quickstart_generate_dataset.py` and read the comments throughout to do the tutorial.

## SMAC Instructions
Inorder to run the SMAC benchmarking script `examples/benchmark_smac.py` you need to follow all the steps above and then as a final step run the SMAC instalation script:

`bash install_environments/smac.sh`

## MAMuJoCo Instructions
Inorder to run the MAMuJoCo benchmarking script `examples/benchmark_mamujoco.py` you need to follow all the steps in the quickstart instructions, and then as a final step run the MAMuJoCo instalation script:

`bash install_environments/mamujoco.sh`

Don't worry if you see the following error. 

`ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.8.4 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
id-mava 0.1.3 requires numpy~=1.21.4, but you have numpy 1.24.1 which is incompatible.`

 IMPORTANT!!!! You will need to set these environment variables everytime you start a new terminal or add them to your .bashrc file.

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/:$MUJOCOPATH/mujoco210/bin:/usr/lib/nvidia`

`export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so`


# Downloading Datasets
You will need to make sure you download the datasets from the OG-MARL website.
https://sites.google.com/view/og-marl

Make sure the unzip the dataset and add it to the path 
* `datasets/smac/<env_name>/<dataset_quality>/` for SMAC datasets.
* `datasets/mamujoco/<env_name>/<dataset_quality>/` for MAMuJoCo datasets.

The following error means the code did not find the dataset where it was looking for it. Check that you downloaded the dataset, unzipped it and put it at the right path.

`TypeError: The `filenames` argument must contain `tf.string` elements. Got `tf.float32` elements.`
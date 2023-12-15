# MADiff: Offline Multi-agent Learning with Diffusion Models

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.17330)

This is the official implementation of "MADiff: Offline Multi-agent Learning with Diffusion Models".

![MADiff](/assets/images/madiff.png)

## Performances

### Multi-agent Particle Environment (MPE)

The peformances on MPE datasets from [OMAR paper](https://arxiv.org/abs/2111.11188).

| Dataset | Task | BC | MA-ICQ | MA-TD3+BC | MA-CQL | OMAR | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Expert | Spread | 35.0 ± 2.6 | 104.0 ± 3.4 | 108.3 ± 3.3 | 98.2 ± 5.2 | 114.9 ± 2.6 | 97.0 ± 4.9 | 116.0 ± 3.5 |
| Expert | Tag | 40.0 ± 9.6 | 113.0 ± 14.4 | 115.2 ± 12.3 | 93.9 ± 14.0 | 116.2 ± 19.8 | TBD | 168.3 ± 19.0 |
| Expert | World | 33.0 ± 9.9 | 109.5 ± 22.8 | 110.3 ± 21.3 | 71.9 ± 21.3 | 110.4 ± 25.7 | 115.4 ± 11.1 | 178.9 ± 21.7 |
| Md-Replay | Spread | 10.0 ± 3.8 | 13.6 ± 5.7 | 15.4 ± 5.6 | 20.0 ± 8.4 | 37.9 ± 12.3 | TBD | 43.1 ± 9.8 |
| Md-Replay | Tag | 0.9 ± 1.4 | 34.5 ± 27.8 | 28.7 ± 20.9 | 24.8 ± 17.3 | 47.1 ± 15.3 | 63.0 ± 6.8 | 98.8 ± 11.3 |
| Md-Replay | World | 2.3 ± 1.5 | 12.0 ± 9.1 | 17.4 ± 8.1 | 29.6 ± 13.8 | 42.9 ± 19.5 | 60.3 ± 10.3 | 84.9 ± 3.9 |
| Medium | Spread | 31.6 ± 4.8 | 29.3 ± 5.5 | 29.3 ± 4.8 | 34.1 ± 7.2 | 47.9 ± 18.9 | 64.7 ± 9.6 | 58.0 ± 2.4 |
| Medium | Tag | 22.5 ± 1.8 | 63.3 ± 20.0 | 65.1 ± 29.5 | 61.7 ± 23.1 | 66.7 ± 23.2 | 78.3 ± 7.6 | 133.5 ± 20.2 |
| Medium | World | 25.3 ± 2.0 | 71.9 ± 20.0 | 73.4 ± 9.3 | 58.6 ± 11.2 | 74.6 ± 11.5 | 124.2 ± 6.3 | 157.1 ± 6.8 |
| Random | Spread | -0.5 ± 3.2 | 6.3 ± 3.5 | 9.8 ± 4.9 | 24.0 ± 9.8 | 34.4 ± 5.3 | TBD | TBD |
| Random | Tag | 1.2 ± 0.8 | 2.2 ± 2.6 | 5.7 ± 3.5 | 5.0 ± 8.2 | 11.1 ± 2.8 | TBD | TBD |
| Random | World | -2.4 ± 0.5 | 1.0 ± 3.2 | 2.8 ± 5.5 | 0.6 ± 2.0 | 5.9 ± 5.2 | TBD | TBD |

### Multi-agent Mujoco (MA-Mujoco)

The peformances on MA-Mujoco datasets from [off-the-grid MARL benchmark](https://arxiv.org/abs/2302.00521).

| Dataset | Task | BC | MA-TD3+BC | OMAR | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Good | 2halfcheetah | 6846 ± 574 | 7025 ± 439 | 1434 ± 1903 | **8254 ± 179** | 8662 ± 102 |
| Medium | 2halfcheetah | 1627 ± 187 | **2561 ± 82** | 1892 ± 220 | 2215 ± 27 | 2221 ± 56 |
| Poor | 2halfcheetah | 465 ± 59 | 736 ± 72 | 384 ± 420 | **751 ± 74** | 767 ± 42 |
| Good | 4ant | 2802 ± 133 | 2628 ± 971 | 344 ± 631 | **3090 ± 26** | 3087 ± 32 |
| Medium | 4ant | 1617 ± 153 | **1843 ± 494** | 929 ± 349 | 1679 ± 93 | 1897 ± 44 |
| Poor | 4ant | 1033 ± 122 | 1075 ± 96 | 518 ± 112 | **1268 ± 51** | 1332 ± 45 |

### StarCraft Multi-Agent Challenge (SMAC)

The peformances on SMAC datasets from [off-the-grid MARL benchmark](https://arxiv.org/abs/2302.00521).

| Dataset | Task | BC | QMIX | MA-ICQ | MA-CQL | MADT | MADiff-D | MADiff-C* |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Good | 3m | 16.0 ± 1.0 | 13.8 ± 4.5 | 18.8 ± 0.6 | **19.6 ± 0.3** | 19.0 ± 0.3 | **19.6 ± 0.7** | 20.0 ± 0.0 |
| Medium | 3m | 8.2 ± 0.8 | 17.3 ± 0.9 | 18.1 ± 0.7 | **18.9 ± 0.7** | 15.8 ± 0.5 | 17.2 ± 0.7 | 18.0 ± 0.7 | 
| Poor | 3m | 4.4 ± 0.1 | 10.0 ± 2.9 | **14.4 ± 1.2** | 5.8 ± 0.4 | 4.2 ± 0.1 | 8.9 ± 0.1 | 9.3 ± 0.6 | 
| Good | 5m6m | 16.6 ± 0.6 | 8.0 ± 0.5 | 16.3 ± 0.9 | 13.8 ± 3.1 | 16.8 ± 0.1 | **18.0 ± 1.0** | 18.2 ± 1.0 | 
| Medium | 5m6m | 12.4 ± 0.9 | 12.0 ± 1.1 | 15.3 ± 0.7 | 17.0 ± 1.2 | 16.1 ± 0.2 | TBD | 18.0 ± 1.1 | 
| Poor | 5m6m | 7.5 ± 0.2 | **10.7 ± 0.9** | 9.4 ± 0.4 | 10.4 ± 1.0 | 7.6 ± 0.3 | 8.9 ± 0.3 | 9.5 ± 0.7 |
| Good | 8m | 16.7 ± 0.4 | 4.6 ± 2.8 | **19.6 ± 0.3** | 11.3 ± 6.1 | 18.5 ± 0.4 | 19.2 ± 0.1 | 20.0 ± 0.0 | 
| Medium | 8m | 10.7 ± 0.5 | 13.9 ± 1.6 | 18.6 ± 0.5 | 16.8 ± 3.1 | 18.2 ± 0.1 | **19.2 ± 0.7** | 19.5 ± 0.9 | 
| Poor | 8m | 5.3 ± 0.1 | 6.0 ± 1.3 | **10.8 ± 0.8** | 4.6 ± 2.4 | 4.8 ± 0.1 | 5.1 ± 0.1 | 5.2 ± 0.1 |


## Setup

### Installation

```bash
sudo apt-get update
sudo apt-get install libssl-dev libcurl4-openssl-dev swig
conda create -n madiff python=3.8
conda activate madiff
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Setup MPE

We use the MPE dataset from [OMAR](https://github.com/ling-pan/OMAR). The dataset download link and instructions can be found in OMAR's [repo](https://github.com/ling-pan/OMAR). Since their BaiduPan download links might be inconvenient for non-Chinese users, we maintain a anonymous mirror [repo](https://osf.io/jxawh/?view_only=dd3264a695af4c03bffde0350b8e8c4a) in OSF for acquiring the dataset.

The downloaded dataset should be placed under `diffuser/datasets/data/mpe`.

Install MPE environment:

```bash
pip install -e third_party/multiagent-particle-envs
pip install -e third_party/ddpg-agent
```

### Setup MA-Mujoco

1. Install MA-Mujoco:

    ```bash
    pip install -e third_party/multiagent_mujoco
    ```

2. We use the MA-Mujoco dataset from [off-the-grid MARL](https://sites.google.com/view/og-marl). We preprocess the dataset to concatenate trajectories to full episodes and save them as `.npy` files for easier loading. The original dataset can be downloaded from links below.

+ [2halfcheetah](https://s3.kao.instadeep.io/offline-marl-dataset/2halfcheetah.zip)

+ [4ant](https://s3.kao.instadeep.io/offline-marl-dataset/4ant.zip)

    The downloaded dataset should be placed under `diffuser/datasets/data/mamujoco`.

3. Install off-the-grid MARL and transform the original dataset.

    ```bash
    pip install -r ./third_party/og_marl/requirements.txt
    pip install -e ./third_party/og_marl
    python scripts/transform_og_marl_dataset.py --env_name mamujoco --map_name <map> --quality <dataset>
    ```

### Setup SMAC

1. Run `scripts/smac.sh` to install *StarCraftII*.

2. Install SMAC:

    ```bash
    pip install git+https://github.com/oxwhirl/smac.git
    ```

3. We use the SMAC dataset from [off-the-grid MARL](https://sites.google.com/view/og-marl). We preprocess the dataset to concatenate trajectories to full episodes and save them as `.npy` files for easier loading. The original dataset can be downloaded from links below.

+ [3m](https://s3.kao.instadeep.io/offline-marl-dataset/3m.zip)

+ [5m_vs_6m](https://s3.kao.instadeep.io/offline-marl-dataset/5m_vs_6m.zip)

+ [8m](https://s3.kao.instadeep.io/offline-marl-dataset/8m.zip)

    The downloaded dataset should be placed under `diffuser/datasets/data/smac`.

4. Install off-the-grid MARL and transform the original dataset.

    ```bash
    pip install -r ./third_party/og_marl/requirements.txt
    pip install -e ./third_party/og_marl
    python scripts/transform_og_marl_dataset.py --env_name smac --map_name <map> --quality <dataset>
    ```

## Training and Evaluation
To start training, run the following commands

```bash
# multi-agent particle environment
python run_experiment.py -e exp_specs/mpe/mad_mpe_<task>_attn_<dataset>.yaml  # CTCE
python run_experiment.py -e exp_specs/mpe/mad_mpe_<task>_ctde_<dataset>.yaml  # CTDE
# ma-mujoco
python run_experiment.py -e exp_specs/mamujoco/mad_mamujoco_<map>_attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/mamujoco/mad_mamujoco_<map>_ctde_<dataset>_history.yaml  # CTDE
# smac
python run_experiment.py -e exp_specs/smac/mad_smac_<map>_attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/smac/mad_smac_<map>_ctde_<dataset>_history.yaml  # CTDE
```

To evaluate the trained model, first replace the `log_dir` with those need to be evaluated in `exp_specs/eval_inv.yaml` and run
```bash
python run_experiment.py -e exp_specs/eval_inv.yaml
```

## Citation

```
@article{zhu2023madiff,
  title={MADiff: Offline Multi-agent Learning with Diffusion Models},
  author={Zhu, Zhengbang and Liu, Minghuan and Mao, Liyuan and Kang, Bingyi and Xu, Minkai and Yu, Yong and Ermon, Stefano and Zhang, Weinan},
  journal={arXiv preprint arXiv:2305.17330},
  year={2023}
}
```

## Acknowledgements

The codebase is built upon [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser).

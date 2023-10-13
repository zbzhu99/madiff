# MADiff: Offline Multi-agent Learning with Diffusion Models

This is the official implementation of MADiff: Offline Multi-agent Learning with Diffusion Models.

## Installation

```bash
sudo apt-get update
sudo apt-get install libssl-dev libcurl4-openssl-dev swig
conda create -n madiff python=3.8
conda activate madiff
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Setup MPE

We use the MPE dataset from [OMAR](https://github.com/ling-pan/OMAR). The dataset download link and instructions can be found in their GitHub repo. Since their BaiduPan download links might be inconvenient for non-Chinese users, we maintain a anonymous mirror [repo](https://osf.io/jxawh/?view_only=dd3264a695af4c03bffde0350b8e8c4a) in OSF for acquiring the dataset.

The downloaded dataset should be placed under `diffuser/datasets/data/mpe`.

## Setup SMAC

1. Run `scripts/smac.sh` to install *StarCraftII*.

2. Install SMAC:

    ```bash
    pip install git+https://github.com/oxwhirl/smac.git
    ```

3. We use the SMAC dataset from [off-the-grid MARL](https://sites.google.com/view/og-marl). The `3m` and `5m_vs_6m` dataset is required for reproducing the results in our paper. We preprocess the dataset to concatenate trajectories to full episodes and save them as `.npy` files for easier loading. The original dataset can be downloaded from links below.

+ [3m](https://www.google.com/url?q=https%3A%2F%2Ftinyurl.com%2F3m-dataset&sa=D&sntz=1&usg=AOvVaw2Lqtkshe3Cf-9oU_28R32H)

+ [5m_vs_6m](https://www.google.com/url?q=https%3A%2F%2Ftinyurl.com%2F5m-vs-6m-dataset&sa=D&sntz=1&usg=AOvVaw0l2xBkPAe2rwuqlAMfRHCB)

4. Install off-the-grid MARL and transform the original dataset.

    ```bash
    pip install -r ./third_party/og_marl/requirements.txt
    pip install -e ./third_party/og_marl
    python scripts/transform_smac_dataset_to_npy.py
    ```

## Training and Evaluation
To start training, run the following commands

```bash
# multi-agent particle environment
python run_experiment.py -e exp_specs/mpe/mad_mpe_<task>_attn_<dataset>.yaml  # CTCE
python run_experiment.py -e exp_specs/mpe/mad_mpe_<task>_ctde_<dataset>.yaml  # CTDE
# smac
python run_experiment.py -e exp_specs/smac/mad_smac_<map>_attn_<dataset>_history.yaml  # CTCE
python run_experiment.py -e exp_specs/smac/mad_smac_<map>_ctde_<dataset>_history.yaml  # CTDE
```

To evaluate the trained model, first replace the `log_dir` with those need to be evaluated in `exp_specs/eval_inv.yaml` and run
```bash
python run_experiment.py -e exp_specs/eval_inv.yaml
```

## Acknowledgements

The codebase is built upon [decision-diffuser repo](https://github.com/anuragajay/decision-diffuser).

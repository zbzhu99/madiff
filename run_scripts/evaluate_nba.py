import argparse
import os

import yaml

import diffuser.utils as utils
from diffuser.utils.launcher_util import build_config_from_dict


def evaluate(Config):
    evaluator = None

    for load_step in Config.load_steps:
        ckpt_file_path = os.path.join(
            Config.log_dir, f"checkpoint/state_{load_step}.pt"
        )
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue

        results_file_path = os.path.join(
            Config.log_dir, f"results/step_{load_step}.json"
        )
        if not Config.overwrite and os.path.exists(results_file_path):
            print(
                f"Results file {results_file_path} already exist. Skipping evaluation."
            )
            continue

        if evaluator is None:
            evaluator_config = utils.Config(Config.evaluator, verbose=True)
            evaluator = evaluator_config()
            evaluator.init(
                log_dir=Config.log_dir,
                horizon=Config.horizon,
                use_history=Config.use_history,
                nba_hz=Config.nba_hz,
                history_horizon=Config.history_horizon,
                batch_size=Config.batch_size,
                batch_idx=Config.batch_idx,
            )

        evaluator.evaluate(load_step=load_step, sample_times=Config.sample_times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.SafeLoader)
    Config = build_config_from_dict(exp_specs)

    evaluate(Config)

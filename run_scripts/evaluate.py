import argparse
import os

import diffuser.utils as utils
import yaml
from diffuser.utils.launcher_util import build_config_from_dict


def evaluate(Config):
    evaluator = None
    Config.condition_guidance_w = getattr(Config, "condition_guidance_w", None)

    for load_step in Config.load_steps:
        ckpt_file_path = os.path.join(
            Config.log_dir, f"checkpoint/state_{load_step}.pt"
        )
        if not os.path.exists(ckpt_file_path):
            print(f"Checkpoint file {ckpt_file_path} not found. Skipping evaluation.")
            continue

        results_file_path = os.path.join(
            Config.log_dir,
            f"results/step_{load_step}-ep_{Config.num_eval}-ddim.json"
            if getattr(Config, "use_ddim_sample", False)
            else f"results/step_{load_step}-ep_{Config.num_eval}.json",
        )
        if Config.condition_guidance_w is not None:
            results_file_path = results_file_path.replace(
                ".json", f"-cg_{Config.condition_guidance_w}.json"
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
                num_eval=Config.num_eval,
                num_envs=getattr(Config, "num_envs", Config.num_eval),
                condition_guidance_w=Config.condition_guidance_w,
                use_ddim_sample=Config.use_ddim_sample,
                n_ddim_steps=Config.n_ddim_steps,
            )

        evaluator.evaluate(load_step=load_step)


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

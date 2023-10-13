import matplotlib.pyplot as plt
import numpy as np
from ml_logger import logger

from diffuser.datasets.mpe import load_environment

from .arrays import to_np
from .video import save_video, save_videos

# -----------------------------------------------------------------------------#
# ------------------------------ helper functions -----------------------------#
# -----------------------------------------------------------------------------#


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def atmost_3d(x):
    while x.ndim > 3:
        x = x.squeeze(0)
    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- renderers --------------------------------#
# -----------------------------------------------------------------------------#


class MPERenderer:
    """
    default mpe renderer
    """

    def __init__(self, env_type, env):
        if type(env) is str:
            self.env = load_environment(env, render_mode="rgb_array")
            self.env.reset()
        else:
            self.env = env

    def render(self, observation):
        set_state(self.env, observation)
        data = self.env.render()
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, return_sample_images=False, **kwargs):
        sample_images = self._renders(samples, **kwargs)
        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        if return_sample_images:
            return composite, sample_images
        else:
            return composite

    def composite(self, savepath, paths, **kwargs):
        composite_images, sample_images = [], []
        for path in paths:
            # [ H x agent x obs_dim ]
            path = atmost_3d(path)
            composite_img, sample_img = self.renders(
                to_np(path), return_sample_images=True, **kwargs
            )
            composite_images.append(composite_img)
            sample_images.append(sample_img)
        composite_images = np.concatenate(composite_images, axis=0)
        sample_images = np.concatenate(sample_images, axis=1)

        if savepath is not None:
            fig = plt.figure()
            plt.imshow(composite_images)
            logger.savefig(savepath, fig)
            plt.close()
            print(f"Saved {len(paths)} samples to: {savepath}")
            logger.save_video(
                sample_images,
                savepath.replace(".png", ".mp4"),
                macro_block_size=4,
                fps=2,
            )
            print(
                f"Saved {len(paths)} samples video to: {savepath.replace('.png', '.mp4')}"
            )

        return composite_images

    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list:
            states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        raise NotImplementedError

        # [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)

        # there will be one more state in `observations_real`
        # than in `observations_pred` because the last action
        # does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:, :-1]

        images_pred = np.stack(
            [self._renders(obs_pred, partial=True) for obs_pred in observations_pred]
        )

        images_real = np.stack(
            [self._renders(obs_real, partial=False) for obs_real in observations_real]
        )

        # [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)

    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        """
        diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        """

        raise NotImplementedError

        diffusion_path = to_np(diffusion_path)

        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f"[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}")

            # [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[
                :, :, : self.observation_dim
            ]

            frame = []
            for states in states_l:
                img = self.composite(
                    None,
                    states,
                    dim=(1024, 256),
                    partial=True,
                    qvel=True,
                    render_kwargs=render_kwargs,
                )
                frame.append(img)
            frame = np.concatenate(frame, axis=0)

            frames.append(frame)

        save_video(savepath, frames, **video_kwargs)

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


# -----------------------------------------------------------------------------#
# ---------------------------------- rollouts ---------------------------------#
# -----------------------------------------------------------------------------#


def set_state(env, state):
    # FIXME: this is a hack
    if env.metadata["name"] in ["simple_tag", "simple_world"]:
        agents = env.world.agents[:-1]
    else:
        agents = env.world.agents

    for idx, agent in enumerate(agents):
        agent.state.p_vel = state[idx, :2]
        agent.state.p_pos = state[idx, 2:4]

    for idx, landmark in enumerate(env.world.landmarks):
        # use the first agent to recover the absolute position of landmarks
        landmark.state.p_pos = state[0, 4 + 2 * idx : 4 + 2 * idx + 2] + state[0, 2:4]


def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack(
        [rollout_from_state(env, state, actions) for actions in actions_l]
    )
    return rollouts


def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions) + 1):
        # if terminated early, pad with zeros
        observations.append(np.zeros(obs.size))
    return np.stack(observations)

from typing import List

import numpy as np


def atleast_nd(x, n: int):
    while x.ndim < n:
        x = np.expand_dims(x, axis=-1)
    return x


class ReplayBuffer:
    def __init__(
        self,
        n_agents: int,
        max_n_episodes: int,
        max_path_length: int,
        termination_penalty: float,
        global_feats: List[str] = ["states"],
        use_zero_padding: bool = True,
    ):
        self._dict = {
            "path_lengths": np.zeros(max_n_episodes, dtype=int),
        }
        self._count = 0
        self.n_agents = n_agents
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.global_feats = global_feats
        self.use_zero_padding = use_zero_padding

    def __repr__(self):
        return "[ datasets/buffer ] Fields:\n" + "\n".join(
            f"    {key}: {val.shape}" for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self["path_lengths"])

    def _add_keys(self, path):
        if hasattr(self, "keys"):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        """
        can access fields with `buffer.observations`
        instead of `buffer['observations']`
        """
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items() if k != "path_lengths"}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        if len(array.shape) == 3:
            shape = (self.max_n_episodes, self.max_path_length, self.n_agents, dim)
        else:
            assert len(array.shape) == 2, f"Invalid shape {array.shape} of {key}"
            shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)

    def add_path(self, path):
        # path[key] shape: (path_length, n_agents, dim)
        path_length = len(path["observations"])
        assert path_length <= self.max_path_length

        # NOTE(zbzhu): agents must terminate together
        all_terminals = np.any(path["terminals"], axis=1)
        if all_terminals.any():
            assert (bool(all_terminals[-1]) is True) and (not all_terminals[:-1].any())

        # if first path added, set keys based on contents
        self._add_keys(path)

        # add tracked keys in path
        for key in self.keys:
            if key in self.global_feats:  # all agents share the same global state
                array = atleast_nd(path[key], n=2)
            else:
                array = atleast_nd(path[key], n=3)
            if key not in self._dict:
                self._allocate(key, array)
            if not self.use_zero_padding and key not in ["rewards"]:
                self._dict[key][self._count] = array[-1]
            self._dict[key][self._count, :path_length] = array

        # penalize early termination
        if all_terminals.any() and self.termination_penalty is not None:
            if "timeouts" in path:
                assert not path[
                    "timeouts"
                ].any(), "Penalized a timeout episode for early termination"
            self._dict["rewards"][
                self._count, path_length - 1
            ] += self.termination_penalty

        # record path length
        self._dict["path_lengths"][self._count] = path_length

        # increment path counter
        self._count += 1

    def truncate_path(self, path_ind, step):
        old = self._dict["path_lengths"][path_ind]
        new = min(step, old)
        self._dict["path_lengths"][path_ind] = new

    def finalize(self):
        # remove extra slots
        for key in self.keys + ["path_lengths"]:
            self._dict[key] = self._dict[key][: self._count]
        self._add_attributes()
        print(f"[ datasets/buffer ] Finalized replay buffer | {self._count} episodes")

import os
import json
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "data/nba/assets")
GAMES_DIR = os.path.join(os.path.dirname(__file__), "data/nba/baller2vec_data/games")
RAW_DATA_HZ = 25


# TODO: wrap sequence dataset in nba_env so that when it can directly return a batch when evaluating
class NBAEnv:
    def __init__(self, settings, data_split, horizon=64):
        self.metadata = settings
        self.pres_game_idx = 0
        self.pres_timestep = 0
        self.horizon = horizon
        with open(os.path.join(ASSETS_DIR, f"{data_split}_gameids.txt"), "r") as f:
            self.gameids = f.read().split()

    def step(self, action):
        self.pres_timestep += 1
        return (
            self.pres_observation[
                self.pres_timestep : self.pres_timestep + self.horizon
            ],
            self.pres_rewards[self.pres_timestep : self.pres_timestep + self.horizon],
            self.pres_done[self.pres_timestep : self.pres_timestep + self.horizon],
            {},
        )

    def reset(self):
        self.pres_episode = gen_episode_from_id(
            self.gameids[self.pres_game_idx], self.metadata
        )
        self.pres_game_idx = np.random.randint(len(self.gameids))
        self.pres_timestep = 0
        self.pres_observation = np.concatenate(
            [
                self.pres_episode["player_idxs"],
                self.pres_episode["observations"],
                self.pres_episode["player_hoop_sides"],
            ],
            axis=-1,
        )
        self.pres_done = self.pres_episode["terminals"]
        self.pres_rewards = self.pres_episode["rewards"]
        return self.pres_observation[0 : self.horizon]


def make_env(data_split, benchmark=False, **kwargs):
    # load scenario from script
    settings_path = os.path.join(os.path.dirname(__file__), "nba/settings.json")
    with open(settings_path, "r") as f:
        settings = json.load(f)
    return NBAEnv(settings, data_split)


def load_environment(name, **kwargs):
    if type(name) != str:
        # name is already an environment
        return name

    data_split = name
    env = make_env(data_split, **kwargs)
    if hasattr(env, "metadata"):
        assert isinstance(env.metadata, dict)
    else:
        env.metadata = {}
    env.metadata["mode"] = data_split
    env.metadata["HZ"] = kwargs["nba_hz"]
    return env


def gen_episode_from_id(gameid, metadata):
    seed_path = os.path.join(GAMES_DIR, f"{gameid}_X.npy")
    if not os.path.exists(seed_path):
        raise Warning("Data directory not found: {}".format(seed_path))

    X = np.load(seed_path)
    skip = int(RAW_DATA_HZ / metadata["HZ"])
    seq_data = X[0:-1:skip]
    keep_players = np.random.choice(np.arange(10), 10, False)
    if metadata["mode"] in {"valid", "test"}:
        keep_players.sort()

    # prepocessing
    player_xs = seq_data[:, 20:30][:, keep_players]
    player_ys = seq_data[:, 30:40][:, keep_players]
    player_x_diffs = np.diff(player_xs, axis=0)
    player_y_diffs = np.diff(player_ys, axis=0)

    try:
        glitch_x_break = np.where(
            np.abs(player_x_diffs) > 1.2 * metadata["MAX_PLAYER_MOVE"]
        )[0].min()
    except ValueError:
        glitch_x_break = len(seq_data)

    try:
        glitch_y_break = np.where(
            np.abs(player_y_diffs) > 1.2 * metadata["MAX_PLAYER_MOVE"]
        )[0].min()
    except ValueError:
        glitch_y_break = len(seq_data)

    seq_break = min(glitch_x_break, glitch_y_break)
    # unused valid
    valid = True
    if seq_break < len(seq_data):
        seq_break = len(seq_data)
        valid = False

    seq_data = seq_data[:seq_break]

    player_idxs = np.expand_dims(
        seq_data[:, 10:20][:, keep_players].astype(int), axis=-1
    )
    player_xs = np.expand_dims(seq_data[:, 20:30][:, keep_players], axis=-1)
    player_ys = np.expand_dims(seq_data[:, 30:40][:, keep_players], axis=-1)
    player_hoop_sides = np.expand_dims(
        seq_data[:, 40:50][:, keep_players].astype(int), axis=-1
    )

    # Randomly rotate the court because the hoop direction is arbitrary.
    if metadata["mode"] == "train" and np.random.random() < 0.5:
        player_xs = metadata["COURT_LENGTH"] - player_xs
        player_ys = metadata["COURT_WIDTH"] - player_ys
        player_hoop_sides = (player_hoop_sides + 1) % 2

    observations = np.concatenate([player_xs, player_ys], axis=-1)
    dones = np.zeros((player_idxs.shape[0], player_idxs.shape[1], 1))
    dones[-1, :] = 1.0

    max_timestep = min(metadata["MAX_TIMESTEP"], len(player_idxs))
    episode_data = {}
    episode_data["player_idxs"] = player_idxs[:max_timestep]
    episode_data["observations"] = observations[:max_timestep]
    episode_data["player_hoop_sides"] = player_hoop_sides[:max_timestep]
    episode_data["terminals"] = dones[:max_timestep]
    episode_data["rewards"] = np.zeros_like(episode_data["terminals"])

    timeouts = np.zeros_like(episode_data["terminals"])
    if max_timestep == metadata["MAX_TIMESTEP"]:
        timeouts[-1][:] = 1.0
    episode_data["timeouts"] = timeouts

    return episode_data


def sequence_dataset(env, preprocess_fn, mode):
    """
    Returns an iterator through trajectories.
    Args:
        env: An MultiAgentEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """

    assert env.metadata["mode"] in ["train", "test", "valid"]
    with open(
        os.path.join(ASSETS_DIR, f"{env.metadata['mode']}_gameids.txt"), "r"
    ) as f:
        gameids = f.read().split()

    for idx, gameid in enumerate(gameids):
        episode_data = gen_episode_from_id(gameid, env.metadata)
        yield episode_data

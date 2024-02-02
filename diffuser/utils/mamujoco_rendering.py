# TODO: Finish this
class MAMuJoCoRenderer:
    """
    default mamujoco renderer
    """

    def __init__(self, env_type, env):
        pass

    def render(self, observation):
        pass

    def _renders(self, observations, **kwargs):
        pass

    def renders(self, samples, return_sample_images=False, **kwargs):
        pass

    def composite(self, savepath, paths, **kwargs):
        pass

    def render_rollout(self, savepath, states, **video_kwargs):
        pass

    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        pass

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)

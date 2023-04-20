from gym.envs.registration import register

register(
    id='EpMineEnv-v0',
    entry_point='envs.SingleAgent:EpMineEnv',
    max_episode_steps=1800
)
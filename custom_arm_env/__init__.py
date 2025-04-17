from gymnasium.envs.registration import register

register(
    id="CustomArmPick-v0",
    entry_point="custom_arm_env.custom_arm_env:CustomArmEnv",
    max_episode_steps=500,
)

import gymnasium as gym
from stable_baselines3 import PPO
import custom_arm_env  # Import our custom environment

# Load the trained model
model = PPO.load("RLARM_a2c")

# Override the max steps per episode
env = gym.make("CustomArmPick-v0", render_mode="human")
env.spec.max_episode_steps = 2000  # Increase max steps

obs, info = env.reset()

for _ in range(2000):  # Run for 2000 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

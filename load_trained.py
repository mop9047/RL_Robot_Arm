import gymnasium as gym
from stable_baselines3 import PPO,A2C,SAC
import custom_arm_env  # Import our custom environment

algo = input('Which Algorith do you want to train? \n 1: A2C \n 2:SAC ')
if int(algo) == 1:
    model = A2C.load("models/RLARM_a2c")
elif int(algo) == 2:
    model = SAC.load("models/RLARM_sac")
else:
    model = A2C.load("models/RLARM_a2c")

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

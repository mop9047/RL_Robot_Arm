import gymnasium as gym
from stable_baselines3 import A2C
import custom_arm_env  # Import your custom environment

# Create the environment
env = gym.make("CustomArmPick-v0")

# Load the trained model if it exists, otherwise start fresh
model_path = "RLARM_a2c.zip"

try:
    model = A2C.load(model_path, env=env)
    print("Loaded existing model from", model_path)
except FileNotFoundError:
    print("No existing model found. Training from scratch.")
    model = A2C("MlpPolicy", env, verbose=1)

# Continue training
model.learn(total_timesteps=300000)

# Save the updated model
model.save(model_path)
env.close()
print("\a")
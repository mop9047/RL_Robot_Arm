import gymnasium as gym
from stable_baselines3 import A2C,SAC
import custom_arm_env  # Import your custom environment

# Create the environment
env = gym.make("CustomArmPick-v0")

algo = input('Which Algorith do you want to train? \n 1: A2C \n 2:SAC ')

if int(algo) == 1:
    model_path = "models/RLARM_a2c.zip"

    try:
        model = A2C.load(model_path, env=env)
        print("Loaded existing model from", model_path)
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")
        model = A2C("MlpPolicy", env, verbose=1)

elif int(algo) == 2:
    model_path = "models/RLARM_sac.zip"
    try:
        model = SAC.load(model_path, env=env)
        print("Loaded existing model from", model_path)
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")
        model = SAC("MlpPolicy", env, verbose=1)
else:
    model_path = "models/RLARM_a2c.zip"
    try:
        model = SAC.load(model_path, env=env)
        print("Loaded existing model from", model_path)
    except FileNotFoundError:
        print("No existing model found. Training from scratch.")
        model = SAC("MlpPolicy", env, verbose=1)

# Continue training
model.learn(total_timesteps=20000)

# Save the updated model
model.save(model_path)
env.close()
print("\a")
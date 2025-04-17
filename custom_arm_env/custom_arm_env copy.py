import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomArmEnv(gym.Env):
    """ Custom Gymnasium environment for a robotic arm with a cube to pick up. """

    metadata = {"render_modes": ["human"], "render_fps": 60}  # Define supported render modes

    def __init__(self, model_path="~/Documents/Coding\ Projects/Gymnasium/threedof/threedof.xml", render_mode=None):
        super().__init__()
        self.render_mode = render_mode  # Store render mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Find cube ID for tracking
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        
        # Action space: Joint torques
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        # Observation space: Joint positions, velocities, and cube position
        obs_dim = self.model.nq + self.model.nv + 3  # qpos, qvel, and cube position (x, y, z)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize viewer only if rendering is enabled
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_obs(self):
        """ Returns joint positions, velocities, and cube position as float32. """
        joint_pos = self.data.qpos[:self.model.nq].astype(np.float32)
        joint_vel = self.data.qvel[:self.model.nv].astype(np.float32)
        cube_pos = self.data.xpos[self.cube_id].astype(np.float32)  # Cube world position
        return np.concatenate([joint_pos, joint_vel, cube_pos]).astype(np.float32)

    def step(self, action):
        """ Applies action, advances simulation, and calculates reward. """
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        # Extract relevant information
        joint_pos = obs[:self.model.nq]
        joint_vel = obs[self.model.nq:self.model.nq + self.model.nv]
        cube_pos = obs[-3:]  # Cube world position
        end_effector_pos = self.data.site_xpos[self.ee_id]  # Get end-effector position

        # Compute distance to cube
        dist_to_cube = np.linalg.norm(end_effector_pos - cube_pos)

        # Reward function
        reward = -dist_to_cube  # Encourage moving closer to the cube
        if dist_to_cube < 0.05:  # Close enough to touch
            reward += 5.0
        if cube_pos[2] > 0.2:  # Lift bonus
            reward += 10.0

        # Energy penalty (encourage efficient movement)
        reward -= np.linalg.norm(action) * 0.1

        done = True  # Continuous environment

        return obs, reward, done, False, {}


    def reset(self, seed=None, options=None):
        """ Resets simulation and returns observation in float32. """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)  # Reset MuJoCo state
        return self._get_obs(), {}

    def render(self):
        """ Renders the environment using MuJoCo viewer and keeps it open. """
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            while self.viewer.is_running():
                mujoco.mj_step(self.model, self.data)  # Update render state
                self.viewer.sync()  # Keep viewer active

    def close(self):
        """ Clean up resources. """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
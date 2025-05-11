import mujoco
import mujoco.viewer
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Union
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class CustomArmEnv(MujocoEnv, utils.EzPickle):
    """ Custom Gymnasium environment for a robotic arm with a cube to pick up. """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }  # Define supported render modes

    def __init__(
            self, 
            xml_file="./armModel_threedof/threedof.xml", 
            frame_skip: int = 5,
            default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
            reward_near_weight: float = 0.5,
            reward_dist_weight: float = 2,
            reward_control_weight: float = 0.1,
            idle_penalty_rate: float = 0.1,
            **kwargs,
        ):

            utils.EzPickle.__init__(
                self,
                xml_file,
                frame_skip,
                default_camera_config,
                reward_near_weight,
                reward_dist_weight,
                reward_control_weight,
                **kwargs,
        )
            
            self._idle_penalty_rate = idle_penalty_rate
            self._time_steps = 0
            self._prev_dist_ee_cube = None  # To track previous distance
            self._time_without_progress = 0  # Track time without approaching cube
    
            self.model = mujoco.MjModel.from_xml_path(xml_file)
            self.data = mujoco.MjData(self.model)

            # Find cube ID for tracking
            self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")

            #Find end effector to track where palm is
            self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

            self._reward_near_weight = reward_near_weight
            # self._reward_dist_weight = reward_dist_weight
            self._reward_dist_weight = 2
            self._reward_control_weight = reward_control_weight

            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64)

            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
                )
        

            MujocoEnv.__init__(
                self,
                xml_file,
                frame_skip,
                observation_space=observation_space,
                default_camera_config=default_camera_config,
                **kwargs,
            )

            self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 200,
        }
            
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        self._time_steps += 1

        reward, reward_info = self._get_rew(action)
        info = reward_info
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, info

    def _get_rew(self, action):
        # Get the positions of the end-effector, the cube, and the goal using the "side"
        side_ee = self.data.site_xpos[self.ee_id]  # End effector position (side)
        side_cube = self.data.xpos[self.cube_id]  # Cube position (side)
        side_goal = self.get_body_com("goal")  # Goal position (using body COM for goal)

        # Calculate distances using the side positions
        dist_ee_cube = np.linalg.norm(side_ee - side_cube)  # Distance between end-effector and cube
        dist_cube_goal = np.linalg.norm(side_cube - side_goal)  # Distance between cube and goal

          # Calculate idle penalty based on progress toward cube
        reward_idle = 0
        if self._prev_dist_ee_cube is not None:
            # Check if the arm is making progress toward the cube
            progress = self._prev_dist_ee_cube - dist_ee_cube
            
            # If not making sufficient progress (using a small threshold)
            if progress < 0.005:  # Adjust this threshold as needed
                self._time_without_progress += self.frame_skip / 100.0  # Convert to seconds
                # Apply penalty every other second of no progress
                if self._time_without_progress >= 2.0:
                    reward_idle = -self._idle_penalty_rate * (self._time_without_progress // 2.0)
            else:
                # Reset the counter when making progress
                self._time_without_progress = 0
        
        # Store current distance for next comparison
        self._prev_dist_ee_cube = dist_ee_cube

        # Reward for bringing the end-effector closer to the cube
        reward_near = -dist_ee_cube * 5.0  # Increased weight for bringing the arm closer to the cube

        # Encourage moving the cube toward the goal *only if grasped*
        reward_dist = 0
        if dist_ee_cube < 0.05:  # If close enough to grasp the cube
            reward_dist = -dist_cube_goal * 3.0  # Only encourage goal movement after grasping

        # Control penalty to prevent random flailing
        reward_ctrl = -np.square(action).sum() * 0.5  # Increase penalty for flailing
        
        # Bonus for lifting the cube (Z-axis position)
        cube_height = side_cube[2]  # Get the height of the cube (Z-axis)
        reward_lift = 10.0 if cube_height > 0.2 else 0  # Bonus if the cube is lifted above threshold
        
        reward_grasp = 1.0 if dist_ee_cube < 0.05 else 0
        # Total reward is a combination of the above factors
        reward = reward_near + reward_dist + reward_ctrl + reward_lift + reward_grasp + reward_idle

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
            "reward_lift": reward_lift,
        }

        return reward, reward_info



    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                # self.get_body_com("palm_link"),
                self.data.site_xpos[self.ee_id],
                # self.get_body_com("object"),
                self.data.xpos[self.cube_id],
                self.get_body_com("goal"),
            ]
        )

    def reset_model(self):
        self._time_steps = 0
        self._prev_dist_ee_cube = None
        self._time_without_progress = 0

        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()
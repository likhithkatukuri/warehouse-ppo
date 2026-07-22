from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
from networkx.generators import spectral_graph_forge
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.start_goal_generetor import StartGoalGenerator


class WarehouseDatasetEnvironment(gym.Env):

    metadata = {"render_modes": ["human"]}

    # Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [
        (-1, 0),   # Up
        (1, 0),    # Down
        (0, -1),   # Left
        (0, 1)     # Right
    ]
        
    ############################################################
# Constructor
############################################################

    def __init__(
        self,
        dataset_path=None,
        resolution=0.20,
        floor_threshold=0.15,
        max_steps=600
    ):

        super().__init__()

        #    Default dataset path (for local machine)
        if dataset_path is None:
            dataset_path = "dataset"

        self.dataset_path = dataset_path

        self.loader = LiDARLoader(self.dataset_path)

        self.grid_generator = OccupancyGridGenerator(
        resolution=resolution,
        floor_threshold=floor_threshold
    )

        self.scan_names = self.loader.get_scan_names()

        self.max_steps = max_steps

        self.current_step = 0

        self.grid = None
        self.start = None
        self.goal = None
        self.robot_position = None

    ########################################################
    # Determine observation size
    ########################################################

        first_scan = self.loader.load_scan(self.scan_names[0])

        first_grid = self.grid_generator.generate(first_scan)

        h, w = first_grid.shape

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, h, w),
            dtype=np.uint8
        )

        ############################################################
    # Reset
    ############################################################

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        scan_name = random.choice(self.scan_names)

        points = self.loader.load_scan(scan_name)

        self.grid = self.grid_generator.generate(points)

        sg = StartGoalGenerator(self.grid)

        self.start, self.goal = sg.generate()

        self.robot_position = self.start

        self.current_step = 0

        observation = self.get_observation()

        info = {

            "scan_name": scan_name,

            "start": self.start,

            "goal": self.goal

        }

        return observation, info
    ############################################################
    # Get Observation
    ############################################################

    def get_observation(self):

        occupancy = self.grid.copy()

        robot_layer = np.zeros_like(self.grid, dtype=np.uint8)
        goal_layer = np.zeros_like(self.grid, dtype=np.uint8)

        rr, rc = self.robot_position
        gr, gc = self.goal

        robot_layer[rr, rc] = 1
        goal_layer[gr, gc] = 1

        observation = np.stack(
            [
                occupancy,
                robot_layer,
                goal_layer
            ],
            axis=0
        )

        return observation
            ############################################################
    # Distance
    ############################################################

    def distance_to_goal(self, position):

        return (

        abs(position[0] - self.goal[0])

        +

        abs(position[1] - self.goal[1])

        )
            ############################################################
    # Step
    ############################################################

    def step(self, action):

        self.current_step += 1

        terminated = False
        truncated = False
        info = {}

        # Current position
        current_distance = self.distance_to_goal(self.robot_position)

        dr, dc = self.ACTIONS[action]

        new_row = self.robot_position[0] + dr
        new_col = self.robot_position[1] + dc

        ########################################################
        # Check map boundaries
        ########################################################

        if (
            new_row < 0
            or new_row >= self.grid.shape[0]
            or new_col < 0
            or new_col >= self.grid.shape[1]
        ):

            reward = -10

            info["collision"] = True
            info["reason"] = "boundary"

            observation = self.get_observation()

            return observation, reward, terminated, truncated, info

        ########################################################
        # Check obstacle collision
        ########################################################

        if self.grid[new_row, new_col] == 1:

            reward = -10

            info["collision"] = True
            info["reason"] = "obstacle"

            observation = self.get_observation()

            return observation, reward, terminated, truncated, info

        ########################################################
        # Valid movement
        ########################################################

        self.robot_position = (new_row, new_col)

        ########################################################
        # Goal reached
        ########################################################

        if self.robot_position == self.goal:

            reward = 200

            terminated = True

            info["goal_reached"] = True

            observation = self.get_observation()

            return observation, reward, terminated, truncated, info

        ########################################################
        # Reward shaping
        ########################################################

        new_distance = self.distance_to_goal(self.robot_position)

        print(
            f"Current={current_distance:.2f}  "
            f"New={new_distance:.2f}"
        )

        reward = -0.1

        if new_distance < current_distance:

            reward += 2.0

        elif new_distance > current_distance:

            reward -= 2.0

        ########################################################
        # Episode timeout
        ########################################################

        if self.current_step >= self.max_steps:

            truncated = True

        observation = self.get_observation()

        return observation, reward, terminated, truncated, info
            ############################################################
    # Render
    ############################################################

    def render(self):

        print()

        print("=" * 40)

        print(f"Robot Position : {self.robot_position}")

        print(f"Goal Position  : {self.goal}")

        print(f"Steps          : {self.current_step}")

        print("=" * 40)

        print()
    ############################################################
    # Close
    ############################################################

    def close(self):
        pass
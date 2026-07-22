import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WarehouseEnvironment(gym.Env):

    """
    Warehouse Navigation Environment

    Compatible with:
    1. Dijkstra
    2. PPO (Stable-Baselines3)
    3. PP-D
    4. Future A*
    """

    metadata = {"render_modes": ["human"]}

    # -----------------------------
    # Action Definitions
    # -----------------------------
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

    # -----------------------------
    # Constructor
    # -----------------------------
    def __init__(self, grid, start, goal):

        super().__init__()

        self.grid = grid.astype(np.uint8)

        self.start = tuple(start)
        self.goal = tuple(goal)

        self.robot_position = self.start

        self.max_steps = 300
        self.current_step = 0

        # -----------------------------
        # RL Spaces
        # -----------------------------
        self.action_space = spaces.Discrete(4)

        h, w = self.grid.shape

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(3, h, w),
            dtype=np.uint8
        )

    # ===========================================================
    # RESET
    # ===========================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.robot_position = self.start
        self.current_step = 0

        observation = self.get_observation()

        return observation, {}

    # ===========================================================
    # OBSERVATION
    # ===========================================================

    def get_observation(self):

        h, w = self.grid.shape

        observation = np.zeros((3, h, w), dtype=np.uint8)

        # Channel 0 : Occupancy Grid
        observation[0] = self.grid

        # Channel 1 : Robot Position
        r, c = self.robot_position
        observation[1, r, c] = 1

        # Channel 2 : Goal Position
        gr, gc = self.goal
        observation[2, gr, gc] = 1

        return observation

    # ===========================================================
    # VALID POSITION
    # ===========================================================

    def valid_position(self, row, col):

        if row < 0 or row >= self.grid.shape[0]:
            return False

        if col < 0 or col >= self.grid.shape[1]:
            return False

        # 0 = Free
        return self.grid[row, col] == 0

    # ===========================================================
    # MOVE ROBOT
    # ===========================================================

    def move(self, action):

        dr, dc = self.ACTIONS[action]

        r, c = self.robot_position

        nr = r + dr
        nc = c + dc

        if self.valid_position(nr, nc):

            self.robot_position = (nr, nc)

            return True

        return False

    # ===========================================================
    # GET NEIGHBORS (For Dijkstra/A*)
    # ===========================================================

    def get_neighbors(self, position):

        neighbors = []

        r, c = position

        for dr, dc in self.ACTIONS:

            nr = r + dr
            nc = c + dc

            if self.valid_position(nr, nc):

                neighbors.append((nr, nc))

        return neighbors

    # ===========================================================
    # GOAL CHECK
    # ===========================================================

    def reached_goal(self):

        return self.robot_position == self.goal

    # ===========================================================
    # REWARD FUNCTION
    # ===========================================================

    def calculate_reward(self, old_position, new_position, moved):

        # Goal reached
        if new_position == self.goal:
            return 100

        # Hit obstacle / wall
        if not moved:
            return -10

        old_distance = (
            abs(old_position[0] - self.goal[0]) +
            abs(old_position[1] - self.goal[1])
        )

        new_distance = (
            abs(new_position[0] - self.goal[0]) +
            abs(new_position[1] - self.goal[1])
        )

        # Move closer
        if new_distance < old_distance:
            return 2

        # Move away
        if new_distance > old_distance:
            return -2

        # Same distance
        return -1

    # ===========================================================
    # STEP (Gymnasium)
    # ===========================================================

    def step(self, action):

        old_position = self.robot_position

        moved = self.move(action)

        new_position = self.robot_position

        reward = self.calculate_reward(
            old_position,
            new_position,
            moved
        )

        self.current_step += 1

        terminated = self.reached_goal()

        truncated = self.current_step >= self.max_steps

        observation = self.get_observation()

        info = {}

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # ===========================================================
    # RENDER
    # ===========================================================

    def render(self):

        print("\n==============================")
        print("Warehouse Environment")
        print("==============================")
        print("Robot :", self.robot_position)
        print("Goal  :", self.goal)
        print("Steps :", self.current_step)

    # ===========================================================
    # CLOSE
    # ===========================================================

    def close(self):
        pass
from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.start_goal_generetor import StartGoalGenerator

from env.warehouse_env import WarehouseEnvironment

from planners.ppo import PPOTrainer

loader = LiDARLoader(r"C:\Users\FOX TROT\Downloads\warehouse_data\Release_v1.0.0-001\Dataset\bin")

points = loader.load_scan("000000.bin")

generator = OccupancyGridGenerator()

grid = generator.generate(points)

sg = StartGoalGenerator(grid)

start, goal = sg.generate()

env = WarehouseEnvironment(grid, start, goal)

trainer = PPOTrainer(env)

trainer.load()

obs, _ = env.reset()

done = False

total_reward = 0

while not done:

    action = trainer.predict(obs)

    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward

    done = terminated or truncated

    env.render()

print()

print("Episode Finished")

print("Total Reward:", total_reward)
from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.start_goal_generetor import StartGoalGenerator
from env.warehouse_env import WarehouseEnvironment

loader = LiDARLoader(r"C:\Users\FOX TROT\Downloads\warehouse_data\Release_v1.0.0-001\Dataset\bin")
points = loader.load_scan("000000.bin")

generator = OccupancyGridGenerator()
grid = generator.generate(points)

sg = StartGoalGenerator(grid)
start, goal = sg.generate()

env = WarehouseEnvironment(grid, start, goal)

obs, info = env.reset()

print("=" * 50)
print("PPO ENVIRONMENT TEST")
print("=" * 50)

print("Observation Shape :", obs.shape)
print("Action Space      :", env.action_space)
print("Observation Space :", env.observation_space)
print("Start             :", start)
print("Goal              :", goal)

print("\nTaking 10 random actions...\n")

for i in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {i+1:2d} | "
        f"Action={action} | "
        f"Reward={reward:3d} | "
        f"Robot={env.robot_position}"
    )

    if terminated:
        print("\nGoal reached!")
        break

    if truncated:
        print("\nMaximum steps reached!")
        break
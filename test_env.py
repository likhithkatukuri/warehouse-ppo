from env.warehouse_env import WarehouseEnvironment
from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.start_goal_generetor import StartGoalGenerator

loader = LiDARLoader(r"C:\Users\FOX TROT\Downloads\warehouse_data\Release_v1.0.0-001\Dataset\bin")

points = loader.load_scan("000000.bin")

generator = OccupancyGridGenerator()

grid = generator.generate(points)

sg = StartGoalGenerator(grid)

start, goal = sg.generate()

env = WarehouseEnvironment(
    grid,
    start,
    goal
)

print("Robot:", env.robot_position)

print("Neighbors:")

for n in env.get_neighbors(env.robot_position):

    print(n)
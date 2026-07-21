from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.start_goal_generetor import StartGoalGenerator
from utils.map_utils import show_grid

from env.warehouse_env import WarehouseEnvironment

from planners.dijkstra import DijkstraPlanner

##############################################################

loader = LiDARLoader(r"C:\Users\FOX TROT\Downloads\warehouse_data\Release_v1.0.0-001\Dataset\bin")

points = loader.load_scan("000000.bin")

##############################################################

generator = OccupancyGridGenerator()

grid = generator.generate(points)

##############################################################

sg = StartGoalGenerator(grid)

start, goal = sg.generate()

##############################################################

env = WarehouseEnvironment(

    grid,

    start,

    goal

)

##############################################################

planner = DijkstraPlanner(env)

result = planner.find_path()

##############################################################

print()

print("=" * 60)

print("DIJKSTRA RESULTS")

print("=" * 60)

print("Success        :", result["success"])

print("Path Length    :", result["path_length"])

print("Path Cost      :", result["path_cost"])

print("Explored Nodes :", result["explored_nodes"])

print("Execution Time :", result["execution_time"])

##############################################################

show_grid(

    grid,

    start,

    goal,

    result["path"]

)
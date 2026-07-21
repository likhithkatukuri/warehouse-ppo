from utils.lidar_loader import LiDARLoader
from utils.occupancy_grid import OccupancyGridGenerator
from utils.map_utils import show_grid
from utils.start_goal_generetor import StartGoalGenerator

loader = LiDARLoader(
    r"C:\Users\FOX TROT\Downloads\warehouse_data\Release_v1.0.0-001\Dataset\bin"
)



points = loader.load_scan("000000.bin")
generator = OccupancyGridGenerator()

grid = generator.generate(points)


sg = StartGoalGenerator(grid)

start, goal = sg.generate()

print("Start :", start)
print("Goal  :", goal)

show_grid(grid, start, goal)
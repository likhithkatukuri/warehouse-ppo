import numpy as np


class OccupancyGridGenerator:

    def __init__(
        self,
        resolution=0.20,
        floor_threshold=0.15,
        xmin=-30.0,
        xmax=30.0,
        ymin=-30.0,
        ymax=30.0
    ):

        self.resolution = resolution
        self.floor_threshold = floor_threshold

        # Fixed world limits
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # Fixed grid size
        self.width = int(
            np.ceil((self.xmax - self.xmin) / self.resolution)
        )

        self.height = int(
            np.ceil((self.ymax - self.ymin) / self.resolution)
        )

    ############################################################

    def generate(self, points):

        grid = np.zeros(
            (self.height, self.width),
            dtype=np.uint8
        )

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        for xi, yi, zi in zip(x, y, z):

            # Ignore floor points
            if zi < self.floor_threshold:
                continue

            # Ignore points outside the map
            if (
                xi < self.xmin or xi >= self.xmax or
                yi < self.ymin or yi >= self.ymax
            ):
                continue

            col = int(
                (xi - self.xmin) / self.resolution
            )

            row = int(
                (yi - self.ymin) / self.resolution
            )

            if (
                0 <= row < self.height and
                0 <= col < self.width
            ):
                grid[row, col] = 1

        return grid
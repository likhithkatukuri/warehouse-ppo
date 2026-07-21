import numpy as np


class OccupancyGridGenerator:

    def __init__(

        self,

        resolution=0.20,

        floor_threshold=0.15

    ):

        self.resolution = resolution
        self.floor_threshold = floor_threshold

    def generate(self, points):

        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        xmin = x.min()
        xmax = x.max()

        ymin = y.min()
        ymax = y.max()

        width = int(
            np.ceil(
                (xmax-xmin)/self.resolution
            )
        )

        height = int(
            np.ceil(
                (ymax-ymin)/self.resolution
            )
        )

        grid = np.zeros(
            (height,width),
            dtype=np.uint8
        )

        for xi,yi,zi in zip(x,y,z):

            if zi < self.floor_threshold:
                continue

            row = int(
                (yi-ymin)/self.resolution
            )

            col = int(
                (xi-xmin)/self.resolution
            )

            if (
                0 <= row < height
                and
                0 <= col < width
            ):

                grid[row,col]=1

        return grid
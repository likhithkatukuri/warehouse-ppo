import random
from collections import deque
import numpy as np


class StartGoalGenerator:

    def __init__(self, grid):

        self.grid = grid

        self.free_cells = np.argwhere(grid == 0)

        self.rows = grid.shape[0]
        self.cols = grid.shape[1]

    ##########################################################

    def valid(self, r, c):

        if r < 0 or r >= self.rows:
            return False

        if c < 0 or c >= self.cols:
            return False

        return self.grid[r, c] == 0

    ##########################################################

    def reachable_cells(self, start):

        queue = deque([start])

        visited = {start}

        actions = [

            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)

        ]

        while queue:

            r, c = queue.popleft()

            for dr, dc in actions:

                nr = r + dr
                nc = c + dc

                node = (nr, nc)

                if not self.valid(nr, nc):
                    continue

                if node in visited:
                    continue

                visited.add(node)

                queue.append(node)

        return list(visited)

    ##########################################################

    def generate(self, minimum_distance=30):

        while True:

            idx = np.random.randint(len(self.free_cells))

            start = tuple(map(int, self.free_cells[idx]))

            reachable = self.reachable_cells(start)

            if len(reachable) < 2:
                continue

            random.shuffle(reachable)

            for goal in reachable:

                distance = (

                    abs(goal[0] - start[0])

                    +

                    abs(goal[1] - start[1])

                )

                if distance >= minimum_distance:

                    return start, goal
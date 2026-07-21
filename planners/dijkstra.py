import heapq
import time


class DijkstraPlanner:

    def __init__(self, environment):

        self.env = environment
        self.grid = environment.grid

    ##################################################################
    # Reconstruct Path
    ##################################################################

    def reconstruct_path(self, parents):

        path = []

        node = self.env.goal

        while node is not None:

            path.append(node)

            node = parents[node]

        path.reverse()

        return path

    ##################################################################
    # Find Shortest Path
    ##################################################################

    def find_path(self):

        start_time = time.perf_counter()

        start = self.env.start
        goal = self.env.goal

        priority_queue = []

        heapq.heappush(priority_queue, (0, start))

        distances = {

            start: 0

        }

        parents = {

            start: None

        }

        visited = set()

        exploration_order = []

        while priority_queue:

            current_distance, current = heapq.heappop(priority_queue)

            if current in visited:
                continue

            visited.add(current)

            exploration_order.append(current)

            ########################################################

            if current == goal:

                path = self.reconstruct_path(parents)

                execution_time = time.perf_counter() - start_time

                return {

                    "success": True,

                    "path": path,

                    "path_length": len(path),

                    "path_cost": current_distance,

                    "explored_nodes": len(visited),

                    "exploration_order": exploration_order,

                    "execution_time": execution_time

                }

            ########################################################

            for neighbor in self.env.get_neighbors(current):

                if neighbor in visited:
                    continue

                new_distance = current_distance + 1

                if (

                    neighbor not in distances

                    or

                    new_distance < distances[neighbor]

                ):

                    distances[neighbor] = new_distance

                    parents[neighbor] = current

                    heapq.heappush(

                        priority_queue,

                        (

                            new_distance,

                            neighbor

                        )

                    )

        ##############################################################

        execution_time = time.perf_counter() - start_time

        return {

            "success": False,

            "path": [],

            "path_length": 0,

            "path_cost": float("inf"),

            "explored_nodes": len(visited),

            "exploration_order": exploration_order,

            "execution_time": execution_time

        }
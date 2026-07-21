from pathlib import Path
import numpy as np


class LiDARLoader:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

    def load_scan(self, filename):

        filepath = self.dataset_path / filename

        points = np.fromfile(
            filepath,
            dtype=np.float32
        )

        points = points.reshape(-1, 4)

        return points

    def get_scan_names(self):

        return sorted(
            f.name
            for f in self.dataset_path.glob("*.bin")
        )

    def number_of_scans(self):

        return len(self.get_scan_names())
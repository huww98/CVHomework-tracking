import argparse
import os
import glob
import math
import re
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

GRAPH_PATH = './output/graph'

class Rect:
    cell_split_pattern = re.compile('\t|,')

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @classmethod
    def from_line(cls, line: str):
        if line.strip() == 'Tracking failure detected':
            return None
        cell = cls.cell_split_pattern.split(line)
        if len(cell) != 4:
            raise RuntimeError(f'A line "{line}" should contain 4 part')
        return cls(*[float(c) for c in cell])

    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def area(self):
        return self.h * self.w

    def max_point(self):
        return (self.x + self.w, self.y + self.h)

    def intersection_area_with(self, another):
        self_max = self.max_point()
        another_max = another.max_point()
        dx = min(self_max[0], another_max[0]) - max(self.x, another.x)
        dy = min(self_max[1], another_max[1]) - max(self.y, another.y)
        if dx >= 0 and dy >= 0:
            return dx * dy
        else:
            return 0

class RectSequence:
    def __init__(self, filename):
        with open(filename) as f:
            lines = f.readlines()
        self.rects = [Rect.from_line(l) for l in lines]

    def __len__(self):
        return len(self.rects)

    def center_distance_to(self, rects):
        if len(self) != len(rects):
            raise RuntimeError(f'rects len not equals: {len(self)} != {len(rects)}')
        distance = []
        for r1, r2 in zip(self.rects, rects.rects):
            if r1 is None or r2 is None:
                distance.append(float('inf'))
            else:
                p1 = np.array(r1.center())
                p2 = np.array(r2.center())
                squared_distance = ((p1 - p2) ** 2).sum()
                distance.append(math.sqrt(squared_distance))
        return np.array(distance)

    def area(self):
        return np.array([(r.area() if r else 0) for r in self.rects])

    def intersetion_area_with(self, rects):
        area = []
        for r1, r2 in zip(self.rects, rects.rects):
            if r1 is None or r2 is None:
                area.append(0.)
            else:
                area.append(r1.intersection_area_with(r2))
        return np.array(area)

class GraphData:
    def __init__(self, precision: Dict[str, np.ndarray], success: Dict[str, np.ndarray], frame_count: int, path: str):
        self.precision = precision
        self.success = success
        self.frame_count = frame_count
        self.path = path

    dis_thre = np.linspace(0, 50)
    overlap_thre = np.linspace(0, 1)

    @classmethod
    def from_result(cls, path: str):
        truth = RectSequence(os.path.join('data', path, 'groundtruth_rect.txt'))
        frame_count = len(truth)

        precision_dict = dict()
        success_dict = dict()
        for r in glob.glob(os.path.join('data', path, '*result.txt')):
            algrithm_name = os.path.basename(r)[:-10]
            result = RectSequence(r)

            distance = truth.center_distance_to(result)
            precision = (distance < cls.dis_thre[:, np.newaxis]).sum(axis=1) / frame_count
            precision_dict[algrithm_name] = precision

            intersection = truth.intersetion_area_with(result)
            union = truth.area() + result.area() - intersection
            success = (intersection / union > cls.overlap_thre[:, np.newaxis]).sum(axis=1) / frame_count
            success_dict[algrithm_name] = success
        return cls(precision_dict, success_dict, frame_count, path)

    @classmethod
    def merge(cls, data: list, path: str):
        algrithms = set.intersection(*[set(d.precision.keys()) for d in data])
        frame_count_list = [d.frame_count for d in data]
        frame_count = sum(frame_count_list)
        precision = {alg: np.average([d.precision[alg] for d in data], weights=frame_count_list, axis=0) for alg in algrithms}
        success = {alg: np.average([d.success[alg] for d in data], weights=frame_count_list, axis=0) for alg in algrithms}
        return cls(precision, success, frame_count, path)

    def draw(self):
        figure, axes = plt.subplots(2, 1, figsize=(8, 8))
        for alg in self.precision:
            axes[0].plot(self.dis_thre, self.precision[alg], label=alg)
            axes[1].plot(self.overlap_thre, self.success[alg], label=alg)

        axes[0].set_title('Precision Rate')
        axes[0].set_ylabel('Precision Rate')
        axes[0].set_xlabel('Location error threshold')

        axes[1].set_title('Success Rate')
        axes[1].set_ylabel('Success Rate')
        axes[1].set_xlabel('Overlap area rate threshold')

        axes[0].legend()
        axes[1].legend()

        name = '-'.join(self.path.split(os.sep)) if self.path else 'All'
        figure.suptitle(name, fontsize=18)
        figure.tight_layout(rect=[0, 0, 1, 0.95])

        head, tail = os.path.split(self.path)
        filename = tail or 'all'
        path = os.path.join(GRAPH_PATH, head)
        os.makedirs(path, exist_ok=True)
        figure.savefig(os.path.join(path, f'{filename}.png'))
        plt.close(figure)

def graph(path):
    dirlist = os.listdir(os.path.join('data', path))
    if 'groundtruth_rect.txt' in dirlist:
        data = GraphData.from_result(path)
    else:
        children_data = [graph(os.path.join(path, p)) for p in dirlist]
        data = GraphData.merge(children_data, path)

    data.draw()
    print(path or 'all')
    return data

def main(names=None):
    if names is None:
        names = ['']

    for n in names:
        graph(n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*')
    args = parser.parse_args()

    main(args.names or None)

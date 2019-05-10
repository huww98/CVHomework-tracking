import argparse
import os
import glob
import math
import re

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

def graph(name):
    truth = RectSequence(os.path.join('./data', name, 'groundtruth_rect.txt'))
    frame_count = len(truth)
    figure, axes = plt.subplots(2, 1, figsize=(8,8), tight_layout=True)

    for r in glob.glob(f'./data/{name}/*result.txt'):
        algrithm_name = os.path.basename(r)[:-10]
        rects = RectSequence(r)

        distance = truth.center_distance_to(rects)
        dis_thre = np.linspace(0, 50)
        precision = (distance < dis_thre[np.newaxis].T).sum(axis=1) / frame_count
        axes[0].plot(dis_thre, precision, label=algrithm_name)

        intersection = truth.intersetion_area_with(rects)
        union = truth.area() + rects.area() - intersection
        overlap_thre = np.linspace(0, 1)
        success = (intersection / union > overlap_thre[np.newaxis].T).sum(axis=1) / frame_count
        axes[1].plot(overlap_thre, success, label=algrithm_name)

    axes[0].set_title('Precision Rate')
    axes[0].set_ylabel('Precision Rate')
    axes[0].set_xlabel('Location error threshold')

    axes[1].set_title('Success Rate')
    axes[1].set_ylabel('Success Rate')
    axes[1].set_xlabel('Overlap area rate threshold')

    axes[0].legend()
    axes[1].legend()
    figure.savefig(os.path.join(GRAPH_PATH, f'{name}.png'))
    print(name)

def main(names=None):
    if names is None:
        names = os.listdir('./data')

    os.makedirs(GRAPH_PATH, exist_ok=True)
    for n in names:
        graph(n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*')
    args = parser.parse_args()

    main(args.names or None)

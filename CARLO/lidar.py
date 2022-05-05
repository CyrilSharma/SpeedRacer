# CUSTOM CODE
# This finds the closest direction at each line at a spacing of (2pi/200) radians between each line
# The distance to the nearest collision is found by simply iterating over every distance
# I could do binary search but I don't want to think about that right now

from world import World, Car
from geometry import Point, Line
import numpy as np


def get_first_collision_n(center: Point, world: World, dx, dy, maximum):
    d = (dx ** 2 + dy ** 2) ** 0.5
    for i in range(1, maximum):
        line = Line(center, Point(center.x + dx * i, center.y + dy * i))
        for obstacle in world.static_agents:
            if obstacle.collidable and obstacle.obj.intersectsWith(line):
                return i * d

    return maximum


dx = []
dy = []

for angle in np.linspace(0, 2 * np.pi, 200):
    dx.append(np.cos(angle))
    dy.append(np.sin(angle))


def read_lidar(w: World, c: Car, n_divisions: int):
    distances = []
    for angle in np.linspace(c.heading, c.heading + 2 * np.pi, n_divisions + 1)[:-1]:
        dx = np.cos(angle) * 0.1
        dy = np.sin(angle) * 0.1
        distance = get_first_collision_n(c.center, w, dx, dy, 1000)
        distances.append(distance)

    return distances

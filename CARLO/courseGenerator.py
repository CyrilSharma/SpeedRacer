import copy
import math
import random
import re
import time
from email.errors import BoundaryError
from tkinter import *

import numpy as np

from agents import (Car, CircleBuilding, Painting, Pedestrian,
                    RectangleBuilding, RingBuilding)
from geometry import Line, Point
# from graphics import Line as GraphicsLine
# from graphics import Point as GraphicsPoint
# from graphics import Rectangle as GraphicsRectangle
from interactive_controllers import KeyboardController
# from lidar import read_lidar
from world import World


def CourseGenerator():
    # time steps in terms of seconds. In other words, 1/dt is the FPS.
    dt = 0.1
    w = World(dt, width=120, height=120, ppm=6)
    # w.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25)))
    # w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))
    # w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

    # w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))
    # need to generate a continous path with a specified number of ups and downs that add to 0

    path1 = generatePath(Point(20, 20), Point(100, 120))
    path2 = generatePath(Point(100, 20), Point(100, 120))
    path3 = generatePath(Point(20, 20), Point(100, 20))

    grid = pathToGrid(path1)
    grid = pathToGrid(path2, grid)
    grid = pathToGrid(path3, grid)

    for rect in grid_to_rects(grid):
        w.add(rect)

    car = Car(Point(20, 20), np.pi/2)
    w.add(car)
    w.render()
    controller = KeyboardController(w)
    while True:
        try:
            car.set_control(controller.steering, controller.throttle)
            w.render()
            w.visualizer.win.flush()
            time.sleep(dt)
        except:
            break
    return [car, w]


def generatePath(startpoint, endpoint):
    # randomly choose number of ups and rights [minimum distance].
    differencePoint = endpoint - startpoint
    totalNumRights = int(differencePoint.x / 5)
    totalNumUps = int(differencePoint.y / 5)

    randomUps = np.random.randint(0, 5)
    randomRights = np.random.randint(0, 5)

    numUps = randomUps + totalNumUps
    numDowns = randomUps
    numRights = randomRights + totalNumRights
    numLefts = randomRights

    pathSteps = [*[Point(0, 5) for i in range(numUps)], *[Point(5, 0) for i in range(numRights)],
                 *[Point(0, -5) for i in range(numDowns)], *[Point(-5, 0) for i in range(numLefts)]]

    lines = []
    previousStep = None
    pos = startpoint
    remainingSteps = copy.deepcopy(pathSteps)
    for _ in range(len(pathSteps)):
        if previousStep is not None:
            tries = 0
            backwardStep = Point(-previousStep.x, -previousStep.y)
            while previousStep == backwardStep and tries < 5:
                index = random.randint(0, len(remainingSteps) - 1)
                step = remainingSteps[index]
                tries += 1
            if tries >= 5:
                steps = [Point(5, 0), Point(-5, 0), Point(0, 5), Point(0, -5)]
                steps.remove(backwardStep)
                steps.remove(previousStep)
                step = random.choice(steps)
            else:
                remainingSteps.pop(index)
        else:
            index = random.randint(0, len(remainingSteps) - 1)
            step = remainingSteps[index]
            remainingSteps.pop(index)

        line = Line(pos, pos + step)
        lines.append(line)
        pos += step

    return lines


grid_width = 120
grid_height = 120


def pathToGrid(lines, grid=None):

    import cv2

    if grid is None:
        grid = np.ones((grid_height, grid_width))

    for line in lines:
        p1, p2 = line.p1, line.p2

        def p2i(pt): return (int(pt.x), int(pt.y))

        cv2.line(grid, p2i(p1), p2i(p2), 0, 2)

    return grid


def grid_to_rects(grid):
    rects = []
    road_width = 2
    wall_width = 1

    for y in range(120):
        for x in range(120):
            # If this grid square is fully enclosed, suppress it
            road_bin1 = False

            road_check_indexes = [
                (surrounding_y, surrounding_x)
                for surrounding_y in range(y - road_width, y + road_width + 1)
                for surrounding_x in range(x - road_width, x + road_width + 1)

                if (surrounding_y != y) or (surrounding_x != x)
            ]

            for index in road_check_indexes:
                if index[0] < 0 or index[0] >= grid_height:
                    continue

                if index[1] < 0 or index[1] >= grid_height:
                    continue

                if grid[index[0]][index[1]] == 0:
                    road_bin1 = True

            road_bin2 = False

            obstacle_check_indexes = [
                (surrounding_y, surrounding_x)
                for surrounding_y in range(y - road_width - wall_width, y + road_width + wall_width + 1)
                for surrounding_x in range(x - road_width - wall_width, x + road_width + wall_width + 1)

                if (surrounding_y != y) or (surrounding_x != x)
            ]

            for index in obstacle_check_indexes:
                if index[0] < 0 or index[0] >= grid_height:
                    continue

                if index[1] < 0 or index[1] >= grid_height:
                    continue

                if grid[index[0]][index[1]] == 0:
                    road_bin2 = True

            if road_bin2 and not road_bin1:
                # print("add square")
                # add square
                center = Point(x + 0.5, y + 0.5)
                rects.append(RectangleBuilding(center, Point(1, 1)))

    # print("done")

    return rects


def lineToRects(line, width):
    rects = []
    dist = line.p1.distanceTo(line.p2)
    difference = (line.p2 - line.p1)
    centerPoint = line.p1 + difference / 2
    perDir = Point(-difference.y, difference.x) / difference.norm()
    rects.append(RectangleBuilding(
        centerPoint + width * 0.5 * perDir, Point(1, dist)))
    rects.append(RectangleBuilding(
        centerPoint - width * 0.5 * perDir, Point(1, dist)))
    return rects


def angleBetween(l1, l2):
    p1 = l1.p2 - l1.p1
    p2 = l2.p2 - l2.p1
    # find angle between vectors p1 and p2
    return 180 * math.atan2(p1.x * p2.y - p1.y * p2.x, p1.x * p2.x + p1.y * p2.y) / math.pi


CourseGenerator()

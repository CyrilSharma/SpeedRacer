import copy
from email.errors import BoundaryError
import math
import random
import re
import numpy as np
from geometry import Line
from agents import RectangleBuilding
from lidar import read_lidar
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time
from tkinter import *
from graphics import Line as GraphicsLine, Point as GraphicsPoint, Rectangle as GraphicsRectangle

def CourseGenerator():
    dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
    w = World(dt, width=120, height=120, ppm=6)
    # w.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25)))
    # w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))
    # w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

    # w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))
    line1 = Line(Point(20,20), Point(20,90))
    # need to generate a continous path with a specified number of ups and downs that add to 0

    path = generatePath(Point(20,20), Point(20,60))
    rects = [lineToRects(line, 10) for line in path]
    rects = [item for sublist in rects for item in sublist]
    for rect in rects:
        w.add(rect)
    car = Car(Point(20, 20), np.pi/2)
    w.add(car)
    while True:
        try: 
            w.render()
            time.sleep(dt)
        except:
            break
    return [car, w]

def generatePath(startpoint, endpoint):
    # randomly choose number of ups and rights [minimum distance].
    differencePoint = endpoint - startpoint
    totalNumRights = differencePoint.x / 5
    totalNumUps = differencePoint.y / 5

    randomUps = np.random.randint(0, 5)
    randomRights = np.random.randint(0, 5)
    numDowns = int(abs(totalNumUps - randomUps))
    numLefts = int(abs(totalNumRights - randomRights))
    print(numLefts)

    pathSteps = [*[Point(0,5) for i in range(randomUps)], *[Point(5,0) for i in range(randomRights)], *[Point(0,-5) for i in range(numDowns)], *[Point(-5,0) for i in range(numLefts)]]

    lines = []
    pos = startpoint
    remainingSteps = copy.deepcopy(pathSteps)
    for _ in range(len(pathSteps)):
        index = random.randint(0, len(remainingSteps) - 1)
        step = remainingSteps[index]
        remainingSteps.pop(index)
        line = Line(pos, pos + step)
        lines.append(line)
        pos += step
    
    return lines

def lineToRects(line, width):
    rects = []
    dist = line.p1.distanceTo(line.p2)
    difference = (line.p2 - line.p1)
    centerPoint = line.p1 + difference / 2
    perDir = Point(-difference.y, difference.x) / difference.norm()
    rects.append(RectangleBuilding(centerPoint + width * 0.5 * perDir, Point(1, dist)))
    rects.append(RectangleBuilding(centerPoint - width * 0.5 * perDir, Point(1, dist)))
    return rects


CourseGenerator()
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
    print(path)
    rects = pathToRects(path, 5)
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
    totalNumRights = int(differencePoint.x / 5)
    totalNumUps = int(differencePoint.y / 5)

    randomUps = np.random.randint(0, 5)
    randomRights = np.random.randint(0, 5)

    numUps = randomUps + totalNumUps
    numDowns = randomUps
    numRights = randomRights + totalNumRights
    numLefts = randomRights

    pathSteps = [*[Point(0,5) for i in range(numUps)], *[Point(5,0) for i in range(numRights)], *[Point(0,-5) for i in range(numDowns)], *[Point(-5,0) for i in range(numLefts)]]

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
                steps = [Point(5,0), Point(-5,0), Point(0,5), Point(0,-5)]
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

def pathToRects(lines, width):
    rects = []
    for i in range(len(lines) - 1):
        currentLine = lines[i]
        nextLine = lines[i + 1]

        # figure out if nextLine is left, right, or forward from current line
        dist = currentLine.p1.distanceTo(currentLine.p2)
        difference = (currentLine.p2 - currentLine.p1)
        centerPoint = currentLine.p1 + difference / 2
        perDir = Point(-difference.y, difference.x) / difference.norm()
        lineDir = difference / difference.norm()
        if lineDir.x != 0:
            size = Point(dist, 1)
        elif lineDir.y != 0:
            size = Point(1, dist)
        else:
            print("huH")
            size = Point(1, 1)

        if angleBetween(currentLine, nextLine) == 90:
            # next line is right
            rects.append(RectangleBuilding(centerPoint + width * 0.5 * perDir, size))
            rects.append(RectangleBuilding(centerPoint - width * 0.5 * perDir - width * 0.5 * lineDir, size))
            pass
        elif angleBetween(currentLine, nextLine) == -90:
            # next line is left
            rects.append(RectangleBuilding(centerPoint + width * 0.5 * perDir - width * 0.5 * lineDir, size))
            rects.append(RectangleBuilding(centerPoint - width * 0.5 * perDir, size))
            pass
        elif angleBetween(currentLine, nextLine) == 0:
            # next line is forward
            rects.append(RectangleBuilding(centerPoint + width * 0.5 * perDir, size))
            rects.append(RectangleBuilding(centerPoint - width * 0.5 * perDir, size))
        elif angleBetween(currentLine, nextLine) == 180 or angleBetween(currentLine, nextLine) == -180:
            # next line is backwards
            # this should never happen.
            pass
    return rects

def lineToRects(line, width):
    rects = []
    dist = line.p1.distanceTo(line.p2)
    difference = (line.p2 - line.p1)
    centerPoint = line.p1 + difference / 2
    perDir = Point(-difference.y, difference.x) / difference.norm()
    rects.append(RectangleBuilding(centerPoint + width * 0.5 * perDir, Point(1, dist)))
    rects.append(RectangleBuilding(centerPoint - width * 0.5 * perDir, Point(1, dist)))
    return rects

def angleBetween(l1, l2):
    p1 = l1.p2 - l1.p1
    p2 = l2.p2 - l2.p1
    # find angle between vectors p1 and p2
    return 180 * math.atan2(p1.x * p2.y - p1.y * p2.x, p1.x * p2.x + p1.y * p2.y) / math.pi


CourseGenerator()
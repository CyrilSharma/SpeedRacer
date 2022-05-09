import numpy as np
from lidar import read_lidar
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
from graphics import Line as GraphicsLine, Point as GraphicsPoint, Rectangle as GraphicsRectangle
import time

human_controller = True

dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.


def create_intersection_world():
    # The world is 120 meters by 120 meters. ppm is the pixels per meter.
    w = World(dt, width=120, height=120, ppm=6)

    # Let's add some sidewalks and RectangleBuildings.
    # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
    # A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
    # For both of these objects, we give the center point and the size.
    # We build a sidewalk.
    w.add(Painting(Point(71.5, 106.5), Point(97, 27), 'gray80'))
    # The RectangleBuilding is then on top of the sidewalk, with some margin.
    w.add(RectangleBuilding(Point(72.5, 107.5), Point(95, 25)))

    # Let's repeat this for 4 different RectangleBuildings.
    w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
    w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))

    w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
    w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

    w.add(Painting(Point(71.5, 41), Point(97, 82), 'gray80'))
    w.add(RectangleBuilding(Point(72.5, 40), Point(95, 80)))

    # Let's also add some zebra crossings, because why not.
    w.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
    w.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))

    return w


if __name__ == '__main__':
    w = create_intersection_world()

    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    c1 = Car(Point(20, 20), np.pi/2)
    w.add(c1)

    c2 = Car(Point(118, 90), np.pi, 'blue')
    # We can also specify an initial velocity just like this.
    c2.velocity = Point(3.0, 0)
    w.add(c2)

    # Pedestrian is almost the same as Car. It is a "circle" object rather than a rectangle.
    p1 = Pedestrian(Point(28, 81), np.pi)
    # We can specify min_speed and max_speed of a Pedestrian (and of a Car). This is 10 m/s, almost Usain Bolt.
    p1.max_speed = 10.0
    w.add(p1)

    w.render()  # This visualizes the world we just constructed.

    if not human_controller:
        # Let's implement some simple scenario with all agents
        # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
        p1.set_control(0, 0.22)
        c1.set_control(0, 0.35)
        c2.set_control(0, 0.05)
        for k in range(400):
            # All movable objects will keep their control the same as long as we don't change it.
            # Let's say the first Car will release throttle (and start slowing down due to friction)
            if k == 100:
                c1.set_control(0, 0)
            elif k == 200:  # The first Car starts pushing the brake a little bit. The second Car starts turning right with some throttle.
                c1.set_control(0, -0.02)
            elif k == 325:
                c1.set_control(0, 0.8)
                c2.set_control(-0.45, 0.3)
            elif k == 367:  # The second Car stops turning.
                c2.set_control(0, 0.1)
            w.tick()  # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/4)  # Let's watch it 4x

            # We can check if the Pedestrian is currently involved in a collision. We could also check c1 or c2.
            if w.collision_exists(p1):
                print('Pedestrian has died!')
            elif w.collision_exists():  # Or we can check if there is any collision at all.
                print('Collision exists somewhere...')
        w.close()

    # Let's use the steering wheel (Logitech G29) for the human control of car c1
    else:
        # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
        p1.set_control(0, 0.22)
        c2.set_control(0, 0.35)

        from interactive_controllers import KeyboardController
        controller = KeyboardController(w)
        for k in range(400):
            c1.set_control(controller.steering, controller.throttle)
            w.tick()  # This ticks the world for one time step (dt second)

            n_divisions = 10

            # CUSTOM CODE STARTS HERE
            # Relative to the car, read the distances in each direction
            # The indexes are sort of arbitrary
            start = time.time()
            lidar_measurements = read_lidar(w, c1, n_divisions)
            end = time.time()

            # (end - start) == ~0.02

            min_lidar = 120  # min(lidar_measurements)
            max_lidar = 0  # max(lidar_measurements)

            for i in range(n_divisions):
                start_x = int(640 / n_divisions * i)
                end_x = int(640 / n_divisions * (i + 1))
                start_y = 0
                end_y = 100

                img = GraphicsRectangle(GraphicsPoint(start_x, start_y), GraphicsPoint(
                    end_x, end_y))

                red_color = int(255 - 255 * (lidar_measurements[i] - min_lidar) / (
                    max_lidar - min_lidar))

                # Format {:.02X} does hex
                img.setFill(f"#{red_color:02X}0000")
                start = time.time()
                img.draw(w.visualizer.win)
                end = time.time()

            w.render()
            time.sleep(dt/4)  # Let's watch it 4x
            if w.collision_exists():
                import sys
                sys.exit(0)

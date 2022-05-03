from typing import List
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

from world import World
from agents import Car
from lidar import read_lidar


class RLCar(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        # self.conv1 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=1, out_channels=16, kernel_size=3)

        # self.conv2 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=16, out_channels=4, kernel_size=3)

        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 16)

        # Forward, backward, left, right
        self.output = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.output(x))

        return x


class Simulator:
    def __init__(self, world: World, car: Car, n_lidar_divisions: int):
        self.world = world
        self.car = car
        self.n_lidar_divisions = n_lidar_divisions

        self.dt = world.dt

    def step(self, forward_acceleration: float, steering_angle: float):
        self.car.set_control(steering_angle, forward_acceleration)

        return self.get_state()

    def get_state(self):
        lidar_measurements = read_lidar(
            self.world, self.car, self.n_lidar_divisions)

        return lidar_measurements

    def collided(self):
        return self.world.collision_exists(self.car)


def train(agent: RLCar, simulators: List[Simulator]):
    gamma = 0.7
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    for simulator in simulators:
        # Run simulation.
        # Array of (state, action, reward, next_state)
        history = []
        for _ in range(100):
            state = simulator.get_state()

            processed_state = torch.tensor([state]) / 100

            action = agent(processed_state)[0]

            forward_or_backward = torch.argmax(action[:2])
            left_or_right = torch.argmax(action[2:])

            simulator.step(forward_or_backward, left_or_right)

            # Q(s, a) = r + gamma * max_a' Q(s', a')
            # Q(s) = [r(a) + gamma * max_a' Q(s', a') for a in action space]

            collided = False

            prev_reward = simulator.car.velocity.norm()
            if simulator.collided():
                collided = True
                prev_reward += -100

            next_state = simulator.get_state()

            history.append((state, action, prev_reward, next_state))

            if collided:
                break

        # Discount rewards and run backprop.
        optimizer.zero_grad()

        running_reward = 0
        for i in range(len(history) - 1, -1, -1):
            state, action, reward, next_state = history[i]
            running_reward = running_reward * gamma + reward

            eps = 1e-5

            highest_p_fb = torch.log(torch.max(action[:2]) + eps)
            highest_p_lr = torch.log(torch.max(action[2:]) + eps)

            loss = highest_p_fb * reward + highest_p_lr * reward
            loss.backward()

        print(loss)

        optimizer.step()


if __name__ == "__main__":
    agent = RLCar(input_size=64)

    from example_circularroad import w as CircularWorld
    from geometry import Point
    import numpy as np

    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    car = Car(Point(91.75, 60), np.pi/2)
    car.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
    car.velocity = Point(0, 3.0)

    train(agent, [Simulator(CircularWorld, car, 64)] * 100)

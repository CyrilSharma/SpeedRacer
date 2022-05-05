from typing import List
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

from world import World
from agents import Car
from lidar import read_lidar, get_distance

import cv2


class RLCar(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        # self.conv1 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=1, out_channels=16, kernel_size=3)

        # self.conv2 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=16, out_channels=4, kernel_size=3)

        # self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(input_size + 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 16)

        # Forward/backward, left/right mean, f/b and l/r std
        self.output = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.flatten(x)
        x = F.relu(self.fc1(x.type(torch.float)))
        x = F.relu(self.fc2(x))
        x = self.output(x)
       # x = F.relu(self.fc3(x))
        x = torch.cat((F.tanh(x[:, :2]) * 10, F.sigmoid(x[:, 2:]) * 2), dim=1)
        return x


class Simulator:
    def __init__(self, world: World, car: Car, n_lidar_divisions: int):
        self.world = world
        self.car = car
        self.n_lidar_divisions = n_lidar_divisions
        self.throttle = 0
        self.steering = 0

        self.dt = world.dt

    def step(self, forward_or_backward, left_or_right):
        # if forward_or_backward >= 0.5:
        #     self.throttle += 1.5
        # else:
        #     self.throttle -= 1.5

        # if left_or_right >= 0.5:
        #     self.steering += 0.5
        # else:
        #     self.steering -= 0.5

        # self.car.set_control(self.steering, self.throttle)
        print("settting to",left_or_right, forward_or_backward)
        self.car.set_control(left_or_right, forward_or_backward)
        self.world.tick()
        self.car.tick(self.dt)

    def get_state(self):
        lidar_measurements = read_lidar(
            self.world, self.car, self.n_lidar_divisions)

        return lidar_measurements

    def get_distance(self):
        return get_distance(self.world, self.car)

    def collided(self):
        return self.world.collision_exists(self.car)


def train(agent: RLCar, simulators: List[Simulator]):
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)

    eps_std = 1

    gamma = 0.1
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.01)
    turnsZero = 0
    for it, simulator in enumerate(simulators):
        # Run simulation.
        # Array of (state, action, reward, next_state)
        history = []
        total_distance = 0
        # torch.manual_seed(42)
        for runtime in range(500):
            state = simulator.get_state()

            total_distance += simulator.car.velocity.norm()

            velocity_tensor = torch.tensor(
                [simulator.car.velocity.x, simulator.car.velocity.y])

            lidar_tensor = 1 / (torch.tensor(state) + 1)

            processed_state = torch.cat(
                [velocity_tensor, lidar_tensor]).unsqueeze(0)

            action_distribution = agent(processed_state)[0]
            #print(action_distribution)
            # sample from distribution to get action
            action = torch.normal(mean=action_distribution[:2], std=action_distribution[2:]/(eps_std*np.power(1.001, runtime))).detach().numpy()
            canvas[:] = 0
            center = simulator.car.center
            x, y = center.x, center.y
            cv2.circle(canvas, (int(x), int(y)), 5, (255, 255, 255), -1)

            cv2.imshow("Car", canvas)
            cv2.waitKey(1)
            
            simulator.step(action[0], action[1])

            # Q(s, a) = r + gamma * max_a' Q(s', a')
            # Q(s) = [r(a) + gamma * max_a' Q(s', a') for a in action space]

            end = False

            d = simulator.get_distance()

            reward = simulator.car.velocity.norm()
            print(simulator.car.velocity.norm(), turnsZero)
            if simulator.car.velocity.norm() < 0.1:
                turnsZero+=1
            else:
                turnsZero = 0

            if (d < 5):
                reward -= (5-d)*20
            # reward = total_distance
            # reward = 5 - max(simulator.car.velocity.norm() - 5, 0)
            if simulator.collided():
                print(it, "Collided after", runtime, "steps")
                end = True
                reward += -500
            if turnsZero > 5:
                print(it, "Stopped after", runtime, "steps")
                #reward += -100
                end = True

            next_state = simulator.get_state()

            history.append((state, processed_state, action, action_distribution, reward, next_state))

            if end:
                break
        else:
            print(it, "Finished after", runtime,
                  "steps and travelled", total_distance, "m")

        print(" - Final car velocity:", simulator.car.velocity.norm())

        # Discount rewards and run backprop.
        optimizer.zero_grad()

        total_loss = 0
        running_reward = 0
        discounted_rewards = []
        for i in range(len(history) - 1, -1, -1):
            state, processed_state, action, action_distribution, reward, next_state = history[i]
            running_reward = running_reward * gamma + reward
            discounted_rewards.append(running_reward)

        discounted_rewards = discounted_rewards[::-1]
       # discounted_rewards = (
       #     discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)
       # print("r: ", [x[3] for x in history], "rewards:", discounted_rewards)
        # agent.train(True)
        # for i in range(len(history)):
        #     state, processed_state, action, action_distribution, reward, next_state = history[i]

        #     discounted_reward = discounted_rewards[i]

        #     eps = 1e-5

        #     new_action_distribution = agent(processed_state)[0]

        #     assert((action_distribution == new_action_distribution).all())

        #     a_fb, a_lr = action 
        #     mu_fb, mu_lr, sigma_fb, sigma_lr = action_distribution
        #     # print(action_distribution)
        #     # print(mu_fb, mu_lr, sigma_fb, sigma_lr )
        #     # # probability density function values
        #     p_fb = torch.exp(-0.5 *((a_fb - mu_fb) / (sigma_fb))**2) * 1 / (sigma_fb * sqrt(2 * np.pi))
        #     p_lr = torch.exp(-0.5 *((a_lr - mu_lr) / (sigma_lr))**2) * 1 / (sigma_lr * sqrt(2 * np.pi))
        #     # print(p_fb, p_lr)
        #     # print(-torch.log(p_fb+1e-5) * discounted_reward)
        #     # print(-torch.log(p_lr+1e-5) * discounted_reward)
        #     # x = input()
        #     loss = -torch.log(p_fb+1e-5) * discounted_reward + -torch.log(p_lr+1e-5) * discounted_reward
        #     # print(loss, p_fb, p_lr, discounted_reward)
        #     loss.backward()

        #     total_loss += loss.item()

        # print(it, total_loss, loss)

        # optimizer.step()

    return agent


if __name__ == "__main__":
    agent = RLCar(input_size=64)

    from example_circularroad import w as CircularWorld
    from geometry import Point
    import numpy as np

    def get_simulator():
        # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
        car = Car(Point(91.75, 60), np.pi/2)
        car.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
        car.velocity = Point(0.0, 1.0)

        return Simulator(CircularWorld, car, 64)

    trained_agent = train(agent, [get_simulator() for _ in range(50)])

    torch.save(trained_agent.state_dict(), "trained_agent.pt")

import random
from collections import deque, namedtuple
from typing import Callable, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from agents import Car
from entities import RectangleEntity
from example_circularroad import create_circular_world
from geometry import Point
from lidar import read_lidar
from world import World


class RLCar(nn.Module):
    def __init__(self, input_size: int, n_actions: int):
        super().__init__()

        # self.conv1 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=1, out_channels=16, kernel_size=3)

        # self.conv2 = nn.Conv1d(
        #     padding_mode='zeros', in_channels=16, out_channels=4, kernel_size=3)

        # self.flatten = nn.Flatten()

        # One for speed, one for angular velocity
        self.fc1 = nn.Linear(input_size + 2, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 16)

        # Forward/backward, left/right
        self.output = nn.Linear(16, n_actions)

    def forward(self, x: torch.Tensor):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = self.flatten(x)
        x = F.relu(self.fc1(x.type(torch.float)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.output(x), dim=-1)

        return x


class Simulator:
    def __init__(self, world_factory: Callable[[], World], car: Car, n_lidar_divisions: int):
        self.world_factory = world_factory
        self.car = car
        self.car_initial_state = (
            car.center, car.velocity, car.acceleration, car.angular_velocity)
        self.n_lidar_divisions = n_lidar_divisions
        self.throttle = 0
        self.steering = 0

        self.world = None

    def reset(self):
        if self.world:
            self.world.close()

        car = Car(Point(93, 60), np.pi/2)
        car.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
        car.velocity = Point(0.0, 1.0)
        self.car = car
        self.world = self.world_factory()
        self.world.add(self.car)

    def step(self, action: int):
        if action in [0, 1, 2]:
            self.throttle += 1.5 * self.world.dt
        elif action in [3, 4, 5]:
            self.throttle -= 1.5 * self.world.dt

        if action in [0, 6, 3]:
            self.steering += 1 * self.world.dt
        elif action in [2, 7, 5]:
            self.steering -= 1 * self.world.dt

        self.car.set_control(self.steering, self.throttle)
        self.world.tick()
        self.car.tick(self.world.dt)

        reward = 0
        done = False

        if self.world.collision_exists():
            reward = -10
            print("Strong negative reward from collision")
            done = True
        elif self.car.velocity.norm() == 0:
            reward = -1
            print("Strong negative reward from stopping")
            done = True
        else:
            # state: Closer -> 1, further -> 0
            # reward is cross product of vector from center of circle with velocity
            # world width and height are 60
            cx, cy = self.car.center.x - 60, self.car.center.y - 60
            vx, vy = self.car.velocity.x, self.car.velocity.y

            reward = (
                ((cx * vy) - (cy * vx)) / (cx * cx + cy * cy) ** 0.5) ** 1/3
            done = False
            print("Reward for time step:", reward)

        return reward, done

    def get_state(self):
        lidar_measurements = read_lidar(
            self.world, self.car, self.n_lidar_divisions)

        # Keep value between 0 and 1
        lidar_measurements_normalized = 1 / \
            (1 + torch.tensor(lidar_measurements))

        # velocity_tensor = torch.tensor(
        #     [simulator.car.velocity.x, simulator.car.velocity.y])
        velocity_tensor = torch.tensor(
            [self.car.velocity.norm(), self.car.angular_velocity])

        # Combine lidar measurements and car kinematics
        combined = torch.cat([lidar_measurements_normalized, velocity_tensor])

        return combined

    def collided(self):
        return self.world.collision_exists(self.car)


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def train(simulators: List[Simulator], input_size: int):
    # Some inspiration from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    n_actions = 8
    # Description of Action Space
    """
    0  1  2

    6 Car 7

    3  4  5

    1 and 4 are forward and backward
    0 and 2 are forward/left and forward/right
    3 and 5 are backward/left and backward/right
    6 and 7 are left/right with no throttle
    """

    device = torch.device('cpu')

    policy_net = RLCar(input_size, n_actions).to(device)
    target_net = RLCar(input_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    # Set the target net to evaluation mode
    target_net.eval()

    memory = ReplayMemory(10000)
    BATCH_SIZE = 12
    GAMMA = 0.5
    TARGET_UPDATE = 10
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_STEPS = 200

    def select_action(state: torch.Tensor, eps: float):
        # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        if random.random() > eps:
            # We don't explore, we just take the best action
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                _, max_indices = policy_net(state).max(0)

                return max_indices.view(1)
        else:
            # Explore
            return torch.tensor([random.randrange(n_actions)], device=device, dtype=torch.long)

    def optimize_model():
        # Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # A final state has .next_state is None == True.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                             if s is not None])
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.tensor(batch.reward).view(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(
            non_final_next_states).max(1)[0].detach()

        # Discount factor does not have to be calculated recursively.
        # Instead, we multiply the expected future state by gamma.
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # Compare the predicted Q(S, A) with the expected Q(S, A)
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    canvas = np.zeros((400, 400, 3), dtype=np.uint8)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0001)

    num_episodes = 2500

    distance_history = []

    simulator = simulators[0]

    RENDER_EPISODE = 1

    for i_episode in range(num_episodes):
        simulator.reset()
        total_distance = 0
        total_reward = 0
        for t in range(1000):
            state = simulator.get_state()

            if (i_episode % RENDER_EPISODE) == 0:
                simulator.world.render()

            EPS_STEPS += 1
            eps = EPS_END + (EPS_START - EPS_END) * np.exp(-t / EPS_STEPS)
            action = select_action(state, eps)

            reward, done = simulator.step(action)
            total_reward += reward
            if not done:
                next_state = simulator.get_state()
            else:
                next_state = None

            memory.push(state, action, next_state, reward)

            total_distance += simulator.car.velocity.norm() * simulator.world.dt

            # print(t, total_distance)
            canvas[:] = 0
            car_pos = simulator.car.center
            cv2.circle(canvas, (int(car_pos.x * 400/120), int(car_pos.y * 400/120)),
                       5, (0, 0, 255), -1)
            cv2.imshow('canvas', canvas[::-1])

            if cv2.waitKey(1) == ord('q'):
                exit()

            optimize_model()
            if done:
                print(i_episode, t, total_distance, total_reward)
                break

        distance_history.append(t)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    import matplotlib.pyplot as plt

    plt.title("Distance over time")
    plt.xlabel("Episodes")
    plt.ylabel("Distance")
    plt.plot(distance_history)
    plt.show()

    return policy_net


INPUT_SIZE = 60


def get_simulator():
    # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
    car = Car(Point(91.75, 60), np.pi/2)
    car.max_speed = 30.0  # let's say the maximum is 30 m/s (108 km/h)
    car.velocity = Point(0.0, 1.0)

    return Simulator(create_circular_world, car, INPUT_SIZE)


if __name__ == "__main__":
    # agent = RLCar(input_size=64)

    trained_agent = train([get_simulator()], input_size=INPUT_SIZE)

    torch.save(trained_agent.state_dict(), "trained_agent.pt")

import matplotlib.pyplot as plt
import numpy as np
import collections
# import pygame
# import time


time_steps = 10000

# Initialize pygame display
w = 640
h = 480
# d = pygame.display.set_mode((w, h))


def get_pid_runtime(k1, k2, k3):
    x = 0
    velocity = 0
    total_error = 0
    target_x = 400
    error_history = collections.deque(maxlen=3)
    eps = 0.001
    steps = 0

    dt = 0.01

    within_range_in_a_row = 0
    while within_range_in_a_row < 10 and steps < 20000:
        steps += 1

        error = target_x - x
        error_history.append(error)
        total_error += error
        if len(error_history) == 3:
            d_error = (error_history[2] - error_history[0]) / (2 * dt)
        else:
            d_error = 0

        hoped_velocity = k1 * error + k2 * total_error + k3 * d_error
        velocity += (hoped_velocity - velocity) * 0.5

        x += velocity * dt

        if abs(x - target_x) < eps:
            within_range_in_a_row += 1
        else:
            within_range_in_a_row = 0

    return steps


def clamp(x):
    return max(min(x, 640), 0)


def plot_random_sampling():
    np.random.seed(42)

    result_k2 = []
    result_k3 = []
    result_values = []
    for i in range(1000):
        k1 = 1
        k2 = np.random.rand() * 3
        k3 = np.random.rand() * 3
        result_k2.append(k2)
        result_k3.append(k3)
        result_values.append(get_pid_runtime(k1, k2, k3))

    plt.title("Effect of k2 and k3 on velocity")
    plt.scatter(result_k2, result_k3, c=result_values)
    plt.xlabel("K2")
    plt.ylabel("K3")
    plt.colorbar()
    plt.show()


def plot_image():
    n_k2 = 250
    n_k3 = 250
    results = np.zeros((n_k2, n_k3))
    k1 = 1
    k2_values = np.linspace(0, 1, n_k2)
    k3_values = np.linspace(0, 1, n_k3)

    plt.title("PID convergence rate")

    for i in range(n_k2):
        print(i)
        for j in range(n_k3):
            k2 = k2_values[i]
            k3 = k3_values[j]
            results[i, j] = get_pid_runtime(k1, k2, k3)

    plt.imshow(results)
    plt.show()


plot_random_sampling()

# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # d.fill((255, 255, 255))
#     # pygame.draw.rect(
#     #     d, (255, 0, 0), ((clamp(x - 5), 320 - 5), (10, 10)))

#     pygame.display.flip()

#     # time.sleep(1/60)

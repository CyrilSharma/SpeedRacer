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
    error_history = []  # collections.deque(maxlen=30)
    eps = 0.001
    steps = 0

    dt = 0.01

    within_range_in_a_row = 0
    while within_range_in_a_row < 10 and steps < 20000:
        steps += 1

        error = target_x - x
        error_history.append(error)
        total_error += error
        if len(error_history) >= 3:
            d_error = (error_history[-1] - error_history[-3]) / (2 * dt)
        else:
            d_error = 0
        # total_error = sum(error_history)

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


def metamodel():
    import pickle
    import os

    np.random.seed(42)

    if os.path.exists("./data.pkl"):
        with open("./data.pkl", "rb") as f:
            X, y = pickle.load(f)
    else:
        X, y = run_random_sampling(plot=False)

        with open("./data.pkl", "wb") as f:
            pickle.dump((X, y), f)

    print("Loaded training samples")

    from sklearn.model_selection import train_test_split

    y = y[:, np.newaxis]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    method = 'random_forest'

    if method == 'dense_network':
        import tensorflow as tf
        import tensorflow.keras.models as models
        import tensorflow.keras.layers as layers

        regressor = models.Sequential([
            layers.Normalization(axis=-1),
            layers.Dense(16, input_shape=(2,), activation='sigmoid'),
            layers.Dense(8, activation='sigmoid'),
            layers.Dense(1, activation='relu'),
        ])
        regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])

        regressor.fit(X_train, y_train, epochs=100)
        y_pred = regressor.predict(X_test)[:, 0]
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor

        regressor = RandomForestRegressor()
        regressor.fit(X_train, y_train)

        import time

        start_time = time.time()
        y_pred = regressor.predict(X_test)
        end_time = time.time()

        print(f"Random forest regression time: {end_time - start_time:.5f}")

        result = regressor.score(X_test, y_test)

        print("Scoring")
        print(result)

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 1, 1)
    plt.title("Real")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=(y_train[:, 0]))

    plt.subplot(2, 1, 2)
    plt.title("Predicted")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)

    plt.show()


def run_random_sampling(plot=True):
    import time

    n = 2000
    result_points = np.zeros((n, 2))
    result_values = []
    start_time = time.time()
    for i in range(n):
        k1 = 1
        k2 = np.random.rand() * 10
        k3 = np.random.rand() * 10
        result_points[i] = [k2, k3]
        # np.append(result_points, [k2, k3], axis=0)
        result_values.append(get_pid_runtime(k1, k2, k3))
    end_time = time.time()

    print(f"Elapsed time: {end_time - start_time:.2f}")

    if plot:
        plt.title("Effect of k2 and k3 on velocity")
        plt.scatter(result_points[:, 0], result_points[:, 1], c=result_values)
        plt.xlabel("K2")
        plt.ylabel("K3")
        plt.colorbar()
        plt.show()

    return result_points, np.array(result_values)


def run_imaging():
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


# run_random_sampling()

metamodel()

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

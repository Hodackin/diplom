import numpy as np


class KalmanFilter:

    def __init__(self, mes_noise, mod_noise):
        # transition matrix
        self.A = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        self.R = np.identity(3) * mes_noise # measurement cov
        self.Q = np.identity(3) * mod_noise # process cov
        # measurement matrix
        self.H = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])
        self.P = np.identity(3) # estimate cov
        self.X_hat_prev = np.zeros(3).reshape(3, -1) # init state

    def prediction(self):
        x_hat_next = self.A @ self.X_hat_prev
        p_next = self.A @ self.P @ self.A.T + self.Q
        return x_hat_next, p_next

    def update(self, x_hat, p, z):
        k = p @ self.H.T @ np.linalg.inv(self.H @ p @ self.H.T + self.R) # kalman gain
        x = x_hat + k @ (z - self.H @ x_hat)
        p = p - k @ self.H @ p # ?
        return x, p

    def first_observation(self, p):
        self.X_hat_prev = p
        self.P = self.R.copy()


def test():
    import matplotlib.pyplot as plt

    np.random.seed(100)
    # generate data
    x = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    y = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = x ** 2 + y ** 2

    # add noise to generated data
    x_noised = x + np.random.normal(0, 0.3, 100)
    y_noised = y + np.random.normal(0, 0.3, 100)
    z_noised = z + np.random.normal(0, 0.3, 100)

    data = np.stack((x_noised, y_noised, z_noised), axis=-1) # synthetic measurement

    kf = KalmanFilter(mes_noise=2, mod_noise=0.3)
    kf.first_observation(data[0, :])

    pred = [kf.X_hat_prev]

    for point in data[1:, :]:
        x_hat_next, p_next = kf.prediction()
        x_, p = kf.update(x_hat_next, p_next, point)
        kf.P, kf.X_hat_prev = p, x_
        pred.append(x_hat_next)
    pred = np.array(pred).reshape(-1, 3)

    # visualize
    fig = plt.figure(figsize=(14, 14))
    ax = plt.axes(projection='3d')

    ax.plot3D(x, y, z, c="red", linestyle="dashed")
    ax.plot3D(pred[:, 0], pred[:, 1], pred[:, 2], c="blue")
    ax.scatter3D(x_noised, y_noised, z_noised, s=15, c="green")

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_zlabel("Altitude")

    ax.view_init(45, 310)
    plt.legend(["truth ground", "FK", "radar data"])
    plt.show()


if __name__ == "__main__":
    test()
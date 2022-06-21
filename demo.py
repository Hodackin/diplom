import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import cv2
from keras.models import model_from_json

import os
from experiments.fk import KalmanFilter

IMG_SIZE = 150
IMG_LIST = os.listdir(os.getcwd() + "/vd/drone")
LIST_SIZE = len(IMG_LIST)
CATEGORIES = ["BIRD", "DRONE"]


def load_model(model_file_path, weights_file_path):
    with open(model_file_path, "r") as f:
        loaded_model = f.read()

    model = model_from_json(loaded_model)
    model.load_weights(weights_file_path)
    # model.summary()
    return model


def prepare(file_path):
    img_array = cv2.imread(file_path)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)


def ReLU(x):
    return x * (x > 0)


def live_plot(frame, *args):
    ax, kf, pred, data, model = args
    x, y, z = data[frame]

    if frame == 0:
        kf.first_observation(data[frame])
        pred.append(data[frame])
    elif frame > 0:
        x_hat_next, p_next = kf.prediction()
        x_, p = kf.update(x_hat_next, p_next, data[frame])
        kf.P, kf.X_hat_prev = p, x_
        pred.append(x_hat_next)

        temp = np.array(pred).reshape(-1, 3)
        ax.plot3D(temp[:, 0], temp[:, 1], temp[:, 2], c="blue")

    ax.scatter3D(x, y, z, s=20, c="green")

    if (frame + 1) % (150 // LIST_SIZE) == 0:
        n = (frame+1) // (150 // LIST_SIZE)
        print(n)
        img_name = IMG_LIST[n - 1]
        model_pred = model.predict(prepare("vd/drone/" + img_name))
        ax.text(temp[-1, 0], temp[-1, 1], temp[-1, 2],
                CATEGORIES[int(model_pred[0, 0])], size=15, zorder=1)
    plt.draw()


def main():
    np.random.seed(10)
    x = np.linspace(-4 * np.pi, 4 * np.pi, 150)
    y = np.linspace(-4 * np.pi, 4 * np.pi, 150)
    z = x + y
    z = ReLU(z)
    # add noise to generated data
    x_noised = x + np.random.normal(0, 0.01, 150)
    y_noised = y + np.random.normal(0, 0.3, 150)
    z_noised = z + np.random.normal(0, 0.3, 150) + 2

    data = np.stack((x_noised, y_noised, z_noised), axis=-1) # synthetic measurement

    fig = plt.figure(figsize=(12, 12))
    print(fig.get_dpi(), fig.get_size_inches())
    ax = plt.axes(projection='3d')

    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    ax.set_zlabel("Altitude")
    ax.set_xlim(min(x_noised), max(x_noised))
    ax.set_ylim(min(y_noised), max(y_noised))
    ax.set_zlim(0, max(z_noised))
    ax.view_init(30, 315)

    kf = KalmanFilter(mes_noise=2, mod_noise=0.3)
    pred = []
    model = load_model("model.json", "model.h5")
    anim = FuncAnimation(plt.gcf(), live_plot, fargs=(ax, kf, pred, data, model),
                         interval=100, frames=len(z), repeat=False)
    writergif = PillowWriter(fps=30)
    anim.save("viz.gif", writer=writergif)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    # plt.show()


if __name__ == "__main__":
    main()


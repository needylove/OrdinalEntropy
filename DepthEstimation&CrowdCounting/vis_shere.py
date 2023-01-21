"""
Visualize the features on a sphere
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import matplotlib


def main(pth):
    radius = 1
    u = np.linspace(0,  2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = radius * np.outer(np.sin(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.cos(v))
    z = radius * np.outer(np.cos(u), np.ones_like(v))

    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x, y, z, color='black', linewidths = 0.3)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # ax.set_aspect("equal")
    ax.set_box_aspect([1,1,1])

    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])

    plt.axis('off')
    plt.figure(dpi=3000)


    data = sio.loadmat(pth)
    embeds = data['embeds']
    labels = data['labels']
    preds = data['preds']

    value =preds


    xs = embeds[:,0]
    ys = embeds[:,1]
    zs = embeds[:,2]


    cmapper = matplotlib.cm.get_cmap('magma_r')
    sphere = ax.scatter(xs, ys, zs, c = value, cmap= 'brg', vmin=np.min(value), vmax=np.max(value))
    for i in range(xs.size):
        ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], color = 'lightgrey')
    fig.colorbar(sphere, fraction=0.02)
    plt.show()

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)

if __name__ == "__main__":
    """
    Select one to visualization
    """
    pth = 'regression.mat'               # regression model
    # pth = 'classification_256.mat'       # classification model with 256 classes trained with cross-entropy loss
    # pth = 'regression+LdPrime.mat'       # regression model + L'_d
    # pth = 'regression+Ld.mat'            # regression model + L_d
    # pth = 'regression+Ld+Lt.mat'            # regression model + L_d +L_t


    main(pth)
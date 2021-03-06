"""Scripts to plot solutions."""

import matplotlib.pyplot as plt
from cmcrameri import cm  # library for colors for a fitting color scheme
import numpy as np


def plot_3d_solution(x_values, y_values, u_values, x_label, y_label, plot_title, value_range=None):
    plt.figure(figsize=(5, 5))
    plt.rc('font', **{'size': '11'})

    x = np.linspace(x_values[0], x_values[-1], len(x_values) + 1)
    y = np.linspace(y_values[0], y_values[-1], len(y_values) + 1)

    if type(value_range) == list:
        plt.pcolor(x, y, u_values, cmap=cm.batlow, vmin=value_range[0], vmax=value_range[1])
    else:
        plt.pcolormesh(x, y, u_values, cmap=cm.batlow, snap=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.title(plot_title)
    plt.tight_layout()

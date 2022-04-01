"""Scripts to plot solutions."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_3d_solution(x_values, y_values, u_values, x_label, y_label, plot_title, value_range):
    plt.figure(figsize=(5, 5))
    plt.rc('font', **{'size': '11'})
    plt.pcolor(x_values, y_values, u_values, shading='auto', vmin=value_range[0], vmax=value_range[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar()
    plt.title(plot_title)
    plt.tight_layout()


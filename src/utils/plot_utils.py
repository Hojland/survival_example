import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def nuuday_palette(palette: str = "primary", color_list: list = "all"):
    primary = {
        "sky_blue": "#6e6eff",
        "bright_green": "#5bfe60",
        "burning_yellow": "#fffa1e",
        "dark": "#453740",
        "blue_0": "#3f3fff",
        "blue_1": "#8c8cff",
        "blue_2": "#b2b3ff",
        "blue_light": "#d9d9ff",
        "green_0": "#83ff4a",
        "green_1": "#b5ff92",
        "green_2": "#cdffb7",
        "green_light": "#e6ffdb",
        "yellow_0": "#ffff00",
        "yellow_1": "#ffff66",
        "yellow_2": "#ffff99",
        "yellow_light": "#ffffcc",
        "black": "#000000",
        "white": "#ffffff",
    }
    secondary = {
        "light_blue_0": "#91f8ff",
        "bright_pink": "#ffa0ff",
        "bright_orange": "#ffa082",
        "light_blue_1": "#bdfbff",
        "light_blue_2": "#d3fcff",
        "light_blue_3": "#e9feff",
        "light_red_0": "#ff9f82",
        "light_red_1": "#ffc6b4",
        "light_red_2": "#ffd9cd",
        "light_red_3": "#ffece6",
        "purple_1": "#ff76ff",
        "purple_2": "#ff38ff",
        "purple_3": "#cf00cf",
    }
    yousee = {"dark_green": "#14503a", "green": "#2c795b", "light_green": "#3cb84d"}
    if palette == "primary":
        out = primary
    elif palette == "secondary":
        out = secondary
    elif palette == "yousee":
        out = yousee
    else:
        raise NotImplementedError()
    if color_list == "all":
        hex_list = [color for color in out.values()]
    else:
        hex_list = [out[color] for color in color_list]
    return hex_list


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = "{:.2f}".format(p.get_height())
            ax.text(_x, _y, value, ha="center")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def set_ax_style(ax, font):
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_color("white")
    # ax.spines["left"].set_color("white")
    # ax.tick_params(axis="x", colors="white")
    # ax.tick_params(axis="y", colors="white")
    return ax
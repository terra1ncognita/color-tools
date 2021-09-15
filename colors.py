"""Module providing functions on color conversions and relative luminosity.

Based on code from https://github.com/Peter-Slump/python-contrast-ratio/
(MIT Licence, Copyright (c) 2014 Peter Slump)
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import re

RGBColor = Tuple[float, float, float]  # Normalized 0-255 int -> 0-1 float
HexColor = str

# CIE XYZ <-> sRGB conversion coefficients for Y
_alpha_r, _alpha_g, _alpha_b = 0.2126, 0.7152, 0.0722


def get_hex_code(color: HexColor) -> RGBColor:
    """Convert hex triplet string into RGB float tuple."""
    if not color.startswith("#"):
        raise ValueError("Color must start with #")

    if (result := re.match(r"^#?([a-f0-9]{3,3}|[a-f0-9]{6,6})$", color)) is None:
        raise RuntimeError(f"Could not parse color string {color}")

    result = list(result.group(1))

    if len(result) == 6:
        result = [result[i] + result[i + 1] for i in range(0, len(result), 2)]
    else:
        result = [result[i] + result[i] for i in range(0, len(result))]

    return tuple(int(hex_code, 16) / 255.0 for hex_code in result)


def calculate_luminance(srgb_value: float) -> float:
    """Converts gamma-compressed (sRGB) values into gamma-expanded (linear)."""
    if srgb_value < 0.03928:
        return srgb_value / 12.92
    return ((srgb_value + 0.055) / 1.055) ** 2.4


def calculate_relative_luminance(rgb: RGBColor) -> float:
    """Calculates relative luminance (Y) from RGB tuple.
    
    Based on CIE XYZ <-> sRGB conversion for gamma-expanded (linear) values.
    """
    return (
            _alpha_r * calculate_luminance(rgb[0])
            + _alpha_g * calculate_luminance(rgb[1])
            + _alpha_b * calculate_luminance(rgb[2])
    )


def get_cr_rgb(color_one: RGBColor, color_two: RGBColor) -> float:
    """Computes contrast ratio of two RGB colors."""
    return get_cr_lum(calculate_relative_luminance(color_one), calculate_relative_luminance(color_two))


def get_cr_lum(lum1: float, lum2: float) -> float:
    """Compute contrast ratio of two luminance values."""
    val = (lum1 + 0.05) / (lum2 + 0.05)
    return val if val >= 1 else 1 / val


def get_contrast_ratio(color_1: HexColor, color_2: HexColor) -> float:
    """Get contrast ratio of two Hex colors."""
    color_one, color_two = get_hex_code(color_1), get_hex_code(color_2)
    return get_cr_rgb(color_one, color_two)


def rgb2hex_str(rgb_color: RGBColor) -> HexColor:
    """Converts RGB tuple into Hex color string."""
    def clamp(x: int) -> int:
        return max(0, min(x, 255))

    r, g, b = map(lambda x: int(x*255), rgb_color)
    return f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"


def inv_luminance(lum: float) -> float:
    """Convert gamma-expanded (linear) value into gamma-compressed value."""
    if lum < 0.0028218390804597704:
        return lum * 12.92
    return lum ** (1 / 2.4) * 1.055 - 0.055


def rl_from_garmonic_cr(rl_1: float, rl_2: float = 1.0) -> float:
    """Computes new relative luminance to get best contrast with two colors."""
    if rl_1 > rl_2:
        rl_1, rl_2 = rl_2, rl_1
    return np.sqrt((rl_1 + 0.05) * (rl_2 + 0.05)) - 0.05


# TODO check if it is correct from math/logic perspective
def _find_rl(rls: np.ndarray) -> float:
    """Finds relative luminance that is the most contrast to given luminances."""
    minimal, maximal = min(rls), max(rls)
    l, r = get_cr_lum(minimal, 0), get_cr_lum(maximal, 1)

    diffs = np.ediff1d(rls)
    max_diff_idx = diffs.argmax()

    mid_cr = np.sqrt(get_cr_lum(rls[max_diff_idx], rls[max_diff_idx + 1]))

    print(l, mid_cr, r)

    if mid_cr >= l and mid_cr >= r:
        return rl_from_garmonic_cr(rls[max_diff_idx], rls[max_diff_idx + 1])
    elif l >= r:
        return 0
    return 1


def _get_g_for_rl(rl: float, r: float, b: float) -> float:
    """Computes green channel value G from given R, B and relative luminance."""
    return inv_luminance((rl - _alpha_r * calculate_luminance(r) - _alpha_b * calculate_luminance(b)) / _alpha_g)


def _build_colors_with_lum(opt_lum: float, n_points: int = 20, plot: bool = True) -> List[RGBColor]:
    """Creates a list of colors with given relative luminance and (optionally) plots them.
    
    :param opt_lum: target relative luminance to achieve
    :param n_points: number of grid points for each color channel
    :param plot: whether to plot created colors in a square heatmap (empty cells are filled with white color)
    """
    clrs = []
    for r in np.linspace(0, 1, n_points):
        for b in np.linspace(0, 1, n_points):
            if 0 <= (g := _get_g_for_rl(opt_lum, r, b)) <= 1:
                clrs.append((r, g, b))

    _size = int(np.sqrt(len(clrs)))
    clrs = clrs + [(1, 1, 1) for _ in range(_size ** 2 - len(clrs))]

    if plot:
        new_colormap = ListedColormap(clrs)
        gradient = np.linspace(0, 1, _size ** 2).reshape((_size, -1))
    
        fig, ax = plt.subplots(figsize=(_size * 0.25, _size * 0.25))
        ax.imshow(gradient, cmap=new_colormap)
    
        plt.show()
        plt.close()
    return clrs


if __name__ == '__main__':
    pass

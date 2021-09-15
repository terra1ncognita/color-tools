import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import re

_alpha_r, _alpha_g, _alpha_b = 0.2126, 0.7152, 0.0722


def get_hex_code(color: str):
    if not color.startswith("#"):
        raise ValueError("Color must start with #")

    if (result := re.match(r"^#?([a-f0-9]{3,3}|[a-f0-9]{6,6})$", color)) is None:
        raise RuntimeError("Could not parse color string")

    result = list(result.group(1))

    if len(result) == 6:
        result = [result[i] + result[i + 1] for i in range(0, len(result), 2)]
    else:
        result = [result[i] + result[i] for i in range(0, len(result))]

    return [int(hex_code, 16) / 255.0 for hex_code in result]


def calculate_luminace(index):
    if index < 0.03928:
        return index / 12.92
    return ((index + 0.055) / 1.055) ** 2.4


def calculate_relative_luminance(rgb):
    return (
        _alpha_r * calculate_luminace(rgb[0])
        + _alpha_g * calculate_luminace(rgb[1])
        + _alpha_b * calculate_luminace(rgb[2])
    )


def get_cr_rgb(color_one, color_two):
    #     if sum(color_one) > sum(color_two):
    light, dark = color_one, color_two
    #     else:
    #         light, dark = color_two, color_one
    return get_cr_lum(calculate_relative_luminance(light), calculate_relative_luminance(dark))


def get_cr_lum(l1, l2):
    val = (l1 + 0.05) / (l2 + 0.05)
    return val if val >= 1 else 1 / val


def get_contrast_ratio(color_1, color_2):
    color_one, color_two = get_hex_code(color_1), get_hex_code(color_2)
    return get_cr_rgb(color_one, color_two)


def rgb2hex_str(t):
    def clamp(x):
        return max(0, min(x, 255))

    r, g, b = map(int, t)
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


def inv_luminance(l):
    if l < 0.0028218390804597704:
        return l * 12.92
    return l ** (1 / 2.4) * 1.055 - 0.055


def rl_from_garmonic_cr(rl_1, rl_2=1.0):
    if rl_1 > rl_2:
        rl_1, rl_2 = rl_2, rl_1
    return np.sqrt((rl_1 + 0.05) * (rl_2 + 0.05)) - 0.05


def _find_rl(rls):
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


def _get_g_for_rl(rl, r, b):
    return inv_luminance((rl - _alpha_r * calculate_luminace(r) - _alpha_b * calculate_luminace(b)) / _alpha_g)


def _build_colors_with_lum(opt_lum):
    clrs = []
    for r in np.linspace(0, 1, 20):
        for b in np.linspace(0, 1, 20):
            if 0 <= (g := _get_g_for_rl(opt_lum, r, b)) <= 1:
                clrs.append((r, g, b))

    _size = int(np.sqrt(len(clrs)))
    clrs = clrs + [(1, 1, 1) for _ in range(_size ** 2 - len(clrs))]

    new_colormap = ListedColormap(clrs)

    gradient = np.linspace(0, 1, _size ** 2).reshape((_size, -1))

    fig, ax = plt.subplots(figsize=(_size * 0.25, _size * 0.25))
    ax.imshow(gradient, cmap=new_colormap)

    plt.show()
    plt.close()
    return clrs

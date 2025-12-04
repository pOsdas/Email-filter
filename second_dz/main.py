import matplotlib
matplotlib.use("TkAgg")

import random
import math
import matplotlib.pyplot as plt


def exp_from_uniform(u, lam=1.0):
    # X = -ln(u) / лямбда
    # 1 - u == 0 когда u = 1, но random.random() не возвращает 1
    return -math.log(1-u) / lam


def generate_exp_sample(n, lam=1.0):
    return [exp_from_uniform(random.random(), lam) for i in range(n)]


def histogram(samples, otrezoks=30, range_max=None):
    n = len(samples)
    if n == 0:
        return [], [], 0.0
    if range_max is None:
        range_max = max(samples)

    range_max = max(1.0, range_max)
    otrezok_width = range_max / otrezoks
    counts = [0] * otrezoks
    for s in samples:
        if s < 0:
            continue
        if s >= range_max:
            # все что выходит за правую границу в последний отрезок
            counts[-1] += 1
        else:
            idx = int(s / otrezok_width)
            if idx >= otrezoks:
                idx = otrezoks - 1
            counts[idx] += 1
    # плотности: count / (n * otrezok_width)
    densities = [c / (n * otrezok_width) for c in counts]
    centers = [(i + 0.5) * otrezok_width for i in range(otrezoks)]
    return centers, densities, otrezok_width


# def plot_hist_and_pdf(samples, lam=1.0, otrezoks=30, title_prefix="n"):
#     centers, densities, otrezok_width = histogram(samples, otrezoks=otrezoks, range_max=None)
#     if not centers:
#         print("Нет данных для построения")
#         return
#
#     fig = plt.figure(figsize=(8, 5))
#     ax = fig.add_subplot(111)
#
#     ax.bar(centers, densities, width=otrezok_width, align='center')
#
#     max_x = max(max(samples), centers[-1] + otrezok_width / 2)
#     xs = [i * max_x / 400 for i in range(401)]
#     ys = [lam * math.exp(-lam * x) for x in xs]
#     ax.plot(xs, ys)
#
#     ax.set_title(f"{title_prefix}: нормированная гистограмма и теоретическая плотность (λ={lam})")
#     ax.set_xlabel("y")
#     ax.set_ylabel("Плотность")
#     ax.set_xlim(0, max_x)
#     ax.grid(True)
#
#     rgb = fig_to_rgb_array(fig)
#     plt.show()
#
#     return rgb
#
#
# def fig_to_rgb_array(fig):
#     fig.canvas.draw()
#     buf = fig.canvas.buffer_rgba()
#
#     import numpy as np
#     rgba = np.frombuffer(buf, dtype=np.uint8)
#     rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
#
#     # альфа канал отброс
#     rgb = rgba[:, :, :3]
#     return rgb


def plot_hist_and_pdf_return_fig(samples, lam=1.0, otrezoks=30, title_prefix="n"):
    centers, densities, otrezok_width = histogram(samples, otrezoks=otrezoks, range_max=None)
    if not centers:
        print("Нет данных для построения")
        return None

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(centers, densities, width=otrezok_width, align='center')

    max_x = max(max(samples), centers[-1] + otrezok_width / 2)
    xs = [i * max_x / 400 for i in range(401)]
    ys = [lam * math.exp(-lam * x) for x in xs]
    ax.plot(xs, ys)

    ax.set_title(f"{title_prefix}: нормированная гистограмма и теоретическая плотность (λ={lam})")
    ax.set_xlabel("y")
    ax.set_ylabel("Плотность")
    ax.set_xlim(0, max_x)
    ax.grid(True)
    return fig


if __name__ == "__main__":
    lam = 1.0
    sample_sizes = [100, 1000, 10000, 50000]
    otrezoks = 30

    figs = []
    for n in sample_sizes:
        samples = generate_exp_sample(n, lam=lam)
        mean_sample = sum(samples) / n
        mean_theoretical = 1.0 / lam
        print(f"n = {n}: среднее выборки = {mean_sample:.4f}, теоретическое E[X] = {mean_theoretical:.4f}")
        fig = plot_hist_and_pdf_return_fig(samples, lam=lam, otrezoks=otrezoks, title_prefix=f"n={n}")
        if fig is not None:
            figs.append(fig)

    # показ всех открытых окон одновременно
    plt.show()
import collections  # used to store collections of data, for example, list, dict, set, tuple etc.
import cv2  # to work with images
import matplotlib.pyplot as plt
import numpy as np
import gym


def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    s_plt1 = fig.add_subplot(111, label="1")  # "111" means "1x1 grid, first subplot",
    # also "234" means "2x3 grid, 4th subplot".
    s_plt2 = fig.add_subplot(111, label="2", frame_on=False)  # "frame_on=False" means showing two subplots
    # in one frame at the same time, transparently

    s_plt1.plot(x, epsilons, color="C0")
    s_plt1.set_xlabel("Training Steps", color="C0")
    s_plt1.set_ylabel("Epsilon", color="C0")
    s_plt1.tick_params(axis="x", color="C0")  # "tick_params" is used to change the appearance of ticks,
    # tick labels, and gridlines.
    s_plt1.tick_params(axis="y", color="C0")

    n = len(scores)
    running_avg = np.empty(n)
    for i in range(n):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    s_plt2.scatter(x, running_avg, color="C1")
    s_plt2.axes.get_xaxis().set_visible(False)  # "axes.get_xaxis()" returns the XAxis instance
    s_plt2.yaxis.tick_right()
    s_plt2.set_ylabel('Score', color="C1")
    s_plt2.yaxis.set_label_position('right')
    s_plt2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)



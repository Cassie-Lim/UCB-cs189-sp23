import numpy as np
import matplotlib.pyplot as plt


def p_norm(vec, p):
    return np.power(np.sum(np.power(abs(vec), p), axis=2), 1/p)

w1, w2 = np.mgrid[-2:2:.05, -2:2:.05]
points = np.dstack((w1, w2))
for p in [0.5, 1, 2]:
    plt.contourf(w1, w2, p_norm(points, p), 20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.title(f'Isocontours of $\ell_{p}$ p_norm')
    plt.show()
    plt.savefig(f'ml-hw4-p5-{p}.png')


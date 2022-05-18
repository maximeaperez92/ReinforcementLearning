import numpy as np


def reset_line_world():
    NB = 5

    S = np.arange(NB)
    A = np.array([0, 1])  # 0 = Gauche et 1 = Droite
    R = np.array([-1, 0, 1])
    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(1, NB - 2):
        p[i, 1, i + 1, 1] = 1.0

    for i in range(2, NB - 1):
        p[i, 0, i - 1, 1] = 1.0

    p[1, 0, 0, 0] = 1.0
    p[NB - 2, 1, NB - 1, 2] = 1.0

    V = np.zeros((len(S),))

    pi = np.zeros((len(S), len(A)))
    pi[:] = 0.5

    gamma = 0.999999  # facteur d'amoindrissement
    threshold = 0.000001

    return S, A, R, p, gamma, threshold, pi, V

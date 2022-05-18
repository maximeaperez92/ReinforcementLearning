import numpy as np


def reset_grid_world():
    NB = 5

    S = np.arange(NB*NB)
    A = np.array([0, 1, 2, 3])  # 0 = haut, 1 = bas, 2 = gauche et 3 = droite
    R = np.array([-1, 0, 1])
    p = np.zeros((len(S), len(A), len(S), len(R)))

    for i in range(NB, NB * NB):  # haut
        p[i, 0, i-5, 1] = 1.0

    for i in range(0, NB * NB - NB):  # bas
        p[i, 1, i+5, 1] = 1.0

    for i in range(1, NB * NB):  # gauche
        if i % 5 != 0:
            p[i, 2, i-1, 1] = 1.0

    cpt = 1
    for i in range(0, NB * NB - 2):  # droite
        if cpt == NB:
            cpt = 1
        else:
            p[i, 3, i+1, 1] = 1.0
            cpt += 1

    # rendre les états terminaux
    # case en haut à droite
    p[NB - 1, 1, NB - 1 + NB, 1] = 0.0
    p[NB - 1, 2, NB - 2, 1] = 0.0
    p[NB - 1, 3, NB, 1] = 0.0
    # case en bas à droite
    p[NB * NB - 1, 0, NB * NB - 1 - NB, 1] = 0.0
    p[NB * NB - 1, 2, NB * NB - 2, 1] = 0.0

    # enlever les rewards neutres sur les cases terminales
    # case en haut à droite
    p[NB - 2, 3, NB - 1, 1] = 0.0
    p[2 * NB - 1, 0, NB - 1, 1] = 0.0
    # case en bas à droite
    p[NB * NB - 2, 3, NB * NB - 1, 1] = 0.0
    p[NB * (NB - 1) - 1, 1, NB * NB - 1, 1] = 0.0

    # activer les rewards positifs et négatifs
    # case en haut à droite
    p[NB - 2, 3, NB - 1, 0] = 1.0
    p[2 * NB - 1, 0, NB - 1, 0] = 1.0
    # case en bas à droite
    p[NB * NB - 2, 3, NB * NB - 1, 2] = 1.0
    p[NB * (NB - 1) - 1, 1, NB * NB - 1, 2] = 1.0

    gamma = 0.999999  # facteur d'amoindrissement
    threshold = 0.000001

    pi = np.zeros((len(S), len(A)))
    pi[:] = 0.25

    # V = np.zeros((len(S)))
    # print(V)
    V = np.zeros((len(S),))

    return S, A, R, p, gamma, threshold, pi, V

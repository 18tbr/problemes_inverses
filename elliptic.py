#!/usr/bin/python3

import sys
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# from matplotlib.pyplot import ion
from math import *


def qcoeff(x):
    return 1 + x


def rhs(x):
    return pi ** 2 * (1 + x) * np.sin(pi * x) - pi * np.cos(pi * x)


def uex(x):
    return np.sin(pi * x)


def makesys(q, dx):
    """ Calcul des matrices de rigidité et de masse """
    nx = np.size(q) - 1
    # K est une matrice tridiagonale
    K = 1 / dx * (np.diag(q[:-1] + q[1:]) - np.diag(q[1:-1], 1) - np.diag(q[1:-1], -1))
    # M est une autre matrice tridiagonale, np.ones(int) renvoie une liste.
    M = dx * (
        np.diag(2.0 / 3.0 * np.ones(nx))
        + np.diag(1.0 / 6.0 * np.ones(nx - 1), 1)
        + np.diag(1.0 / 6.0 * np.ones(nx - 1), -1)
    )
    return (K, M)


def direct(q, f, dx):
    K, M = makesys(q, dx)
    return linalg.solve(K, np.dot(M, f))


def adjoint(q, r, dx):
    """ ADJOINT l'adjoint de la fonction 'directe'
    %   Entree: q  = coefficient de diffusion
    %           r  = second membre pour l'état adjoint
    %           dx = pas de discrétisation
    %
    %  Sortie: p = etat adjoint (solution de K'p = -M'r)
"""
    K, M = makesys(q, dx)
    # K et M sont symétriques
    return linalg.solve(K, -np.dot(M, r))


def tstadj(q, f, dx):
    """ TSTADJ Test du produit scalaire
    %
    % Entree : q = coefficient de diffusion
    %          f = second membre du problème direct
    %          dx = pas de discrétisation
    %
    % Sortie: fp = produit scalaire (f, p)
    %         uu = produit scalaire (u, u)
    %         avec u = état direct avec 2nd membre f
    %              p = état adjoint avec 2nd membre u
    %
    %   Les 2 nombres en sortie doivent être égaux.
    %   Attention à la matrice de masse dans le produit scalaire
    """
    K, M = makesys(q, dx)
    u = direct(q, f, dx)
    r = u
    p = adjoint(q, r, dx)
    fp = -f @ M @ p
    uu = u @ M @ u
    return fp, uu


def simul(q, d, f, dx):
    """ SIMUL Calcul de la fonction coût pour un paramètre de simulation q
    %  Entrée:  q = coefficient de diffusion
    %           d = mesures
    %           f = second membre du l'état direct
    %           dx = pas de discrétisation
    %
    % Sortie    cout = valeur de la fonction objectif
    %
    % Cette fonction doit calculer cout = 1/2 (u(q) - d)' * M * (u(q) -d)
    % où u(q) est le résultat de l'état direct
    """
    K, M = makesys(q, dx)
    u = direct(q, f, dx)
    cout = 1 / 2 * np.transpose(u - d) @ M @ (u - d)
    return cout


def gradient(q, d, f, dx):
    """" GRADIENT Gradient par état adjoint
    %
    % Entrées:  q = coefficient de diffusion
    %           d = mesures
    %           f = second membre de l'état direct
    %           dx = pas de discretisation
    % Sortie :  g = gradient de la fonction coût (voir simul)
    %
    % Après calcul des états direct u et adjoint p, il pourra être commode de
    %   rajouter des bords a u et p pour calculer le gradient
    """
    u = direct(q, f, dx)
    r = u - d
    p = adjoint(q, r, dx)
    utemp0 = np.array(list(u) + [0])
    ptemp0 = np.array(list(p) + [0])
    utemp1 = np.array([0] + list(u))
    ptemp1 = np.array([0] + list(p))
    g = (1/dx) ** 2 * (utemp1 - utemp0) * (ptemp1 - ptemp0)
    # g = np.zeros(np.size(u) + 1)
    # for i in range(1, np.size(u) - 1):
    #     g[i] = (u[i + 1] - u[i]) * (p[i + 1] - p[i]) / (dx ** 2)
    # # On complète le gradient avec les conditions au bord
    # last = np.size(u) - 1
    # g[0] = (u[0] - 0) * (p[0] - 0) / (dx ** 2)
    # g[last + 1] = (0 - u[last]) * (0 - p[last]) / (dx ** 2)
    return g


def simopt(qopt, d, f, dx):
    """ SIMOPT Simulation en fonction du paramètre d'optimisation
    %  Implémenté seulement si nsim est multiple de nopt
    """

    nopt = np.size(qopt)
    nx = np.size(f)
    nsim = nx + 1
    if nsim % nopt == 0:
        frac = nsim // nopt
        qsim = np.zeros(nsim)
        for i in range(nsim):
            qsim[i] = qopt[i // frac]
        return simul(qsim, d, f, dx)
    else:
        raise Exception("Pas implémenté")


def gradopt(qopt, d, f, dx):
    """ GRADOPT Gradient par rapport au paramètre d'optimisation
    Implémenté seulement si nsim est un multiple de nopt
    """
    nopt = np.size(qopt)
    nx = np.size(f)
    nsim = nx + 1
    # print(f"{nopt} {nx} {nsim}")
    if nsim % nopt == 0:
        frac = nsim // nopt
        qsim = np.zeros(nsim)
        for i in range(nopt):
            for j in range(frac):
                qsim[i * frac + j] = qopt[i]

        gsim = gradient(qsim, d, f, dx)
        # On calcule la taille de qopt
        sopt = np.size(gsim) // frac
        gopt = np.zeros(sopt)
        for i in range(sopt):
            for k in range(frac):
                gopt[i] += gsim[i * frac + k]
        gopt /= frac
        return gopt
        # return gradient(qopt, d, f, dx)
    else:
        raise Exception("Pas implemente")


def tstgrad(q0, qdir, d, f, dx, delta):
    """ TSTGRAD verification du gradient par differences finies
    %
    % Entrée: q0    = points où est calculé le gradient
    %         qdir  = direction de la dérivée directionnelle (i.e. gradient)
    %         d     = mesures
    %         f     = second membre de l'état direct
    %         dx    = pas de discrétisation
    %         delta = pas pour le calcul par différences finies
    %  gref est la valeur par etat adjoint
    %  g le vecteur des approximations par DF pour les pas dans delta
    """

    cref = simopt(q0, d, f, dx)
    gopt = gradopt(q0, d, f, dx)
    gref = dx * np.dot(qdir, gopt)

    c = np.zeros(np.size(delta))
    for i in np.arange(np.size(delta)):
        c[i] = simopt(q0 + delta[i] * qdir, d, f, dx)
    # print(c)
    g = (c - cref) / delta
    return gref, g

def main():
    # ion()
    nx = 31
    dx = 1 / (nx + 1)
    x = np.linspace(dx, 1 - dx, nx)
    fx = rhs(x)
    xdemi = np.linspace(dx / 2, 1 - dx / 2, nx + 1)
    qx = qcoeff(xdemi)
    u = direct(qx, fx, dx)
    x1 = np.append(np.append([0], x), [1])
    u1 = np.append(np.append([0], u), [0])

    plt.figure(1)
    plt.subplot(221)
    plt.plot(xdemi, qx)
    plt.xlabel("x")
    plt.ylabel("Coefficient")
    plt.subplot(222)
    plt.plot(x, fx)
    plt.xlabel("x")
    plt.ylabel("Second membre")
    plt.subplot(223)
    plt.plot(x1, uex(x1), x1, u1)
    plt.xlabel("x")
    plt.ylabel("Solution")
    plt.subplot(224)
    plt.plot(x, (uex(x) - u) / uex(x))
    plt.xlabel("x")
    plt.ylabel("Erreur")
    plt.pause(0.001)

    input("---> Appuyer sur une touche <---")
    fp, uu = tstadj(qx, fx, dx)
    print("Test du produit scalaire : fp={0:18.15e}, uu={1:18.15e}".format(fp, uu))

    input("---> Appuyer sur une touche <---")
    d = uex(x)
    d = d * (1 + 1e-2 * np.random.rand(np.size(d)))
    q0 = 0.5 * np.ones(np.size(xdemi))
    qdir = gradopt(q0, d, fx, dx)
    print(f"Computed gradient : {qdir}")
    delta = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    gref, g = tstgrad(q0, qdir, d, fx, dx, delta)
    # Améliorer la présentation
    print(f"Adjoint gradient : {gref}")
    print(f"Finite difference gradient : {g}")
    print(f"Relative gradient error : {(g - gref) / g}")

    input("---> Appuyer sur une touche <---")
    # Test de l'optimisation

    nopt = nx + 1
    print(f"\nOptimisation simple {nopt}\n")
    qinit = np.ones(nopt)
    dopt = 1.0 / nopt
    # xopt représente l'espace dans lequel vit q (les milieux de segments)
    xopt = np.linspace(dopt / 2, 1 - dopt / 2, nopt)
    resopt = minimize(
        simopt,
        qinit,
        method="BFGS",
        jac=gradopt,
        args=(d, fx, dx),
        options={"disp": True},
    )
    qopt = resopt.x
    usim = direct(qopt, fx, dx)
    plt.figure(2)
    plt.subplot(211)
    plt.plot(xopt, qopt, label = "Optimum found")
    plt.plot(xopt, qcoeff(xopt), label = "Reference value")
    plt.legend()
    plt.subplot(212)
    plt.plot(x, usim, label = "Approximation found")
    plt.plot(x, d, label = "Input curve")
    plt.legend()
    plt.pause(0.001)
    print(f"Approximation error {nopt} : {simopt(qopt, d, fx, dx)}")
    input("---> Appuyer sur une touche <---")

    # Optimisation hiérarchique
    nopt = 1
    qinit = np.ones(nopt)
    for i in range(6):
        print(f"\nOptimisation hierarchique {nopt}\n")
        dopt = 1.0 / nopt
        xopt = np.linspace(dopt / 2, 1 - dopt / 2, nopt)
        qexact = qcoeff(xopt)
        resopt = minimize(
            simopt,
            qinit,
            method="BFGS",
            jac=gradopt,
            args=(d, fx, dx),
            options={"disp": True},
        )
        qopt = resopt.x
        print("Optimisation avec {0:d} paramètres".format(nopt))
        print("Solution    paramètre exact")
        for j in np.arange(nopt):
            print(qopt[j], qexact[j])
        print(f"Approximation error {nopt} : {simopt(qopt, d, fx, dx)}")
        if nopt > 1:
            plt.figure()
            plt.subplot(111)
            plt.plot(np.arange(nopt), [qopt[i] for i in range(nopt)], label = f"Optimum computed {nopt}")
            plt.plot(np.arange(nopt), [qexact[i] for i in np.arange(nopt)], label = "Exact solution")
            plt.legend()
            plt.pause(0.001)
        nopt = 2 * nopt
        qinit = np.ones(nopt)
        qinit[0 : nopt - 1 : 2] = qopt
        qinit[1:nopt:2] = qopt
    input("---> Fin du programme <---")
    sys.exit(0)

if __name__ == "__main__":
    # execute only if run as a script
    main()
    plt.show()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:17:23 2017

@author: yellow
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import scipy.linalg as linalg
import emcee
import random as ran


def get_U(x):
    """
    Returns the potential energy of a function that is a quartic double well of form (x^4-2x^2).

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.


    Returns
    -------
    U : float
        Value of the potential.
    """

    return -np.cos(2*x) + 1


def get_D(x):
    """
    Returns the value of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    D : float
        Value of the diffusion function.
    """
    return (2+np.sin(x))*0.1


def unravelR(x, N):
    """
    input MCMC run x(a vector contains upper triangular elements of R and first N-1 element in P) and output full rate matrix R and stationary distribution P.

    Parameters
    ----------
    x: a vector contains upper triangular elements of R and first N-1 element in P.
    N: number of bins of R matrix.

    Returns
    -------
    R : Full rate matrix.
    P : full stationary distribution
    """

    R_num_par = (N*(N-1))/2 # the number of upper triangular element from rate matrix

    R = np.zeros((N, N))
    U = np.zeros((N, N)) # upper trangular matrix used to get R
    P = np.zeros(N)

    P[0:N-1] = x[R_num_par:]
    P[-1] = (1 - sum(P))


    U[np.triu_indices(N, k=1)] = x[0:R_num_par] # set an upper triangular matrix with diagonal and rest part == 0, the northeastern part would be rate matrix free parameters from x

    P_reciprocal = 1/P
    # scaling the row to get a square reciprocal matrix
    square_P_reciprocal = np.ones((N,N)) * P_reciprocal.reshape(-1, 1)
    # scaling the column to get Pi/Pj coefficient
    P_coeff = square_P_reciprocal * P
    # set a lower triangular matrix to store southwestern elements of R matrix
    L = np.transpose(np.multiply(U, P_coeff))
    R = U + L
    # set diagnal
    np.fill_diagonal(R, -np.sum(R, axis = 0))

    return R, P


def lnlikelihood(x, Nij, N, t=0.5):
    """
    Define log likelihood in MCMC

    Parameters:
    ----------
    x: a vector contains upper triangular elements of R and first N-1 element in P.

    P: stationary distribution, N by 1 array.

    Nij : the count of particle moving from state i to state j. N by N array.

    t: at a time t after the system was in state j, the system is found in state i. Float, Default is 0.5.

    Return:
    ----------
    lnL: loglikelihood used in MCMC; likelihood: given parameters, the probability of the observed data happening

    """

    R_num_par = (N*(N-1))/2 # the number of upper triangular element from rate matrix

    # check if R and P is valid
    for i in xrange(R_num_par):
        if x[i] < 0 :
            return -np.inf

    for i in xrange(R_num_par, len(x)):
        if x[i] < 0 or x[i] > 1:
            return -np.inf
        elif sum(x[R_num_par:]) > 1:
            return -np.inf


    # unravel R to return a full rate matrix
    R, P = unravelR(x, N)

    exp_R_t = linalg.expm(R*t)  #calculate exponential


    # caluluate Nij/sum_i(Nij)
    # Nijsum = np.sum(Nij, axis=0)
    # prob = Nij / Nijsum
    #
    # for i in xrange(N):
    #     for j in xrange(N):
    #         if np.isnan(prob[i,j]):
    #             prob[i,j] = 0


    # print"Nij", Nij
    # print "R", R # check
    # print "exp_R_t", exp_R_t
    # print "prob", prob
    # print "stationary prob", P

    lnL_element = np.multiply(np.log(exp_R_t), Nij)
    lnL = lnL_element.sum()

    # print "lnL", lnL

    return lnL

def countN(traj, bin_edges):
    """
    Nij counts how many times the particles move from i to j

    Parameters:
    -----------
    traj: Trajectory produced by Brownian dynamics.

    bin_edges: bin_edges defines the bin range for rate matrix. [N+1,1] array.

    Return:
    -----------
    Nij: Nij counts how many steps in traj move from i to j. N by N matrix, each element in matrix corresponds to the count Nij.
    """
    N = len(bin_edges) - 1
    nsteps = len(traj)
    Nij = np.zeros([N,N]) # set Nij matrix to store the count, originally all elements equal zero
    bin_width = bin_edges[1]-bin_edges[0]

    for m in xrange(nsteps-1):
        for i in xrange(N):
            if (bin_edges[i+1]>traj[m]) and (traj[m]> bin_edges[i]):
                move = (bin_edges[i+1]+bin_edges[i])/2 - traj[m]
                next_step = (traj[m+1]+move)
                if (next_step > np.pi):
                    j = int(math.floor(abs(next_step - np.pi)/bin_width))
                    Nij[i,j]+=1
                elif (next_step < -np.pi):
                    j = int(N - math.ceil(abs(-np.pi - next_step)/bin_width))
                    Nij[i,j]+=1
                else:
                    for j in xrange(N):
                        if (bin_edges[j+1] > next_step) and (next_step > bin_edges[j]):
                            Nij[i,j]+=1

    return Nij


def runMCMC(x, N, lnlikelihood, Nij, steps, burnt_in, ndim, nwalkers, t = 0.5):
    """
    use emcee package to run MCMC and return a list of R and P from distribution of parameters given traj data.

    Parameters:
    -----------
    x: initialized parameter positions. nwalkers by ndim matrix.

    N:bin numbers of full rate matrix.

    lnlikelihood: loglikelihood function used in MCMC. It is lnlikelihood function defined ahead.

    Nij: Calculated from traj. how many times particles move from state i to j during a period t.

    steps: how many steps MCMC run. int.

    burnt_in: number of steps burnt_in. int.

    ndim: dimension of the system. int.

    nwalkers: number of walkers used in mcmc run. int.

    t: time period used in generating traj, and calculating Nij and lnlikelihood. default = 0.5.

    Return:
    -----------
    outcomes: flatchain of MCMC result. Dimension:  (nwalker*steps) by ndim

    """

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood, args = [Nij, N, t])
    pos, prob, state = sampler.run_mcmc(x, burnt_in)
    sampler.reset()
    sampler.run_mcmc(pos, steps)
    outcomes = sampler.flatchain


    return outcomes, sampler.acceptance_fraction



def generateR(Nij, N, t= 0.5):
    """
    generate the initial rate matrix R.

    Parameters:
    -----------
    Nij: N by N matrix, stores the count of steps that the particle moves from bin i to bin j
    N: number of bins of rate matrix R.
    t : time step when producing the trajectory. int, default = 0.5.

    Return:
    -----------
    x_noise: all initial free parameters in rate matrix and stationary vector.

    """

    # calculate Nij/ sum_column(Nij)
    Nijsum = np.sum(Nij, axis=0)
    M = Nij / Nijsum
    for i in xrange(N):
        for j in xrange(N):
            if np.isnan(M[i,j]):
                M[i,j] = 0

    # set up R and P
    I = np.identity(N)
    R_0 = (M - I)/t
    values, P = scipy.sparse.linalg.eigs(R_0, k=1, sigma = 0)
    # normalize P
    s = sum(P)
    P = [p_i/s for p_i in P]
    P = (np.array(P)).reshape(N)


    # add noise to nonzero element in R
    noise_1 = (np.random.uniform(0.75,1.25,(N*(N-1)/2))) # noise for free parameters in rate matrix
    noise_2 = (np.random.uniform(0.75,1.25,(N-1))) # noise for free parameters in stationary vector
    noise_P = np.multiply(P[0:N-1], noise_2)
    R_0_free_par = R_0[np.triu_indices(N, k=1)]
    noise_R = np.multiply(R_0_free_par, noise_1)
    x_noise = np.append(noise_R, noise_P)
    # print "noise_R", noise_R
    # print "noise_P", noise_P
    # print "x_noise", x_noise

    return x_noise


def minimum_image_convention(x):
    """
    Minimum image convention.

    Parameters:
    -----------
    x: Coordinate of the particle.

    Return:
    -----------
    x_period: The new coordinates after moving according to the periodic boundary conditions.

    """
    pi = np.pi

    if (x < -pi):
        x = x + 2*pi * np.rint(abs(x)/(2*pi))

    if (x > pi):
        x = x - 2*pi * np.rint(abs(x)/(2*pi))

    return x


def main():
    """
    Getting trajectory
    """
    print("reading trajectory...")

    traj = np.load('reproduce_traj.npy')

    print("minimum image convention...")

    traj = [minimum_image_convention(x) for x in traj]


    # Calculate Nij
    print("Calculate Nij...")
    dt = 0.5 # subsampling 500 steps of 0.001ps
    N = 24 # set bin number
    bin_edges = np.linspace(-np.pi,np.pi,N+1)
    Nij = countN(traj, bin_edges)



    # initialize R (detailed balance and P)
    print("start initializing rate matrix...")
    ndim = ((N*(N-1))/2 + (N-1)) # upper triangular matrix elements + n-1 element in P
    nwalkers = 2*ndim

    # store the upper triangular element of full R matrix and the first N-1 element in P matrix as initilized parameter for MCMC
    x = np.zeros((nwalkers, ndim))

    for i in xrange(nwalkers):
        x[i, :] = generateR(Nij, N)



    # plug in R_0 and run MCMC
    print("MCMC run...")
    steps = 1000
    burnt_in = 1000

    #Dimension:  (nwalker*steps) by ndim. MCMC outcomes flatchain
    outcomes, accep_rate = runMCMC(x, N, lnlikelihood, Nij, steps, burnt_in, ndim, nwalkers, dt)


    print("Calculating position and its corresponding D...")
    bin_width = bin_edges[1] - bin_edges[0]
    D_chain = np.zeros([nwalkers * steps, N-1])


    # Coordinate
    Q = np.zeros([N,1])
    for i in xrange(N):
        Q[i] = (bin_edges[i] + bin_edges[i+1])/2


    # caluluate P_true, R_true
    I = np.identity(N)
    Nijsum = np.sum(Nij, axis=0)
    prob = Nij / Nijsum
    R_true = (prob - I)/dt

    F = get_U(Q)
    exp_beta_f = [np.exp(-f) for f in F]
    sum_exp_f = sum(exp_beta_f)
    P_true = exp_beta_f / sum_exp_f

    for i in xrange(nwalkers * steps):
        R, P = unravelR(outcomes[i,:], N)
        for j in xrange(N-1):
            D_chain[i,j] = ((bin_width)**2 * R[j+1, j] * (P[j]/P[j+1])**(0.5))



    D = np.zeros([N-1,1])
    D = np.mean(D_chain, axis = 0)
    np.save("truevalue_D_traj=%i,bin=%i step=%i, burn=%i" %(len(traj), N, steps, burnt_in), D)



    D_true = np.zeros([N-1,1])
    for i in xrange(N-1):
        D_true[i] = ((bin_width)**2 * R_true[i+1, i] * (P_true[i]/P_true[i+1])**(0.5))
    np.save("truevalue_D_traj_traj = %i,bin=%i step=%i, burn=%i" %(len(traj), N, steps, burnt_in), D_true)

    print "accep_rate", accep_rate
    np.save("truevalue accep_rate traj = %i,bin=%i step=%i, burn=%i" %(len(traj), N, steps, burnt_in), accep_rate)

    """
    drawing figures
    """
    print("plot true D and estimated D...")
    plt.figure()
    plt.plot(Q[0:-1],D,c = 'y', label = 'Bayesian')
    plt.plot(Q[0:-1], get_D(Q)[0:-1],label = 'true')
    plt.plot(Q[0:-1], D_true, c = 'r', label = 'traj')
    # plt.xlim(-2, 2)
    # plt.ylim(0, 10)
    plt.xlabel('position(x)')
    plt.ylabel('Diffusion coefficient(D)')
    plt.title('Bayesian traj = %i,bin=%i step=%i, burn=%i' %(len(traj), N, steps, burnt_in))
    plt.legend(loc = 2)
    plt.show()
    plt.savefig("truevalue_bin_%i_new" %N)

    return





if __name__ == "__main__":
    main()

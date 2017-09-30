#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 10:56:53 2017

@author: yellow
"""

import numpy as np
import matplotlib.pyplot as plt



def get_U(x):
    """
    Returns the potential energy of a function.

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

def get_F(x):
    """
    Returns the force of the potential energy.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    F : float
        Value of the force.
    """
    return -2*np.sin(2*x)

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

def get_dD(x):
    """
    Returns the value of the divergence of the diffusion function at x.

    Parameters
    ----------
    x : float
        The location in the one-dimensional coordinate.

    Returns
    -------
    dD : float
        Value of the divergence of the diffusion function.
    """
    return np.cos(x)*0.1

def brownian_dynamics(nsteps,x0,force_method,get_divD,get_D,dt=0.001,kT=1.0):
    """
    Runs brownian dynamics.

    Parameters
    ----------
    nsteps : int
        Number of dynamics steps to run.
    x0 : 1d array-like
        Starting coordinate from which to run the dynamics.
    force_method : subroutine
        Subroutine that yields the force of the system.  Must take in an array of the same shape as x0, and return a force vector of the same size.
    dt : float, optional
        Timestep for the dynamics.  Default is 0.001, which is a decent choice for a harmonic with force constant 1.
    get_D : Subroutine
         Subroutine that yields diffusion tensor for the system.
    get_divD: Subroutine
         Subroutine that yields the divergence of D
    kT : float, optional
        Boltzmann factor for the system (k_B * T).  Default is natural units (1.0)


    Returns
    -------
    traj : 2D array
        Two dimensonal array, where the element i,j is the j'th coordinate of the system at timestep i.

    """
    # Set defaults and parameters
    ndim = len(x0) # Find dimensionality of the system
    # Propagate Brownian dynamics according to the Euler-Maruyama method.
    traj = []
    cfg = np.copy(x0)
    sig = np.sqrt(2.* dt) # Perform some algebra ahead of time.
    for j in xrange(int(nsteps)):
        D = get_D(cfg) # Typecast to array for easy math.
        c = np.sqrt(D) # Square root of Diffusion matrix hits the noise
        rando = np.dot(c,np.random.randn(ndim))
        force = np.dot(D,force_method(cfg))
        divD = get_divD(cfg)
        cfg += dt * force + sig * rando + divD * dt
        traj.append(np.copy(cfg))
    return np.array(traj)


def main():
    """
    Getting trajectory
    """
    print("start generating trajectory")
    nsteps = 100000000
    x0 =  np.array([0.0])
    kT = 1.0
    dt = 0.001
    subsampling = 500

    traj = brownian_dynamics(nsteps,x0,get_F,get_dD,get_D,dt=dt,kT=kT)
    # subsampling: save every 500 steps
    traj = traj[::subsampling]
    print "len of traj", np.shape(traj)
    np.save('reproduce_traj', traj)




if __name__ == "__main__":
    main()

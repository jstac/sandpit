#%% -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:21:02 2018

@author: dongchenzou
"""
import numpy as np
import numpy.matlib as npm
import numpy.linalg as npl

eye = npm.identity
solve = npl.solve
inv = npl.inv



def doubleo(A,C,Q,R):
    """
    Description
    ==========
    This program uses the "doubling algorithm" to solve the
    Riccati matrix difference equations associated with the
    Kalman filter.  A is nxn, C is kxn, Q is nxn, R is kxk.
    The program returns the gain K and the stationary covariance
    matrix of the one-step ahead errors in forecasting the state.
    The program creates the Kalman filter for the following system:
          
        x(t+1) = A * x(t) + e(t+1)
          y(t) = C * x(t) + v(t)
             
    where E e(t+1)*e(t+1)' = Q, and E v(t)*v(t)' = R, and v(s) is orthogonal
    to e(t) for all t and s. The program creates the observer system
      
        xx(t+1) = A * xx(t) + K * a(t)
           y(t) = C * xx(t) + a(t)
               
    where K is the Kalman gain, S = E (x(t) - xx(t))*(x(t) - xx(t))', and
    a(t) = y(t) - E[y(t)| y(t-1), y(t-2), ... ], and xx(t)=E[x(t)|y(t-1),...].
    
    Input
    =========
    A: n x n numpy matrix
    C: k x n numpy matrix
    Q: n x n numpy matrix
    R: k x k numpy matrix
    
    Output
    =========
    K: Kalman gain, n x k numpy matrix
    S: E(x(t) - xx(t))*(x(t) - xx(t))', n x n numpy matrix
    
    Note
    =========
    By using DUALITY, control problems can also be solved.
    
    """
    a0 = A.T
    b0 = C.T @ solve(R, C)
    g0 = Q
    tol = 1e-15
    dd = 1
    
    ss = np.max(A.shape)
    
    v = eye(ss)
    
    while dd > tol:
        a1 = a0 @ solve((v + b0 @ g0), a0)
        b1 = b0 + a0 @ solve((v + b0 @ g0), (b0 @ a0.T))
        g1 = g0 + a0.T @ g0 @ solve((v + b0 @ g0), a0)
        k1 = A @ g1 @ C.T @ inv(C @ g1 @ C.T + R)
        k0 = A @ g0 @ C.T @ inv(C @ g0 @ C.T + R)
        dd = np.max(abs(k1-k0))
        a0 = a1
        b0 = b1
        g0 = g1
    
    K = k1
    S = g1
    

    
    return K, S

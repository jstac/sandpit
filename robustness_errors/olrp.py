#%% -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:11:47 2018

@author: dongchenzou
"""

import numpy as np
import numpy.matlib as npm
import numpy.linalg as npl
from doubleo import doubleo

zeros = np.zeros
eye = npm.identity
solve = npl.solve
eig = npl.eig
sqrt = np.sqrt



def olrp(beta, A, B, Q, R, W=0):
    """
    Description
    =========
    OLRP can have arguments: (beta,A,B,Q,R) if there are no cross products
    (i.e. W=0). Set beta=1, if there is no discounting.
    OLRP calculates f of the feedback law:
		u = -fx
    that maximizes the function:
          sum {beta^t [x'Qx + u'Ru +2x'Wu]}
    subject to 
		x[t+1] = Ax[t] + Bu[t] 
    where x is the nx1 vector of states, u is the kx1 vector of controls,
    A is nxn, B is nxk, Q is nxn, R is kxk, W is nxk.
    Also returned is p, the steady-state solution to the associated 
    discrete matrix Riccati equation.
    
    Input
    =========
    beta: discount factor, 0 <= beta <= 1
    A: n x n numpy matrix
    B: n x k numpy matrix
    Q: n x n numpy matrix
    R: k x k numpy matrix
    W: n x k numpy matrix, default zero matrix
    
    Output
    =========
    f: matrix that solves u = -fx, k x n numpy matrix
    p: steady state solution to the Riccati equation, n x n numpy matrix
    """  
    L = [A, B, Q, R]    
    for i,mat in enumerate(L):
        if type(mat) == np.matrixlib.defmatrix.matrix:
            continue
        else:    
            L[i] = np.asmatrix(mat)
    A = L[0]
    B = L[1]
    Q = L[2]
    R = L[3]


      
    m = np.max(A.shape)
    rb, cb = B.shape
    
    if W==0:
       W = np.matrix(zeros((m,cb)))



    if min(abs(eig(R)[0])) > 1e-5:
       A = sqrt(beta) * (A - B @ solve(R, W.T))
       B = sqrt(beta) * B
       Q = Q - W @ solve(R, W.T)
       k, s = doubleo(A.T, B.T, Q, R)
       
       f = k.T + solve(R, W.T)
       p = s
    
    
    else:
      p0 = -.01 * eye(m)
      dd = 1
      it = 1
      maxit = 1000
      tol = 1e-6 # for greater accuracy set it to 1e-10

      while dd>tol and it<=maxit: 
        f0 = solve((R + beta * B.T @ p0 @ B), (beta * B.T @ p0 @ A + W.T))
        p1 = beta * A.T @ p0 @ A + Q - (beta * A.T@ p0 @ B + W) @ f0
        f1 = solve((R + beta * B.T @ p1 @ B), (beta * B.T @ p1 @ A + W.T))
        dd = np.max(abs(f1-f0))    
        it = it+1
        p0 = p1

      f = f1
      p = p0
    
      if it>=maxit:
          print('WARNING: Iteration limit of 1000 reached in OLRP')
          
    
    return f, p
        
    
    
    


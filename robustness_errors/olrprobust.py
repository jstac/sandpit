#%% -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:38:05 2018

@author: dongchenzou
"""
import numpy as np
import numpy.matlib as npm
import numpy.linalg as npl
from olrp import olrp

zeros = np.zeros
eye = npm.identity
inv = npl.inv


def olrprobust(beta,A,B,C,Q,R,sig):
    """
    Description
    =========    
    Solves the robust control problem 
    
        min sum beta^t(x'Qx + u'Ru) 
    
    for the state space system
    
        x` = Ax + Bu + Cw
    
    sig < 0 indicates preference for robustness.
    sig = -1/theta where theta is the robustness multiplier.
    Please note that because this is a MINIMUM problem, the convention
    is that Q and R are `positive definite' matrices (subject to the usual
    detectability qualifications).  
     
    olrprobust solves the problem by tricking it into a stacked olrp problem.  
    as in Hansen-Sargent, Robustness in Macroeconomics, chapter 2.
    The optimal control with observed state is 
        
        u_t = - F x_t
      
    The value function is -x'Px; note that the program returns a positive
    semi-definite P. Pt is the matrix D(P) where D(.) is the operator
    described in Hansen-Sargent.  The conservative measure
    of continuation value is -y' D(P) y = -y' Pt y,
    where y=x`, next period's state.
    The program also returns a worst-case shock matrix K
    where w_{t+1} = K x_t is the worst case shock.
    
    Input
    =========
    beta: discount factor, 0 <= beta <= 1, float
    A: n x n numpy matrix
    B: n x k numpy matrix
    C: n x k numpy matrix
    Q: n x n numpy matrix
    R: k x k numpy matrix
    sig: robustness parameter, float
    
    Output
    =========
    F: control matrix, k x n numpy matrix
    K: worst-case shock matrix, k x n numpy matrix
    P: positive semi-definite matrix, n x n numpy matrix
    Pt: D(P), n x n numpy matrix
    """
    L = [A, B, C, Q, R]    
    for i,mat in enumerate(L):
        if type(mat) == np.matrixlib.defmatrix.matrix:
            continue
        else:    
            L[i] = np.asmatrix(mat)
    A = L[0]
    B = L[1]
    C = L[2]
    Q = L[3]
    R = L[4]

      
    theta = -1/sig
    Ba = np.hstack([B, C])
    rR, cR = R.shape
    rC, cC = C.shape
    Ra1 = np.hstack([R, zeros((rR,cC))])
    Ra2 = np.hstack([zeros((cC,cR)), -beta * eye(cC) * theta]) # there is only one shock
    Ra = np.vstack([Ra1, Ra2])
    
    f, P = olrp(beta,A,Ba,Q,Ra)
    rB, cB = B.shape
    F = f[:cB,:]
    rf, cf = f.shape
    K = -f[cB:rf,:]
    Pt = P +  theta**(-1) * P @ C @ inv(eye((C.T @ P @ C).shape[0]) - theta**(-1) * C.T @ P @ C) @ C.T @ P

    return F, K, P, Pt
         
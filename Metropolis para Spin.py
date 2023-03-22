"""
Created on Fri Sep 16 08:47:21 2022

@author: Geraldo Siqueira
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from numba import jit

J = 1
s=1
Kb = 1

@jit
def IniSpins(L):
    S = 2*np.random.randint(2, size = [L, L]) - 1
    return S
 
@jit   
def DeltaE (S, i, j, L):
    D_E = - 2*J*S[i,j]*(S[(i-1)%L,j] + S[(i+1)%L,j] + S[i, (j-1)%L] + S[i, (j+1)%L])
    return D_E
    
@jit   
def MCStep(S, L, T):
    for k in range (0,L*L):
        i = random.randint(0,L-1)
        j = random.randint(0,L-1)
        S[i,j] = S[i,j]*(-1)
        
        D_E = DeltaE(S, i, j, L)
        if D_E > 0:
            u = random.uniform(0,1)
            p = np.exp((-D_E)/(Kb*T))
            if u > p:
                S[i,j] = S[i,j]*(-1)
    return S

L = 50
MCS = 10000
T = 0.5

S = IniSpins(L)
for n in range(MCS):
    S = MCStep(S, L, T)

plt.pcolor(S, cmap = 'bwr')
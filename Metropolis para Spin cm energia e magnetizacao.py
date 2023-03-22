# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:56:07 2022

@author: geral
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
    E = 0
    S = 2*np.random.randint(2, size = [L, L]) - 1
    for i in range (L):
        for j in range (L):
            E += -J*S[i,j]*(S[(i-1)%L,j] + S[(i+1)%L,j] + S[i, (j-1)%L] + S[i, (j+1)%L])
    return S, E
 
@jit   
def DeltaE (S, i, j, L):
    D_E = - 2*J*S[i,j]*(S[(i-1)%L,j] + S[(i+1)%L,j] + S[i, (j-1)%L] + S[i, (j+1)%L])
    return D_E
    
@jit    
def MCStep(S, L, E, T):
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
                D_E = 0
        E += D_E
    return S, E

L = 100
MCS = 10000
T = 3.5

S, E = IniSpins(L)
print(E)
M = np.zeros(MCS)
Energia = np.zeros(MCS)   

for n in range(MCS):
    S, E = MCStep(S, L, E, T)
    M[n] = (1/(L*L))*np.sum(S)
    Energia[n] = E



plt.figure(1)
plt.pcolor(S, cmap = 'bwr')
plt.figure(2)
plt.plot(M)
plt.title("Magnetização em função de n (T = 3.5)")
plt.figure(3)
plt.plot(Energia)
plt.title("Energia em função de n (T = 3.5)")
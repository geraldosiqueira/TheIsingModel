
"""
Created on Fri Sep 16 10:55:46 2022

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
#MCS = 100000
T = [0.5, 1.0, 1.5, 1.75, 2.0, 2.25, 2.3, 2.35, 2.5, 2.75, 3.0, 3.5]

M_media = np.zeros(len(T))
E_media = np.zeros(len(T))
varM = np.zeros(len(T))
varE = np.zeros(len(T))
S, E = IniSpins(L)

for m in range (len(T)):
    if T[m] < 2.2:
        MCS = 10000
    else:
        MCS = 20000
    M = np.zeros(MCS)
    Energia = np.zeros(MCS)   
    for n in range(MCS):
        S, E = MCStep(S, L, E, T[m])
        M[n] = (1/(L*L))*np.sum(S)
        Energia[n] = E
    M_media[m] = np.mean(M[6*(MCS-1)//10:MCS-1])
    E_media[m] = np.mean(Energia[6*(MCS-1)//10:MCS-1])
    varM[m] = np.var(M[6*(MCS-1)//10:MCS-1])
    varE[m] = np.var(Energia[6*(MCS-1)//10:MCS-1])

plt.figure(1)
plt.plot(T, M_media, '.-')
plt.ylabel(r'$\langle M \rangle$')
plt.xlabel("T")
plt.title("Magnetização média em função da Temperatura")
#plt.savefig("M em funcao de T.png")
plt.figure(2)
plt.plot(T, E_media, '.-')
plt.ylabel(r'$\langle E \rangle$')
plt.xlabel("T")
plt.title("Energia média em função da Temperatura")
#plt.savefig("E em funcao de T.png")
plt.figure(3)
plt.plot(T, varM, '.-')
plt.ylabel(r'$\Delta M ^2$')
plt.xlabel("T")
plt.title("Desvio quadrático da magnetização em função da Temperatura")
#plt.savefig("M em funcao de T.png")
plt.figure(4)
plt.plot(T, varE, '.-')
plt.ylabel(r'$\Delta E ^2$')
plt.xlabel("T")
plt.title("Desvio quadrático da energia em função da Temperatura")
#plt.savefig("M em funcao de T.png")
"""
Created on Fri Sep 23 08:26:50 2022

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
MCS = 800
T = 3.0
dt = 0.01

#S = IniSpins(L)

s = IniSpins(L)
y, x = np.meshgrid(np.arange(0,L), np.arange(0, L))
fig, ax = plt.subplots(figsize=(4,4))
fig.suptitle("Spins em um ferromagnético")
confs = ax.pcolormesh(x, y, s, cmap="bwr")
#ax.set_xlabel("x")
#ax.set_ylabel("y")

def init():
    confs.set_array([])
    return confs

def animate(n):
    si = MCStep(s, L, T)
    confs.set_array(si.ravel())
    return confs

ani = FuncAnimation(fig, animate, frames=MCS, interval = 50)

plt.show()
ani.save("animação spins ferromagnéticos T = 3.0.gif")


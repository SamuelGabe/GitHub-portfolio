
import numpy as np
import matplotlib.pyplot as plt
import math as math
import scipy.constants as con
import time
from scipy.integrate import odeint, solve_ivp


## non ODEINT version
####### although tbf i'm SURE that odeint isn't the problem

# Euler method python program

# set rough values for constants (taken from wdm1 i think)
h100 = 0.678
Xh = 0.76
Yhe = 1 - Xh
omegaL = 0.692; omegaM = 0.308; omegaB = 0.0482
T = 1e4

# all in SI and from the paper
a = 7.982e-11; b = 0.7480; T0 = 3.148; T1 = 7.036e5
alpha = a/(np.sqrt(T/T0)*(1+np.sqrt(T/T0))**(1-b)*(1+ np.sqrt((T/T1))**(1+b)))
alpha = alpha/(1.0e6)
h_SI = (h100*1.0e5/((3.086e22)))
# dummy value for now, we'll replace that with real C vals later
C = 3


# euler's method 
def euler_method(f, y0, x0, x_end, step):
    # Initialize the solution array
    x = np.arange(x0, x_end, step)
    y = np.zeros(len(x))
    y[0] = y0
    
    # Iterate 
    for i in range(1, len(x)):
        y[i] = y[i-1] + step*f(x[i-1], y[i-1])
    
    
    #print(x)
    return x, y

# the function that actually has our equation in it
def f(z, Q):
    
    Hz = h_SI*(omegaM*((1+z)**3) + omegaL)**0.5
    
    # comoving quanty (so nion has has a factor of z^3 in it), so you need to devide it another comoving quantity
    # that was why this didn't work for so long
    av_dens2 = ((3.0*h_SI*h_SI)/(8.0*con.pi*con.G))*omegaB*Xh
    n_dot_ion = 3e50/((3.086e22)**3)  # in s^-1m^-3  (i'm pretty sure)
    
    mH2 = 1.673556692e-27
    av_nH2 = av_dens2/mH2
    
    trec = (C*alpha*av_nH2*((1+z)**3)*(1 + Yhe/(4*Xh)))**(-1)
    
    dQdz = ((Hz*(1 + z))**(-1)) * (Q/trec - n_dot_ion/av_nH2)
    return dQdz

# set in initial Q, the z range, the number of steps and calc the step size
y0 = 0; x0 = 20; x_end = 0; N = 1000; step = (x_end - x0)/(N - 1)

# run the thing
z, Q = euler_method(f, y0, x0, x_end, step)

#plot
plt.plot(z,Q, linewidth =2, color = 'k', label = 'euler', linestyle = '--')
plt.xlabel('z', fontsize = 20)
plt.ylabel('Q(t)', fontsize = 20)
plt.legend(loc ="upper right",fontsize = 18)
plt.ylim(0,1)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

arr_end = 300

t_arr = np.linspace(0, 10, 1000)[:arr_end]

n_0 = np.load('array_n_{n}_lambda_{lmb}.npy'.format(n=0, lmb=2))[:arr_end]
n_1 = np.load('array_n_{n}_lambda_{lmb}.npy'.format(n=1, lmb=2))[:arr_end]
n_2 = np.load('array_n_{n}_lambda_{lmb}.npy'.format(n=2, lmb=2))[:arr_end]
n_3 = np.load('array_n_{n}_lambda_{lmb}.npy'.format(n=3, lmb=2))[:arr_end]
"""
n_0 = np.load('array_n_{n}_lambda_{lmb}_nrwa.npy'.format(n=0, lmb=2))[:arr_end]
n_1 = np.load('array_n_{n}_lambda_{lmb}_nrwa.npy'.format(n=1, lmb=2))[:arr_end]
n_2 = np.load('array_n_{n}_lambda_{lmb}_nrwa.npy'.format(n=2, lmb=2))[:arr_end]
n_3 = np.load('array_n_{n}_lambda_{lmb}_nrwa.npy'.format(n=3, lmb=2))[:arr_end]
"""
plt.plot(t_arr, n_0, 'b', label='$|0\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)')
#plt.plot(t_arr, n_1, label='$|1\\rangle|e\\rangle\\rightarrow|e\\rangle$ (non_RWA)')
#plt.plot(t_arr, n_2, label='$|2\\rangle|e\\rangle\\rightarrow|e\\rangle$ (non_RWA)')
plt.plot(t_arr, n_3, 'r', label='$|3\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)')
plt.xlabel('$t\omega_0$')
plt.ylabel('Probability')
plt.ylim(0, 1.4)
plt.xlim(0, t_arr[-1])
plt.legend()
plt.show()
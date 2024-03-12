import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Quadrant matrix size.
n = 81
lambda_over_omega = 1
hbar = 1

# Initial vector for excited state
v_e_0 = np.append(np.zeros(n), np.ones(n))

mat1 = mat4 = np.diag(np.arange(-0.5, n-1, 1))

# Populate matrix 2
mat2_rwa = np.zeros((n, n))
position = np.array([1, 0])

# Now put numbers in matrix 2
for i in range(n-1):
    mat2_rwa[position[0], position[1]] = np.sqrt(i+1) * lambda_over_omega
    position += 1

# Create matrix 3. Matrix 3 is just Matrix 2 transposed.
mat3_rwa = np.transpose(mat2_rwa)

# Matrix 4 is the same as Matrix 1
mat4 = mat1

# Clever trick here: turns out Matrix 2 and Matrix 3 are the same without the RWA, and 
# if you add together the RWA versions of the matrices, you get the version without RWA
mat2_nonrwa = mat3_nonrwa = mat2_rwa + mat3_rwa

# Create big matrix by making the rows then putting the rows together.
M_rwa = np.append(np.append(mat1, mat2_rwa, axis=0), np.append(mat3_rwa, mat4, axis=0), axis=1)
M_nonrwa = np.append(np.append(mat1, mat2_nonrwa, axis=0), np.append(mat3_nonrwa, mat4, axis=0), axis=1)

# Now it's time to bring in the time
t_arr = np.linspace(0, 5, 1000)

# First do RWA matrix. Probabilties of e -> e and e -> g
p_ee_rwa = np.zeros(len(t_arr))
p_eg_rwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavefunction_t_rwa = np.zeros((len(t_arr), 2*n))

for i in range(len(t_arr)):
    M_t_rwa = expm(1j * M_rwa * t_arr[i] / hbar)

    # Vector of psi(t) when psi(0) is the excited state, with RWA
    v_e_t_rwa = np.matmul(M_t_rwa, v_e_0)

    # Probabilities get multiplied by n for some reason so we need to divide by n to normalise.
    probabilities = np.abs(v_e_t_rwa)**2 / n
    p_ee_rwa[i] = np.sum(probabilities[n:])
    p_eg_rwa[i] = np.sum(probabilities[:n])
    wavefunction_t_rwa[i] = probabilities

# Now time for non-RWA matrix. Probabilties of e -> e and e -> g
p_ee_nonrwa = np.zeros(len(t_arr))
p_eg_nonrwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavefunction_t_nonrwa = np.zeros((len(t_arr), 2*n))

for i in range(len(t_arr)):
    M_t_nonrwa = expm(1j * M_nonrwa * t_arr[i] / hbar)
    v_e_t_nonrwa = np.matmul(M_t_nonrwa, v_e_0)

    # Probabilities get multiplied by n for some reason so we need to divide by n to normalise.
    probabilities = np.abs(v_e_t_nonrwa)**2 / n
    p_ee_nonrwa[i] = np.sum(probabilities[n:])
    p_eg_nonrwa[i] = np.sum(probabilities[:n])
    wavefunction_t_nonrwa[i] = probabilities

# Plot the graph of probabilities of being found in excited state for RWA and non RWA
plt.plot(t_arr, p_ee_nonrwa, label='$|e\\rangle\\rightarrow|e\\rangle$ (non-RWA)')
#plt.plot(t_arr, p_eg_nonrwa, label='$|e\\rangle\\rightarrow|g\\rangle$')
#plt.plot(t_arr, p_ee_nonrwa + p_eg_nonrwa, label='Sum of both')
plt.xlabel('$t$')
plt.ylabel("Proability")

plt.plot(t_arr, p_ee_rwa, 'g-.', label='$|e\\rangle\\rightarrow|e\\rangle$ (RWA)')
"""
plt.plot(t_arr, p_eg_rwa, label='$|e\\rangle\\rightarrow|g\\rangle$')
plt.plot(t_arr, p_ee_rwa + p_eg_rwa, label='Sum of both')
plt.xlabel('$t$')
plt.ylabel("Proability")
"""

plt.legend()
plt.show()

"""
# Plot individual state probabilities
plt.plot(t_arr, wavefunction_t_nonrwa[:, n + 1], label='$|1\\rangle|e\\rangle$')
plt.plot(t_arr, wavefunction_t_nonrwa[:, n + 2], label='$|2\\rangle|e\\rangle$')
plt.plot(t_arr, wavefunction_t_nonrwa[:, n + 3], label='$|3\\rangle|e\\rangle$')
plt.xlabel('$t$')
plt.ylabel("Proability")
plt.legend()
plt.show()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

#region Define variables and populate matrixes
# Quadrant matrix size.
n = 16
lambda_over_omega = 2.
hbar = 1
t_max = 30

initial_state_number = 0

# Initial vector for excited state
v_0 = np.zeros(2*n)
v_0[n + initial_state_number] = 1.

# Matrix elements are the same and are both diagonals
mat1 = np.diag(np.arange(-0.5, n-1, 1))
mat4 = np.diag(np.arange(0.5, n, 1))

# Populate matrix 2
mat2_rwa = np.zeros((n, n))
position = np.array([0, 1])

# Now put numbers in matrix 2
for i in range(n-1):
    mat2_rwa[position[0], position[1]] = np.sqrt(i+1) * lambda_over_omega
    position += 1

# Create matrix 3. Matrix 3 is just Matrix 2 transposed.
mat3_rwa = np.transpose(mat2_rwa)

# Clever trick here: turns out Matrix 2 and Matrix 3 are the same without the RWA, and 
# if you add together the RWA versions of the matrices, you get the version without RWA
mat2_nonrwa = mat3_nonrwa = mat2_rwa + mat3_rwa

# Create big matrix by making the rows then putting the rows together.
M_rwa = np.append(np.append(mat1, mat2_rwa, axis=0), np.append(mat3_rwa, mat4, axis=0), axis=1)
M_nonrwa = np.append(np.append(mat1, mat2_nonrwa, axis=0), np.append(mat3_nonrwa, mat4, axis=0), axis=1)

# Now it's time to bring in the time
t_arr = np.linspace(0, t_max, 1000)

#endregion

#region Rotating Wave Approximation Calculations

# First do RWA matrix. Probabilties of e -> e and e -> g
p_ee_rwa = np.zeros(len(t_arr))
p_eg_rwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevectors_t_rwa = np.zeros((len(t_arr), 2*n))

for i in range(len(t_arr)):
    M_t_rwa = expm(1j * M_rwa * t_arr[i] / hbar)

    # Vector of psi(t) when psi(0) is the excited state, with RWA
    v_t_rwa = np.matmul(M_t_rwa, v_0)

    probabilities = np.conjugate(v_t_rwa) * v_t_rwa
    p_ee_rwa[i] = np.sum(probabilities[n:])
    p_eg_rwa[i] = np.sum(probabilities[:n])
    wavevectors_t_rwa[i] = probabilities

#endregion

#region Not with rotating wave approximation

# Now time for non-RWA matrix. Probabilties of e -> e and e -> g
p_ee_nonrwa = np.zeros(len(t_arr))
p_eg_nonrwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevectors_t_nonrwa = np.zeros((len(t_arr), 2*n))

# Do the calculations for non rwa
for i in range(len(t_arr)):
    M_t_nonrwa = expm(1j * M_nonrwa * t_arr[i] / hbar)
    v_t_nonrwa = np.matmul(M_t_nonrwa, v_0)

    probabilities = np.conjugate(v_t_nonrwa) * v_t_nonrwa
    p_ee_nonrwa[i] = np.sum(probabilities[n:])
    p_eg_nonrwa[i] = np.sum(probabilities[:n])
    wavevectors_t_nonrwa[i] = probabilities

#endregion

# Plot the graph of probabilities of being found in excited state for RWA and non RWA
plt.plot(t_arr, p_ee_nonrwa, label='$|{initial}\\rangle|e\\rangle\\rightarrow|e\\rangle$ (non RWA)'.format(initial=initial_state_number))
plt.plot(t_arr, p_ee_rwa, label='$|{initial}\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)'.format(initial=initial_state_number))
#plt.plot(t_arr, p_eg_rwa, 'r-.', label='$|1\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)')
plt.xlabel('$\\omega_at$')
plt.ylabel("$P_{|e\\rangle}(t)$")
plt.ylim(bottom=0., top=1.3)
plt.xlim(0, t_max)

np.save('array_n_{n}_lambda_{lmb}_nrwa'.format(n=initial_state_number, lmb=lambda_over_omega), p_ee_nonrwa)

plt.legend()
plt.show()

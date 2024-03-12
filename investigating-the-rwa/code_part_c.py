import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import math

# Quadrant matrix size.
n = 20
hbar = 1

alpha = 1
lambda_over_omega = 0.1
t_max = 50

# Initial vector for excited state
v1 = np.zeros(n)
v2 = np.zeros(n)

# Populate v2. Coefficients follow this formula
for i in range(n):
    v2[i] = alpha ** i / math.factorial(i)

v = np.append(v1, v2)

# Normalise
v = v / np.sqrt(np.sum(v**2))

mat1 = np.diag(np.arange(-0.5, n-1, 1))
mat4 = np.diag(np.arange(0.5, n, 1))

# Populate matrix 2
mat2_JC = np.zeros((n, n))
position = np.array([1, 0])

# Now put numbers in matrix 2
for i in range(n-1):
    mat2_JC[position[0], position[1]] = np.sqrt(i+1) * lambda_over_omega
    position += 1

# Create matrix 3. Matrix 3 is just Matrix 2 transposed.
mat3_JC = np.transpose(mat2_JC)

# Clever trick here: turns out Matrix 2 and Matrix 3 are the same without the RWA, and 
# if you add together the RWA versions of the matrices, you get the version without RWA
mat2 = mat3 = mat2_JC + mat3_JC

# Create big matrix by making the rows then putting the rows together.
M_JC = np.append(np.append(mat1, mat2_JC, axis=1), np.append(mat3_JC, mat4, axis=1), axis=0)
M = np.append(np.append(mat1, mat2, axis=1), np.append(mat3, mat4, axis=1), axis=0)

# Now it's time to bring in the time
t_arr = np.linspace(0, t_max, 1000)

# First do RWA matrix. Probabilties of e -> e and e -> g
p_e_JC = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevector_t_JC = np.zeros((len(t_arr), 2*n))

for i in range(len(t_arr)):
    M_t_JC = expm(1j * M_JC * t_arr[i] / hbar)

    # Vector of psi(t) when psi(0) is the excited state, with RWA
    v_t_JC = np.matmul(M_t_JC, v)

    # Probabilities get multiplied by n for some reason so we need to divide by n to normalise.
    probabilities = np.abs(v_t_JC)**2
    p_e_JC[i] = np.sum(probabilities[n:])
    wavevector_t_JC[i] = probabilities

# Now time for non-RWA matrix. Probabilties of e -> e and e -> g
p_e = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevector_t = np.zeros((len(t_arr), 2*n))

for i in range(len(t_arr)):
    M_t = expm(1j * M * t_arr[i] / hbar)
    v_t = np.matmul(M_t, v)

    # Absolute squared gives us probability. Then sum up second half of array.
    probabilities = np.abs(v_t)**2
    p_e[i] = np.sum(probabilities[n:])
    wavevector_t[i] = probabilities

# Plot the graph of probabilities of being found in excited state for RWA and non RWA
plt.plot(t_arr, p_e, label='$|\\alpha\\rangle|e\\rangle\\rightarrow|e\\rangle$ (non-RWA)', linewidth=0.7)
plt.plot(t_arr, p_e_JC, label='$|\\alpha\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)', linewidth=0.7)
plt.xlabel('$\\omega_at$')
plt.ylabel("Proability")
plt.xlim(0, t_max)
plt.ylim(0, 1.3)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Quadrant matrix size.
n = 20
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

# Create big matrix by making the rows then putting the rows together.
M_rwa = np.append(np.append(mat1, mat2_rwa, axis=0), np.append(mat3_rwa, mat4, axis=0), axis=1)

# Now it's time to bring in the time
t_arr = np.linspace(0, 5, 1000)

# Probabilties of e -> e and e -> g
p_ee = np.zeros(len(t_arr))
p_eg = np.zeros(len(t_arr))

for i in range(len(t_arr)):
    M_t = expm(1j * M_rwa * t_arr[i] / hbar)
    v_e_t = np.matmul(M_t, v_e_0)

    # Probabilities get multiplied by n for some reason so we need to divide by n to normalise.
    probabilities = np.abs(v_e_t)**2 / n
    p_ee[i] = np.sum(probabilities[n:])
    p_eg[i] = np.sum(probabilities[:n])

plt.plot(t_arr, p_ee, label='$|e\\rangle\\rightarrow|e\\rangle$')
plt.plot(t_arr, p_eg, label='$|e\\rangle\\rightarrow|g\\rangle$')
plt.plot(t_arr, p_ee + p_eg, label='Sum of both')
plt.xlabel('$t$')
plt.ylabel("Proability")
plt.legend()
plt.show()
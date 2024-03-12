import numpy as np
import matplotlib.pyplot as plt

n = 20
diag = np.zeros(n)

# create axis variables
hbar_omega_array = np.linspace(0, 1, 1000)
lambda_over_omega_array = np.linspace(0, 1, 1000)

# Create diagonal terms
# Start off with 1/2, add 2 terms, then i += 1, add 2 terms, i += 2, add 2, i += 1, add 2, etc.

# Matrix elements are the same and are both diagonals
mat1 = np.diag(np.arange(-0.5, n-1, 1))
mat4 = np.diag(np.arange(0.5, n, 1))

# Create off-diagonal terms. These vary and we plot the graph with them.
e_vals_array_rwa = np.zeros((1000, 2*n))

# RWA
i = 0
for lambda_over_omega in lambda_over_omega_array:
    mat2_rwa = np.zeros((n, n))
    position = np.array([1, 0])

    # Now put numbers in matrix 2
    for j in range(n-1):
        mat2_rwa[position[0], position[1]] = np.sqrt(j+1) * lambda_over_omega
        position += 1

    # Create matrix 3. Matrix 3 is just Matrix 2 transposed.
    mat3_rwa = np.transpose(mat2_rwa)

    M_rwa = np.append(np.append(mat1, mat2_rwa, axis=1), np.append(mat3_rwa, mat4, axis=1), axis=0)

    # Find eigenvalues and put into array to plot later
    e_vals = np.sort(np.linalg.eigvals(M_rwa))
    e_vals_array_rwa[i] = e_vals
    i += 1

# Create off-diagonal terms. These vary and we plot the graph with them.
e_vals_array_nonrwa = np.zeros((1000, 2*n))

i = 0
for lambda_over_omega in lambda_over_omega_array:
    mat2_nonrwa = np.zeros((n, n))
    position = np.array([1, 0])

    for j in range(n-1):
        mat2_rwa[position[0], position[1]] = np.sqrt(j+1) * lambda_over_omega
        position += 1
    
    mat2_nonrwa = mat2_rwa + np.transpose(mat2_rwa)
    mat3_nonrwa = mat2_nonrwa

    M_nonrwa = np.append(np.append(mat1, mat2_nonrwa, axis=1), np.append(mat3_nonrwa, mat4, axis=1), axis=0)

    e_vals_nonrwa = np.sort(np.linalg.eigvals(M_nonrwa))
    e_vals_array_nonrwa[i] = e_vals_nonrwa
    i += 1

for i in range(n):
    plt.plot(lambda_over_omega_array, e_vals_array_rwa[:, i], '-.', color='green')
    plt.plot(lambda_over_omega_array, e_vals_array_nonrwa[:, i], color='black')
plt.ylim((-1., 3.))
plt.xlabel('$\lambda/\omega_0$')
plt.ylabel('$E/\omega_0$')
plt.xlim(0, 1)
plt.show()
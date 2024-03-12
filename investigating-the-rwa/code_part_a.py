import numpy as np
import matplotlib.pyplot as plt

n = 81
diag = np.zeros(n)

# create axis variables
hbar_omega_array = np.linspace(0, 1, 1000)
lambda_over_omega_array = np.linspace(0, 1, 1000)

# Create diagonal terms
# Start off with 1/2, add 2 terms, then i += 1, add 2 terms, i += 2, add 2, i += 1, add 2, etc.

for i in range(len(diag)):
    term = int(np.ceil(i))
    
    if term % 2 == 0:
        term -= 1
    term /= 2
    
    diag[i] = term

H_diag = np.diag(diag)

# Create off-diagonal terms. These vary and we plot the graph with them.
e_vals_array = np.zeros((1000, n))
e_vals_array_JC = np.zeros((1000, n))

i = 0
for lambda_over_omega in lambda_over_omega_array:
    H_offdiag = np.zeros((n, n))
    x_terms = np.array([0, 1, 2, 3])
    y_terms = np.array([3, 2, 1, 0])
    current_offdiag_row = 0

    while max(x_terms - 2) <= n and max(y_terms - 2) <= n:
        for k in range(len(x_terms)):
            if x_terms[k] < n and y_terms[k] < n:
                H_offdiag[x_terms[k], y_terms[k]] = np.sqrt(current_offdiag_row+1) * lambda_over_omega
        current_offdiag_row += 1
        x_terms += 2
        y_terms += 2

    H = H_diag + H_offdiag

    e_vals = np.sort(np.linalg.eigvals(H))
    e_vals_array[i] = e_vals
    i += 1

i = 0
for lambda_over_omega in lambda_over_omega_array:
    H_offdiag_JC = np.zeros((n, n))
    x_terms = np.array([1, 2])
    y_terms = np.array([2, 1])
    current_offdiag_row = 0

    while max(x_terms - 2) <= n and max(y_terms - 2) <= n:
        for k in range(len(x_terms)):
            if x_terms[k] < n and y_terms[k] < n:
                H_offdiag_JC[x_terms[k], y_terms[k]] = np.sqrt(current_offdiag_row+1) * lambda_over_omega
        current_offdiag_row += 1
        x_terms += 2
        y_terms += 2

    H_JC = H_diag + H_offdiag_JC

    e_vals_JC = np.sort(np.linalg.eigvals(H_JC))
    e_vals_array_JC[i] = e_vals_JC
    i += 1

for i in range(n):
    plt.plot(lambda_over_omega_array, e_vals_array[:, i], color='black')
    plt.plot(lambda_over_omega_array, e_vals_array_JC[:, i], '-.', color='green')
plt.ylim((-1., 3.))
plt.xlabel('$\lambda/\omega_a$')
plt.ylabel('$E/\omega_a$')
plt.show()
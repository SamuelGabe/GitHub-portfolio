import numpy as np
from scipy import constants
from scipy.linalg import expm
import matplotlib.pyplot as plt

n = 3
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
lambda_over_omega = 0.8

# Create off diagonal matrix
H_offdiag = np.zeros((n, n))

# Initial x and y positions of off diagonal elements. These all increase by 2 each iteration.
x_terms = np.array([0, 1, 2, 3])
y_terms = np.array([3, 2, 1, 0])

# This just represents which iteration we're on. Each section of off diagonal matrix
# has a factor of sqrt(current_offdiag_row) and that's what this term does.
current_offdiag_row = 0

#Iterates over each row until it gets to the edge of the matrix
while max(x_terms - 2) <= n and max(y_terms - 2) <= n:
    # Populate this row
    for k in range(len(x_terms)):
        if x_terms[k] < n and y_terms[k] < n:
            H_offdiag[x_terms[k], y_terms[k]] = np.sqrt(current_offdiag_row+1) * lambda_over_omega
    current_offdiag_row += 1
    x_terms += 2
    y_terms += 2

# Do the same thing but making the rotating wave approximation. Here there are 2 terms
# per row instead of 4.
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

# Form the full Hamiltonians by adding diagonal + off diagonal. Diagonal is the same
# whether or not you make RWA.
H = H_diag + H_offdiag
H_JC = H_diag + H_offdiag_JC

# Time axis
t_arr = np.linspace(0,5,1000)

# Now solve time evolution operator for different values of t.
for i in range(len(t_arr)):
    # Time evolution operator
    time_operator = expm(1j * H * t_arr[i])
    time_operator_JC = expm(1j * H_JC * t_arr[i])

    # Calculate eigenvalues and add to array. We will plot array later
    e_vals_array[i] = np.sort(np.linalg.eigvals(time_operator))
    e_vals_array_JC[i] = np.sort(np.linalg.eigvals(time_operator_JC))

# Plot eigenvalues.
for i in range(n):
    #plt.plot(t_arr, e_vals_array[:, i], color='black')
    plt.plot(t_arr, e_vals_array_JC[:, i], '-.', color='green')
plt.title('Evolution of time operator when constants are set to 1, for a 8x8 matrix')
plt.xlabel('$t$')
plt.ylabel('$E\hbar/\omega_0$')
plt.legend()
plt.show()
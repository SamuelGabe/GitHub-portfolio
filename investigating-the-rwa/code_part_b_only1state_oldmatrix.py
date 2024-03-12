import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

#region Define variables and populate matrixes
# Quadrant matrix size.
n = 16
lambda_over_omega = 0.8
hbar = 1
t_max = 70

initial_state_number = 1

# Initial vector for excited state
v_0 = np.zeros(n)
v_0[2 * initial_state_number + 1] = 1.

#region ---- MAKE MATRIX HERE ------
# Create diagonal terms
# Start off with 1/2, add 2 terms, then i += 1, add 2 terms, i += 2, add 2, i += 1, add 2, etc.

diag = np.zeros(n)
for i in range(len(diag)):
    term = int(np.ceil(i))
    
    if term % 2 == 0:
        term -= 1
    term /= 2
    
    diag[i] = term

H_diag = np.diag(diag)

# Create off-diagonal terms for JC matrix
H_offdiag_JC = np.zeros((n, n))
x_terms_JC = np.array([1, 2])
y_terms_JC = np.array([2, 1])
current_offdiag_row = 0

# Create off diagonal rows
while max(x_terms_JC - 2) <= n and max(y_terms_JC - 2) <= n:
    for k in range(len(x_terms_JC)):
        if x_terms_JC[k] < n and y_terms_JC[k] < n:
            H_offdiag_JC[x_terms_JC[k], y_terms_JC[k]] = np.sqrt(current_offdiag_row+1) * lambda_over_omega
    current_offdiag_row += 1
    x_terms_JC += 2
    y_terms_JC += 2

# Combine to make the whole matrix
H_JC = H_diag + H_offdiag_JC

# Now do the same for the non-JC hamiltonian
H_offdiag = np.zeros((n, n))
x_terms = np.array([3, 2, 1, 0])
y_terms = np.array([0, 1, 2, 3])
current_offdiag_row = 0

# Create off diagonal rows
while max(x_terms - 2) <= n and max(y_terms - 2) <= n:
    for k in range(len(x_terms)):
        if x_terms[k] < n and y_terms[k] < n:
            H_offdiag[x_terms[k], y_terms[k]] = np.sqrt(current_offdiag_row+1) * lambda_over_omega
    current_offdiag_row += 1
    x_terms += 2
    y_terms += 2

# Combine
H = H_diag + H_offdiag

#endregion

# Now it's time to bring in the time
t_arr = np.linspace(0, t_max, 1000)

#endregion

#region Rotating Wave Approximation Calculations

# First do RWA matrix. Probabilties of e -> e and e -> g
p_ee_rwa = np.zeros(len(t_arr))
p_eg_rwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevectors_t_rwa = np.zeros((len(t_arr), n))

for i in range(len(t_arr)):
    H_t_JC = expm(1j * H_JC * t_arr[i] / hbar)

    # Vector of psi(t) when psi(0) is the excited state, with RWA
    v_t_rwa = np.matmul(H_t_JC, v_0)

    probabilities = np.conjugate(v_t_rwa) * v_t_rwa
    p_ee_rwa[i] = np.sum(probabilities[1::2])
    p_eg_rwa[i] = np.sum(probabilities[::2])
    wavevectors_t_rwa[i] = probabilities
#endregion

#region Not with rotating wave approximation

# Now time for non-RWA matrix. Probabilties of e -> e and e -> g
p_ee_nonrwa = np.zeros(len(t_arr))
p_eg_nonrwa = np.zeros(len(t_arr))

# I'm also going to store all the arrays of probabilities. Wave vectors have size 2n.
wavevectors_t = np.zeros((len(t_arr), n))

# Do the calculations for non rwa
for i in range(len(t_arr)):
    H_t = expm(1j * H * t_arr[i] / hbar)
    v_t = np.matmul(H_t, v_0)

    probabilities = np.conjugate(v_t) * v_t
    p_ee_nonrwa[i] = np.sum(probabilities[1::2])
    p_eg_nonrwa[i] = np.sum(probabilities[::2])
    wavevectors_t[i] = probabilities


#endregion


# Plot the graph of probabilities of being found in excited state for RWA and non RWA
plt.plot(t_arr, p_ee_nonrwa, color='black', label='$|1\\rangle|e\\rangle\\rightarrow|e\\rangle$ (non-RWA)')
plt.plot(t_arr, p_ee_rwa, 'g-.', label='$|1\\rangle|e\\rangle\\rightarrow|e\\rangle$ (RWA)')
plt.plot(t_arr, p_eg_rwa, 'r-.', label='$|1\\rangle|e\\rangle\\rightarrow|g\\rangle$ (RWA)')
plt.xlabel('$t$')
plt.ylabel("Proability")

plt.legend()
plt.show()

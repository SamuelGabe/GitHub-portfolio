# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import odeint

# x = np.linspace(-20, 20, 100)

# def eqn(y,x): 
#     omega, theta = y
#     return [-0.05*omega - 5*np.sin(theta), omega]

# y0 = [0, 1]

# soln = odeint(eqn, y0, x)
# plt.plot(x, soln[:, 0])
# plt.plot(x, soln[:, 1])
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

x = np.linspace(0, 6, 100)

def eqn(y,x): 
    return y

y0 = 1

soln = odeint(eqn, y0, x)
plt.plot(x, soln)
plt.show()
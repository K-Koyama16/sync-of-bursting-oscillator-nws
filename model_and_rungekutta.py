# Purpose:  In this program, define functions to calculate the simple HR model.
#
# Author:   K.Koyama
#
# Function:
#   - HRmodel
#           Hindmarsh-Rose Model
#   - runge_kutta
#           Runge-Kutta method of the fourth order for calculating differential equations


# Import of python libraries
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit


@jit(nopython = True)
def model(variables, time, parameters): 
    """ Hindmarsh-Rose Model
    Args:
        variables (np.ndarray 1-dim): 3 variables in HRModel
        time (float): time
        parameters (np.ndarray 1-dim): 8 parameters in HRModel
    
    Returns:
        deriv (np.ndarray 1-dim): Model output
    """
    
    I, a, b, c, d, epsilon, s, x_0 = parameters
    x = variables[0]
    y = variables[1]
    z = variables[2] 

    deriv_0 = y - a * x**3 + b * x**2 + I - z
    deriv_1 = c - d * x**2 - y 
    deriv_2 = epsilon * (s * (x - x_0) - z)

    return(np.array([deriv_0, deriv_1, deriv_2]))


@jit(nopython=True)
def runge_kutta(x, t, tau, params):
    """ 4th Runge-Kutta Method 
    Args:
        x (np.ndarray 1-dim): Variables
        t (float): Time
        tau (float): TimeStep
        params (np.ndarray 1-dim): Parameters in model

    Returns:
        x_tmp (np.ndarray): Variables after calculation
    """
    x_tmp = np.zeros(3)

    half_tau = tau / 2 
    t_half = t + half_tau 
    t_full = t + tau
    
    k1 = model(x, t, params) 
    
    x_tmp = x + half_tau * k1
    k2 = model(x_tmp, t_half, params)
    
    x_tmp = x + half_tau * k2
    k3 = model(x_tmp, t_half, params)
    
    x_tmp = x + tau * k3
    k4 = model(x_tmp, t_full, params)
    
    x_tmp = x + tau / 6 * (k1 + k4 + 2 * (k2 + k3))

    return x_tmp


# Test for correct calculations.(Simple Model)
if __name__ =="__main__":

    print("numpy ", np.__version__)
    print("numba ", numba.__version__)
    print("Hindmarsh-Rose Model")

    # Set Parameters
    params = np.array([3.25, 1, 3, 1, 5, 0.005, 4, -1.6])
    
    # Set Step
    step_array = np.array([0.01,300000,60000])
    ttime = 0
    tau, NSKIP, NSTEPS = step_array[0], int(step_array[1]), int(step_array[2])
    NSTEPS += NSKIP

    # Set Initial Variables
    cellout = np.array([-1.31, 7.32, 3.75])
    x_array = np.zeros(NSTEPS-NSKIP)
    
    # Calculate
    for i in range(NSTEPS):
        cellout = runge_kutta(cellout, ttime, tau, params)
        if i>=NSKIP:
            x_array[i-NSKIP] = cellout[0]
        ttime += tau
    print('Calculation completed.')
    
    # Plot
    plt.plot(x_array)
    plt.title("Hindmarsh-Rose")
    plt.ylabel("$x$")
    plt.show()
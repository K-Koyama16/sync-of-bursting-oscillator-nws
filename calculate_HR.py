# Purpose:  In this program, calculate the HR model considering neuronal network with electrical coupling,
#           and obtain the time series of membrane potential and the timing of each spike.
#           For this purpose, define the electrical coupling term. 
#
# Author:   K.Koyama
#
# Function:
#   - cal_HR_spike
#           Calculate HRmodel and obtain the time series of membrane potential and SpikeTiming
#   - k_term
#           Electrical Coupling Term


# Import of python libraries
import numpy as np
from numba import jit
import math
PI = math.pi

from model_and_rungekutta import runge_kutta


@jit(nopython = True)
def cal_HR_spike(cellsout,step_array,force_array,params_n, network = np.empty([0,0])):
    """ Calculate HRmodel and obtain the time series of membrane potential and SpikeTiming
    Args:
        cellsout (np.ndarray 2-dim): Output of neurons
        step_array (np.ndarray 1-dim): tau, NSKIP, NSTEPS
        force_array (np.ndarray 1-dim): Parameters of external and internal forces
        params_n (np.ndarray 2-dim):  8 parameters of N neurons in HRModel
        network (np.ndarray 2-dim): Adjacency matrix showing neuron coupling

    Returns:
        x_array (np.ndarray 2-dim): Time series of x (membrane potential) for each neuron
        st_list_n (list): SpikeTiming of each neurons
    """
    
    #Set Parameter
    N = cellsout.shape[0]
    I = np.copy(params_n)[0][0]
    
    #Set Step
    ttime = 0
    tau,NSKIP,NSTEPS = step_array[0],int(step_array[1]),int(step_array[2])
    
    #Set External and Internal Forces
    A,f,D,noise_seed,k = force_array[0], force_array[1], force_array[2], int(force_array[3]), force_array[4]
    np.random.seed(noise_seed)
    GWN_list = np.random.normal(0,1,NSTEPS)
    
    # About timeseries of x  (membrane potential)
    x_array = np.zeros((N,NSTEPS-NSKIP))
    
    # About SpikeTiming
    th_s = 1.0 # threshold
    st_zero_array_n = np.zeros((N, NSTEPS-NSKIP))
    st_list_n = []
    
    #Calculate
    for i in range(NSTEPS):
        input_noise = D*GWN_list[i]
        for j in range(N):
            params_n[j][0] = I +  A*math.sin(2*PI*f*ttime) + input_noise + k_term(j,cellsout,k,network)
        for j in range(N):
            cellsout[j] = runge_kutta(cellsout[j],ttime,tau,params_n[j])
            if i >= NSKIP:
                x_array[j][i-NSKIP] = cellsout[j][0]
                if x_array[j][i-NSKIP] > th_s and x_array[j][i-NSKIP-1] <= th_s and i>NSKIP:
                    st_zero_array_n[j][i-NSKIP] = i*tau               
        ttime += tau
    
    #Get Spike Timing
    for j in range(N):
        st_list_n.append(st_zero_array_n[j][st_zero_array_n[j].nonzero()])

    # return st_list_n
    return x_array, st_list_n


@jit(nopython = True)
def k_term(i,cellsout,k,network):
    """Electrical Coupling
    Args:
        i (int): cell No.
        cellsout (np.ndarray 2-dim): Output of cells
        k (float): Coupling Strngth
        network (np.array): Adjacency matrix showing neuron coupling
    
    Returns:
        K_i (float): electrical coupling term
    """
    N = cellsout.shape[0]
    K_i = 0
    r_i = 0 #NumOfNeurons_coupled neuron i
    
    if network.shape[0] == 0: #Full-connected NW
        r_i = N-1
        for a in range(N):
            K_i += k*(cellsout[a][0]-cellsout[i][0])
        K_i = 0 if r_i == 0 else K_i/r_i
    else:
        for a in range(N):
            if network[i][a]!=0: # Use Adjacency matrix
                K_i += k*(cellsout[a][0]-cellsout[i][0])
                r_i += 1
        K_i = 0 if r_i == 0 else K_i/r_i

    return K_i


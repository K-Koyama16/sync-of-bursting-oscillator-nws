# Purpose:  In this program, define functions to measure the spike synchronization
#           of each neuron using "precision". For this purpose, determine 
#           the time series of the firing rate and the event time by calculation.
#
# Author:   K.Koyama
#
# Function:
#   - call_firing_rate
#           Calculate firing rate
#   - get_event_and_eachstd
#           Derive the event interval and calculate the standard deviation within the event interval.

# Import of python libraries
import numpy as np
from numba import jit


@jit(nopython = True)
def call_firing_rate(st_array_n,T,stride,step_array):
    """ Calculate firing rate
    Args:
        st_array_n (np.ndarray 2-dim): Spike Timing for each neuron formatted for Precision calculation
        T (int): Initial width of the spike search
        stride (int): Calculation interval of Firing Rate
        step_array (np.ndarray 1-dim): tau, NSKIP, NSTEPS
    
    Retuens:
        FR_array (np.ndarray 1-dim): Time series of firing rate
    """
    
    half_T = T/2
    N = len(st_array_n)
    tau,NSKIP,NSTEPS = step_array[0],int(step_array[1]),int(step_array[2])
    FR_array = np.zeros(int((NSTEPS-NSKIP)/stride))
    index = 0
    
    for i in range(NSKIP,NSTEPS,stride):
        
        flag = np.zeros(N)
        a,stop = 0,0
        while stop == 0:
            a += 1
            half_width = half_T*a
            for j in range(N):
                for k in range(len(st_array_n[j])):
                    if i*tau-half_width <st_array_n[j][k] and st_array_n[j][k]< i*tau + half_width:
                        flag[j] = 1
                    if np.count_nonzero(flag) == N or a == 5:
                        stop = 1 #Stop
        
        FR_array[index] = np.sum(flag)/half_width*2
        index += 1
    
    return FR_array


@jit(nopython = True)
def get_event_and_eachstd(FR_array, st_array_n, stride, step_array):
    """ Derive the event interval and calculate the standard deviation within the event interval.
    Args:
        FR_array  (np.ndarray 1-dim): Time series of firing rate
        st_array_n (np.ndarray 2-dim): Spike Timing for each neuron formatted for Precision calculation
        stride (int): Calculation interval of Firing Rate
        step_array (np.ndarray 1-dim): tau, NSKIP, NSTEPS 
    
    Returns:
        event_list (list): start and end of each event section
        std_list (list): the standard deviation of the spike timings in each event
    """

    tau,NSKIP,NSTEPS = step_array[0],int(step_array[1]),int(step_array[2])
    average_rate = np.mean(FR_array)
    a,b,in_event = 0,0,0 #flag
    N = len(st_array_n)
    event_list,std_list=[],[]
    
    for ts in range(len(FR_array)):
        if FR_array[ts] >= average_rate*2 and in_event == 0:
            a = (NSKIP + ts*stride)*tau
            in_event = 1
        if FR_array[ts] <= average_rate*2 and in_event == 1:
            b = (NSKIP + ts*stride)*tau
            in_event = 2
        if in_event == 2:
            event_list.append([a,b])
            starray_n_in_event =np.zeros(N)
            flag = 0
            in_event = 0
            
            for k in range(N):
                for l in range(len(st_array_n[k])):
                    if st_array_n[k][l] > a and st_array_n[k][l] < b:
                        if starray_n_in_event[k] == 0:
                            starray_n_in_event[k] = st_array_n[k][l]
                            flag += 1
                if flag == N:
                    std_list.append(np.std(starray_n_in_event))
                        
                
    return event_list, std_list
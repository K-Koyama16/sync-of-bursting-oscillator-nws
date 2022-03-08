# Purpose:  In this program, the "precision" calculated for various conditions 
#           is displayed as a heatmap with the external force amplitude (intensity)
#           on the vertica axis and the coupling strength on the horizontal axis.
#
# Author:   K.Koyama
#
# Function:
#   - set_parameters_n
#           Set parameters of N neurons in HRmodel.
#   - speclist2array
#           Move spike timing from list to np.ndarray for Precsion calculation.
#   - simulate_cpg
#           Simulation of noise application in the presence of sinusoida force
#   - simulate_sin
#           Simulation with only sine wave applied


# Import of python libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

import math
import copy
import statistics
PI = math.pi

from calculate_HR import cal_HR_spike
from precision import call_firing_rate, get_event_and_eachstd


def set_parameters_n(N):
    """ Set parameters of N neurons in HRmodel.
    Args:
        N (int): Number of Neurons
    
    Returns:
        params_n (np.ndarray 2-dim): 8 parameters of N neurons in HRModel
    """
    params_only_one_cell = np.array([3.25, 1, 3, 1, 5, 0.005, 4, -1.6])
    params_n = np.tile(params_only_one_cell,(N,1))
    
    return params_n


def speclist2array(spec_list_st):
    """ Move spike timing from list to np.ndarray for Precsion calculation.
    Args:
        spec_list_st (list): Spike timing for each neuron
    
    Returns:
        spec_array_st (np.ndarray 2-dim):  Spike timing for each neuron formatted for Precision calculation
    """
    
    N = len(spec_list_st) 
    num_spike_array = np.zeros(N)
    for j in range(N):
        num_spike_array[j] = int(len(spec_list_st[j]))
    
    max_num_spike = int(np.max(num_spike_array))
    spec_array_st = np.zeros((N,max_num_spike))
    for j in range(N):
        
        if int(num_spike_array[j]) == max_num_spike:
            for k in range(max_num_spike):
                spec_array_st[j][k] = spec_list_st[j][k]
        else:
             for k in range(int(num_spike_array[j])):
                spec_array_st[j][k] = spec_list_st[j][k]    
    
    return spec_array_st



def simulate_cpg(A,f,full_connected=1):
    """ Simulation of noise application in the presence of sinusoida force
    Args:
        A (float): Amplitude of sinusidal force
        f (float): Frequency of sinusidal force
        full_conncted (int): default:1  if nw is not full-connected (average degree is 4), set 0 
    
    Returns:
        precision_matrix (np.ndarray 2-dim): Simulation Result (D-k Matrix)
    """

    D_list = [ (d/20)*25 for d in range(21)]
    k_list = [ (k/20)*1  for k in range(21)]

    num_trial = 5
    precision_matrix = np.zeros((num_trial,len(D_list),len(k_list)))
    
    ini_df = pd.read_csv(f'./initial_value/{initial_file}',header= None)
    ini_cellsout = np.array(ini_df)
    
    N = len(ini_cellsout)
    step_array = np.array([0.01,300000,330000])

    T,stride = 2,100 

    for x in range(num_trial):
        
        if full_connected==1:
            pass
        else:
            er = pd.read_csv(f"./network_csv/network_10_4_{x+1}.csv",header = None)
            network = np.array
            network = er.values
    
        for D in tqdm(D_list):
            for k in k_list:
                params_n = set_parameters_n(N)
                cellsout = copy.deepcopy(ini_cellsout)
                if full_connected==1:
                    force_array = np.array([A,f,D,x,k])# A, f, D, noise_seed, k
                    _, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n)
                else:
                    force_array = np.array([A,f,D,0,k])# A, f, D, noise_seed, k
                    _, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n,network)

                st_array_n = speclist2array(st_list_n)

                #Calculate Precision
                FR_array = call_firing_rate(st_array_n, T,stride, step_array)
                __, std_list = get_event_and_eachstd(FR_array, st_array_n, stride, step_array)
                if len(std_list) == 0:
                    precision = 10000
                else:
                    precision = statistics.mean(std_list)

                precision_matrix[x][D_list.index(D)][k_list.index(k)] = precision
    
    return precision_matrix



def simulate_sin(f, full_connected=1):
    """ Simulation with only sine wave applied
    Args:
        f (float): Frequency of sinusidal force
        full_conncted (int): default:1  if nw is not full-connected (average degree is 4), set 0 

    Returns:
        precision_matrix (np.ndarray 2-dim): Simulation Result (D-k Matrix)
    """

    A_list = [ (a/20)*3 for a in range(21)]
    k_list = [ (k/20)*1  for k in range(21)]

    num_trial = 5
    precision_matrix = np.zeros((num_trial,len(A_list),len(k_list)))
    
    ini_df = pd.read_csv(f'./initial_value/{initial_file}',header= None)
    ini_cellsout = np.array(ini_df)
    
    N = len(ini_cellsout)
    step_array = np.array([0.01,300000,330000])
    
    T,stride = 2,100
    
    for x in range(num_trial):

        if full_connected==1:
            pass
        else:
            er = pd.read_csv(f"./network_csv/network_10_4_{x+1}.csv",header = None)
            network = np.array
            network = er.values
        
        for A in tqdm(A_list):
            for k in k_list:
                params_n = set_parameters_n(N)
                cellsout = copy.deepcopy(ini_cellsout)
                if full_connected==1:
                    force_array = np.array([A,f,0,x,k])# A, f, D, noise_seed, k
                    _, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n)
                else:
                    force_array = np.array([A,f,0,x,k])# A, f, D, noise_seed, k
                    _, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n,network)

                st_array_n = speclist2array(st_list_n)

                #Calculate Precision
                FR_array = call_firing_rate(st_array_n, T,stride, step_array)
                __, std_list = get_event_and_eachstd(FR_array, st_array_n, stride, step_array)
                if len(std_list) == 0:
                    precision = 10000
                else:
                    precision = statistics.mean(std_list)

                precision_matrix[x][A_list.index(A)][k_list.index(k)] = precision
    
    return precision_matrix



if __name__ == "__main__":

    # cpg (change noise strength)
    initial_file = 'initial_value_n10.csv'
    A,f = 0, 0.01 
    pm = simulate_cpg(A,f)
    save_name = 'precision_A0_n10_full'
    np.save(f'./result/{save_name}.npy',pm)

    # # sin
    # initial_file = 'initial_value_n10.csv'
    # f = 0.01 
    # pm = simulate_sin(f)    
    # save_name = 'precision_D0_n10_full'
    # np.save(f'./result/{save_name}.npy',pm)

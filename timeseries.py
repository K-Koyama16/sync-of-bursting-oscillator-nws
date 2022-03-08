# Purpose:  In this program, plot membrane potential timeseries, raster plots,
#           and firing rate time trends. Also, describe the function to create
#           a figure for use in the artice.
# 
# Author:   K.Koyama
#
# Function:
#   - plot_timeseries
#           plot membrane potential timeseries, raster plots, and firing rate time trends.
#   - graph_paper
#           create a figure for the article.


# Import of python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import copy
import itertools

from simulation import set_parameters_n, speclist2array
from calculate_HR import cal_HR_spike
from precision import call_firing_rate, get_event_and_eachstd


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.size"] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def plot_timeseries(cellsout, step_array, force_array, nw_name="full"):
    """ plot membrane potential timeseries, raster plots, and firing rate time trends.
    Args:
        cellsout (np.ndarray 2-dim): Output of neurons
        step_array (np.ndarray 1-dim): tau, NSKIP, NSTEPS
        force_array (np.ndarray 1-dim): Parameters of external and internal forces
        nw_name (str): the name of the file to read (if it is not a full-connected NW)
    
    Returns:
        None
    """
    
    N = len(cellsout)
    params_n = set_parameters_n(N)
    
    # Network
    if nw_name == "full":
        x_array_n, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n)
    else:
        er = pd.read_csv(f"./network_csv/{nw_name}.csv", header=None)
        network = np.array
        network = er.values
        x_array_n, st_list_n = cal_HR_spike(cellsout,step_array,force_array,params_n,network)

    st_all_list = list(itertools.chain.from_iterable(st_list_n))

    # Calculate Precision
    T, stride = 2, 100  # Adjust as needed
    st_array_n = speclist2array(st_list_n)
    FR_array = call_firing_rate(st_array_n,T, stride, step_array)
    event_list, std_list = get_event_and_eachstd(FR_array, st_array_n, stride, step_array)
    if len(std_list) == 0:
        precision = 10000
    else:
        precision = statistics.mean(std_list)
    if precision == 10000:
        print("Cannot calculate Precision.")
    else:
        print("Precision=",precision)

    #-----Plot-----
    # I. Timeseries of Membrane Potential
    plt.figure(figsize=(10,2))
    t = np.arange(step_array[0]*step_array[1],step_array[0]*step_array[2],step_array[0])
    for i in range(N):
        plt.plot(t, x_array_n[i],linewidth = 1.5)
    plt.xlabel('Time (ms)',fontsize=18)
    plt.ylabel('$x$',fontsize=18)
    plt.ylim([-2,2])
    plt.title('$A=$'+str(force_array[0])+'$, D=$'+str(force_array[2])+'$, k=$'+str(force_array[4]),fontsize = 18)
    plt.tight_layout()
    plt.show()


    # Ⅱ.Raster Plot
    plt.figure(figsize=(10,2),facecolor="white")
    neuron_raster = []
    for i in range(N):
        for k in range(len(st_list_n[i])):
            neuron_raster.append(i+1)
    plt.scatter(st_all_list,neuron_raster, c= "black",marker ="|")
    plt.xlabel("Time(ms)")
    plt.ylabel("Neuron")
    plt.yticks([N],[N])
    if 1:
        event_sf = list(itertools.chain.from_iterable(event_list))
        for e in event_sf:
            plt.vlines(x=e, ymin=-0.5,ymax=N+0.5, color="red",linestyle="dashed",linewidth=1.5)
    plt.tight_layout()
    plt.show()


    # I&Ⅱ. Timeseries of Membrane Potential & Raster Plot
    t = np.arange(step_array[0]*step_array[1],step_array[0]*step_array[2],step_array[0])
    fig = plt.figure(figsize=(10,4),facecolor="w")
    fig.suptitle(f"$A={force_array[0]}, D={force_array[2]}, k={force_array[4]}$",fontsize=18)
    # Timeseries of Membrane Potential
    ax1 = plt.subplot(2,1,1)
    for i in range(N):
        ax1.plot(t, x_array_n[i], linewidth =1.5)
    ax1.set_ylabel("$x$")
    ax1.set_ylim(-2,2)
    ax1.yaxis.set_label_coords(-0.05,0.5)
    # Raster Plot
    ax2 = plt.subplot(2,1,2,sharex=ax1)
    neuron_raster = []
    for i in range(N):
        for k in range(len(st_list_n[i])):
            neuron_raster.append(i+1)
    ax2.scatter(st_all_list,neuron_raster, c= "black",marker ="|")
    ax2.set_ylabel("Neuron")
    ax2.set_xlabel("Time(s)")
    ax2.set_yticks([N])   
    ax2.yaxis.set_label_coords(-0.05,0.5)
    plt.setp(ax1.get_xticklabels(),visible=False)
    plt.tight_layout()
    plt.show()

    # Ⅲ. Firing Rate
    plt.figure(figsize=(10,2),facecolor="white")
    t = np.arange(step_array[0]*step_array[1],step_array[0]*step_array[2],step_array[0]*stride)
    plt.plot(t, FR_array,linewidth = 1.5)
    plt.hlines(y = np.mean(FR_array)*2,xmin =t[0], xmax = t[len(t)-1]) #average Rate*2
    plt.xlabel('Time (ms)',fontsize=18)
    plt.ylabel('$Firing Rate$',fontsize=18)
    plt.tight_layout()
    plt.show()





def graph_paper(cellsout, diff=1):
    """ create a diagram for the article.
    Args:
        cellsout (np.ndarray 2-dim): Output of neurons
        diff (int): 1 if display the membrane potential difference between two cells
    
    Returns:
        None
    """
    
    print("Start calculation.")
    N = len(cellsout)
    
    #(a)
    cellsout_copy = copy.deepcopy(cellsout)
    params_n = set_parameters_n(N)
    step_array = np.array([0.01,300000,330000])
    force_array = np.array([1.8,0.01,0,1,0.2])#A,f,D,noise_seed,k
    xa_array_n, _ = cal_HR_spike(cellsout_copy,step_array,force_array,params_n)
    #(b)
    cellsout_copy = copy.deepcopy(cellsout)
    params_n = set_parameters_n(N)
    step_array = np.array([0.01,300000,330000])
    force_array = np.array([1.8,0.01,1.25,1,0.2])#A,f,D,noise_seed,k
    xb_array_n, _ = cal_HR_spike(cellsout_copy,step_array,force_array,params_n)
    #(c)
    cellsout_copy = copy.deepcopy(cellsout)
    params_n = set_parameters_n(N)
    step_array = np.array([0.01,300000,330000])
    force_array = np.array([1.8,0.01,10,1,0.2])#A,f,D,noise_seed,k
    xc_array_n, _ = cal_HR_spike(cellsout_copy,step_array,force_array,params_n)
    #(d)
    cellsout_copy = copy.deepcopy(cellsout)
    params_n = set_parameters_n(N)
    step_array = np.array([0.01,300000,330000])
    force_array = np.array([1.8,0.01,20,1,0.2])#A,f,D,noise_seed,k
    xd_array_n, _ = cal_HR_spike(cellsout_copy,step_array,force_array,params_n)
    
    print("Start graphing.")
    #--------Plot-----------------------------------------
    if diff == 1:

        ii = 49
        dif_a = abs(xa_array_n[ii]-xa_array_n[0])
        dif_b = abs(xb_array_n[ii]-xb_array_n[0])
        dif_c = abs(xc_array_n[ii]-xc_array_n[0])
        dif_d = abs(xd_array_n[ii]-xd_array_n[0])

        dif_color = 'orangered'
        plt.figure(figsize=(15,12),facecolor="white")
        t = np.arange(step_array[0]*step_array[1],step_array[0]*step_array[2],step_array[0])

        #(a)
        plt.subplot2grid((23,1),(0,0),rowspan = 3)
        for i in range(N):
            plt.plot(t, xa_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(a)',loc = 'left',fontsize = 20)
        plt.ylabel('$x$',fontsize=18)
        plt.tick_params(labelbottom = False)

        plt.subplot2grid((23,1),(3,0),rowspan = 2)
        plt.plot(t, dif_a, color = dif_color)
        plt.ylim([-0.5,2.5])
        plt.yticks([0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.tick_params(labelbottom = False)
        plt.ylabel('|$x_{1} - x_{2}$|',fontsize=18)

        #(b)
        plt.subplot2grid((23,1),(6,0),rowspan = 3)
        for i in range(N):
            plt.plot(t, xb_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(b)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        plt.ylabel('$x$',fontsize=18)

        plt.subplot2grid((23,1),(9,0),rowspan = 2)
        plt.plot(t, dif_b, color = dif_color)
        plt.ylim([-0.5,2.5])
        plt.yticks([0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.tick_params(labelbottom = False)
        plt.ylabel('|$x_{1} - x_{2}$|',fontsize=18)

        #(c)
        plt.subplot2grid((23,1),(12,0),rowspan = 3)
        for i in range(N):
            plt.plot(t, xc_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(c)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        plt.ylabel('$x$',fontsize=18)

        plt.subplot2grid((23,1),(15,0),rowspan = 2)
        plt.plot(t, dif_c, color = dif_color)
        plt.ylim([-0.5,2.5])
        plt.yticks([0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.tick_params(labelbottom = False)
        plt.ylabel('|$x_{1} - x_{2}$|',fontsize=18)


        #(d)
        plt.subplot2grid((23,1),(18,0),rowspan = 3)
        for i in range(N):
            plt.plot(t, xd_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(d)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        plt.ylabel('$x$',fontsize=18)

        plt.subplot2grid((23,1),(21,0),rowspan = 2)
        plt.plot(t, dif_d, color = dif_color)
        plt.ylim([-0.5,2.5])
        plt.yticks([0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])


        plt.xticks([3000,3100,3200,3300],['3000','3100','3200','3300'])
        plt.xlabel('Time (ms)',fontsize=20)
        plt.ylabel('|$x_{1} - x_{2}$|',fontsize=18)


    else:
        plt.figure(figsize=(15,12),facecolor="white")
        t = np.arange(step_array[0]*step_array[1],step_array[0]*step_array[2],step_array[0])
        plt.subplot(4,1,1)
        for i in range(N):
            plt.plot(t, xa_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(a)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        
        plt.subplot(4,1,2)
        for i in range(N):
            plt.plot(t, xb_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(b)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        
        plt.subplot(4,1,3)
        for i in range(N):
            plt.plot(t, xc_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(c)',loc = 'left',fontsize = 20)
        plt.tick_params(labelbottom = False)
        plt.ylabel('                                  $x$',fontsize=25)
        
        plt.subplot(4,1,4)
        for i in range(N):
            plt.plot(t, xd_array_n[i],linewidth = 1.5)
        plt.ylim([-2.5,2.8])
        plt.yticks([-2,0,2])
        plt.xlim([t[0]-10, t[len(t)-1]+10])
        plt.title('(d)',loc = 'left',fontsize = 20)
        plt.xticks([3000,3100,3200,3300],['3000','3100','3200','3300'])
        plt.xlabel('Time (ms)',fontsize=25)

    
    plt.show()

if __name__ == "__main__":

    # Full-conneted Network 
    if 0:
        df = pd.read_csv('./initial_value/initial_value_n10.csv',header=None)
        ini_cellsout_10 = np.array(df)
        step_array = np.array([0.01,300000,330000])
        force_array = np.array([5.0,0.01,0,0,0.0]) #A,f,D,noise_seed,k
        #Plot
        plot_timeseries(ini_cellsout_10, step_array, force_array)
    
    # N is 10, and Average degree is 4.
    if 0:
        df = pd.read_csv('./initial_value/initial_value_n10.csv',header=None)
        ini_cellsout_10 = np.array(df)
        step_array = np.array([0.01,300000,330000])
        force_array = np.array([1.0,0.01,0,0,0.2]) #A,f,D,noise_seed,k
        # Plot
        plot_timeseries(ini_cellsout_10, step_array, force_array, nw_name="network_10_4_1")
    
    # Ariticle
    if 1:
        df = pd.read_csv('./initial_value/initial_value_n50.csv',header=None)
        ini_cellsout_50 = np.array(df)
        # Plot
        graph_paper(ini_cellsout_50, diff=1)

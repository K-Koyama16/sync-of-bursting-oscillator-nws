# Purpose:  In this program, the "precision" calculated for various conditions 
#           is displayed as a heatmap with the external force amplitude (intensity)
#           on the vertica axis and the coupling strength on the horizontal axis.
#
# Author:   K.Koyama
#
# Function:
#   - npy_to_df
#           Convert an npy file into an appropriate DataFrame.
#   - df_to_heatmap_precision
#           Draw heatmap of "precision".

# Import of python libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 20 # main_font_size
plt.rcParams['xtick.labelsize'] = 20 # x-axis_size
plt.rcParams['ytick.labelsize'] = 20 # y_axis_size
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 


def npy_to_df(input_file,cpg_yes): 
    """ convert an npy file into an appropriate DataFrame
    Args:
        input_file (.npy): Result for precision
        cpg_yes (int): if input type is cpg (included noise_only), 1; if is sinusoidalforce only, 0
    
    Returns:
        df_pm (pd.DataFrame): DataFrame for drawing heatmap
    """
    
    D_list = [(d/20)*25 for d in range(21)]
    k_list = [(k/20)*1  for k in range(21)]
    A_list = [(a/20)*3  for a in range(21)]
    
    pm = np.load(f'./result/{input_file}')
    pm_average = np.mean(pm, axis = 0)
    if cpg_yes == 1:
        df_pm = pd.DataFrame(pm_average, index = D_list, columns = k_list)
    else:
        df_pm = pd.DataFrame(pm_average, index = A_list, columns = k_list)
    
    return df_pm



def df_to_heatmap_precision(df_pm, cpg_yes, title_name, save_name):
    """ Function to draw Heatmap.
    Args:
        df_pm (pd.DataFrame): DataFrame for drawing heatmap
        cpg_yes (int): if input type is cpg (included noise_only), 1; if is sinusoidalforce only, 0
        title_name (str): tltle
        save_name (str): savename
    
    Returns:
        None
    """
    plt.figure(figsize=(4,3.5))
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams["font.size"] = 20
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in' 

    ax = sns.heatmap(df_pm,cmap='Blues',vmin=0,vmax =0.03,cbar = False)

    ax.invert_yaxis()
    ax.axvline(0,  color = "black")
    ax.axhline(0,  color = "black")
    ax.axvline(21,  color = "black")
    ax.axhline(21,  color = "black")

    plt.xlabel("$k$")
    plt.xticks([0.5,10.5,20.5], ["0","0.5","1"])
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=0)

    if cpg_yes == 1:
        plt.ylabel("$D$")
        plt.yticks([0.5,4.5,8.5,12.5,16.5,20.5],["0","5","10","15","20","25"])
    else:
        plt.ylabel("$A$")
        plt.yticks([0.5,6.833,13.67,20.5],["0","1","2","3"])
    
    plt.title(str(title_name),fontsize = 20)
    #plt.savefig(f'./result/figure_heatmap/{save_name}.png', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    save_name =""
    df_test = npy_to_df('precision_A0_n10_full.npy',1)
    df_to_heatmap_precision(df_test,1,"(b)",save_name)
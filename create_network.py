# Puroose: In this program, 
#
# Author: K.Koyama
#
# Fuction: 
#     - create_nw
#           
#     - show_graph
#           
#     - cal_average_degree
#           

# Import of python libraries
import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd


def create_nw(N,ave_degree):
    """
    Args:
        N (int): number of node
        ave_degree (int): average dgreee
    
    Returns:
        matrix_network (np.ndarray 2-dim): Adjacency matrix showing neuron coupling
    """
    max_edge = (N*ave_degree)/2
    matrix_network = np.zeros((N,N),  dtype=np.int64)

    while max_edge > 0:
        i = random.randint(1,N-1)
        j = random.randrange(i)
        
        if matrix_network[i][j] == 0 and max_edge > 0:
            matrix_network[i][j] = 1
            matrix_network[j][i] = 1
            max_edge -= 1
  
    return matrix_network


def show_graph(network):
    """
    Args:
        network (np.ndarray 2-dim): Adjacency matrix showing neuron coupling

    Returns:
        None
    """
    N = network.shape[0]
    nodes = np.array([str(i) for i in range(N)])
    edges=[]
    for i in range(1,N):
        for j in range(i):
            if network[i][j] == 1:
                edges.append((str(i),str(j)))
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G,k=0.6)
    nx.draw_networkx(G, pos, with_labels=True)
    plt.axis('off') 
    plt.show()
    

def cal_average_degree(network):
    """
    Args:
        network (np.ndarray 2-dim): Adjacency matrix showing neuron coupling

    Returns:
        ave_degree (float): average degree
    """
    size = network.shape[0]
    count = 0
    
    for i in range(1,size):
        for j in range(i):
            if network[i][j]==1:
                count += 1
    
    ave_degree = 2*count/size
    
    return ave_degree
    

    
if __name__ == '__main__':  
    
    if 0:
        matrix = create_nw(8,3)    
        np.savetxt("test.csv",matrix,delimiter=",", fmt='%d') 
        show_graph(matrix)
    
    if 0:
        er = pd.read_csv("./network_csv/network_10_4_5.csv",header=None) 
        network = np.array
        network = er.values
        show_graph(network)
        print(f"Average Degree: {cal_average_degree(network)}")



    

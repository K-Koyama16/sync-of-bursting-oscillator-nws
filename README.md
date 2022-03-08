# Numerical Analysis of synchronization in bursting oscillator neworks   
<!-- Please cite the article if you use the code.   -->
## Article
> **Multiple transition of synchronization by interaction of external and internal forces in bursting oscillator networks**  
> Keita Koyama, Hiroyasu Ando, Kantaro Fujiwara  
> Nonlinear Theory and Its Applications, IEICE/3(12)/pp.545-553, 2021-07  
> https://doi.org/10.1587/nolta.12.545

> **Effect of external and internal forces on synchronization of bursting oscillator networks**  
> Keita Koyama, Hiroyasu Ando, Kantaro Fujiwara   
> Proceedings of The 2020 International Symposium on Nonlinear Theory and Its Applications (NOLTA2020)/pp.131-134, 2020-11  

## CODE
### python script
- model_and_rungekutta.py
- calculate_HR.py
- precision.py
- simulation.py
  - Simulation of changing noise intensity and coupling strength.
  - Simulation of changing sinusoidal amplitude and coupling strength.
- timeseries.py
  - membrane potential
  - raster plots
  - firing rate
  - Figure in the paper
- heatmap.py
- (create_network.py)
### notebook
- ShowHeatmap.ipynb (using functions in "heatmap.py")


## Other Files to be prepared

N: Number of Neurons   

### Network
For networks that are not fully coupled, the adjacency matrix stored in csv file under the "network_csv" directory is used.  
shape: N×N

### Initial value
Initial values are stored in csv file under the "initial_value" directory.  
shape: N×3

## Simulation Results
Simulation results (output of simulation.py) are stored as ".npy" in the "result_matrix" directory.  
shape: Number of trial × D(or A)-Range × k-Range
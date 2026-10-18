# Speedy-Carlo: CUDA-accelerated and Multithreaded Monte Carlo Simulations
This repository is dedicated to making my pre-existing Monte Carlo simulations for option pricing even faster. 
## Directory Structure
```text
speedy-carlo/
├── README.md
├── LICENSE
├── data/
|  ├── CUDA_data.csv
|  ├── MT_data.csv
|  ├── ST_data.csv
├── exp/
|  ├── CUDA_experiment.cu
|  ├── MT_experiment.cu
|  ├── ST_experiment.cu
├── img/
|  ├── calls_convergence.png
|  ├── log_calls_convergence.png
|  ├── log_puts_convergence.png
|  ├── log_time_evolution.png
|  ├── puts_convergence.png
|  ├── time_evolution.png
├── src/
|  ├── GPU_monte_carlo.cu
|  ├── CPU_MT_monte_carlo.cpp
|  ├── CPU_ST_monte_carlo.cpp
├── vis/
|  ├── convergence_runtime_calls.py
|  ├── convergence_runtime_puts.py
|  ├── log_convergence_runtime_calls.py
|  ├── log_convergence_runtime_puts.py
|  ├── log_time_evolution.py
|  ├── time_evolution.py
```

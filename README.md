# Monitoring memory usage

In one terminal:

    mpirun -n 4 python3 demo_poisson.py

And in another

    python3 monitor-memory.py

The memory use stored in the file `memory_usage.csv` for plotting.

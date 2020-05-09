# HPC PDE solver in Python
This project is part of the bachelor project of Gary Folli. This is an HPC PDE solver for the Fisher's equation in python with mpi4py and numpy. This python version has been adapted from the C version of the solver written originally for the ETH summer school here: https://github.com/eth-cscs/SummerSchool2017

This application is used to teach parallel and distributed computing.

**Basic usage:** `mpirun -np <num_proc> python3 main.py <nx> <ny> <nt> <t>`

## Parameters:

* `nx` = number of gridpoints in x-directions (horizontally)
* `ny` = number of gridpoints in y-direction (vertically)
* `nt` = number of timesteps
* `t` = total time
* `-v` = (optional) turn on verbose output
* `-io` = (optional) enable interactive output (python graphs) of the solution
* `-po` = (optional) enable printed output (png image) of the solution generated in current directory
* `-pm` = (optional) print the final solution (matrix) to a text file (.txt) in current directory
* `-d d` = (optional) d parameters (float)
* `-r r` = (optional) r parameters (float)
* `-ic ic` = (optional) points with initial diffusion values for manually specify the intial condition of the model

## Examples

Here are commands showing the usage of the mini-app. Just copy and paste them to see the result ! (Try also these commands with a higher number of processor)

Examples of commands: 

```mpirun -np 4 python3 main.py 128 128 100 0.01
mpirun -np 4 python3 main.py 128 128 100 0.01 -v
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm -d 1.5
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm -d 1.0 -r 1250.0
mpirun -np 4 python3 main.py 256 256 100 0.01 -v -io -po -pm -d 1.0 -r 1100.0 -ic [50,50,0.6/206,206,0.6]
```

Equivalent commands than the one from the slides with the interactive ouput:

```mpirun -np 4 python3 main.py 128 128 100 0.0025 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.005 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io
```

Please be aware that the optional parameters can be entered in any order you like and any of them that you want !! 
The following examples are perfectly valid command: 
```mpirun -np 4 python3 main.py 128 128 100 0.01 -pm
mpirun -np 4 python3 main.py 128 128 100 0.01 -r 1250.0
mpirun -np 4 python3 main.py 228 228 150 0.005 -v -ic [30,30,0.1/196,196,0.1/115,115,0.2] -io
```

# Jupyter notebooks for the HPC PDE solver

The Jupyter notebooks can be run on the CSCS supercomputer or any Jupyter environment which has the cluster extension enabled and has the Ipyparallel and Mpi4py modules. In order to create such an environment in your local machine, follow these steps:
1. If it's not already the case, you need to install an MPI library on your machine such as OpenMPI (https://www.open-mpi.org/) in order to be able to use the mpiexec or mpirun commands to start MPI processes. Just test them in the terminal to make sure that they exist by running `mpirun` or `mpiexec` and you should get a message like `mpirun could not find anything to do`. If you get another message, it probably means that you do not have an MPI library on your local system. 
2. You need a Jupyter environment with Python. I suggest the Anaconda plateform (https://www.anaconda.com/) which comes with Jupyter lab and a lot of Python packages already installed. 
3. You need IPython, ipyparallel and mpi4py, they will allow you to run multiple processes using MPI in Jupyter Lab. So just perform the following command on your Python virtual environment:</br>
`pip install ipython`</br>
`pip install ipyparallel`</br>
`pip install mpi4py`
4. Now you should be able to use the `ipython` command on the terminal. Just run:</br> `ipython profile create --parallel --profile=mpi` to create a new profile called "mpi" when running the IPython clusters. This profile will be used to get MPI in Jupyter Lab.
5. Now you have to edit the file `~/.IPYTHONDIR/profile_mpi/ipcluster_config.py` by adding the line:</br> `c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'`. This line instructs ipcluster to use the MPI launcher. 
6. You are all set! You can now run the cluster using the command: `ipcluster start -n 4 --profile=mpi`. This command will run a cluster of 4 processes. 
7. Once the cluster is running, before running Jupyter Lab, you have to enable the cluster functionality in it with the command:</br>`ipcluster nbextension enable`.  
8. You can now run Jupyter lab in parallel with the ipcluster and execute the cells. Be just careful to change the client command in the cell initializing Ipyparallel:</br>
`rc = Client(profile='mpi')`


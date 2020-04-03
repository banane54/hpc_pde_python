# HPC PDE solver in Python
This project is part of the bachelor project of Gary Folli. This is an HPC PDE solver for the Fisher's equation in python with mpi4py and numpy. This python version has been adapted from the C version of the solver written originally for the ETH summer school here: https://github.com/eth-cscs/SummerSchool2017

This application is used to teach parallel and distributed computing.

Basic usage: `mpirun -np <num_proc> python3 main.py <nx> <ny> <nt> <t>`

Paramters:
`nx` = number of gridpoints in x-directions (horizontally)
`ny` = number of gridpoints in y-direction (vertically)
`nt` = number of timesteps
`t` = total time
`-v` = (optional) turn on verbose output
`-io` = (optional) enable interactive output (python graphs) of the solution
`-po` = (optional) enable printed output (png image) of the solution generated in current directory
`-pm` = (optional) print the final solution (matrix) to a text file (.txt) in current directory
`-d d` = (optional) d parameters (float)
`-r r` = (optional) r parameters (float)
`-ic ic` = (optional) points with initial diffusion values for manually specify the intial condition of the model

## Example

Here are commands showing the usage of the mini-app. Just copy and paste them to see the result ! (Try also these commands with a higher number of processor)

Examples of commands: 

```mpirun -np 4 python3 main.py 128 128 100 0.01
mpirun -np 4 python3 main.py 128 128 100 0.01 -v
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm -d 1.5
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io -po -pm -d 1.0 -r 1250.0
mpirun -np 4 python3 main.py 256 256 100 0.01 -v -io -po -pm -d 1.0 -r 1100.0 -ic [50,50,1.0/200,200,0.1]```
```

Equivalent commands than the one from the slides with the interactive ouput:

```mpirun -np 4 python3 main.py 128 128 100 0.0025 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.005 -v -io
mpirun -np 4 python3 main.py 128 128 100 0.01 -v -io```
```

Please be aware that the optional parameters can be entered in any order you like and any of them that you want !! 
The following examples are perfectly valid command: 
```mpirun -np 4 python3 main.py 128 128 100 0.01 -pm
mpirun -np 4 python3 main.py 128 128 100 0.01 -r 1250.0
mpirun -np 4 python3 main.py 228 228 150 0.005 -v -ic [30,30,0.1/196,196,0.1/115,115,0.2] -io```
```




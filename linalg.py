import numpy as np
from mpi4py import MPI
import math

import data
import operators

# variable for linalg scope
cg_initialized = False
Ap = None
r = None
p = None
Fx = None
Fxold = None
v = None
xold = None

# initialize the global variable in the first cg_iteration
def cg_init(nx, ny):

    # tell python to use the global variable outside this function 
    # and not create new local variable
    global Ap, r, p, Fx, Fxold, v, xold, cg_initialized

    Ap = np.zeros((ny, nx), dtype=np.float64)
    r = np.zeros((ny, nx), dtype=np.float64)
    p = np.zeros((ny, nx), dtype=np.float64)
    Fx = np.zeros((ny, nx), dtype=np.float64)
    Fxold = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    xold = np.zeros((ny, nx), dtype=np.float64)

    cg_initialized = True

# compute the norm and broadcast it to all the other processors
# so that each processor has the same norm
def norm(b): 
    
    result = np.zeros((1), dtype=np.float64)
    global_result = np.zeros((1), dtype=np.float64)

    ## b**2 fastest way to put b to the power 2
    result[0] += np.sum(b**2)

    MPI.COMM_WORLD.Allreduce([result, MPI.DOUBLE], [global_result, MPI.DOUBLE], op=MPI.SUM)

    return math.sqrt(global_result)

# compute the dot and broadcast it to all the other processors
# so that each processor has the same dot result
def dot(x, y):

    result = np.zeros((1), dtype=np.float64)
    global_result = np.zeros((1), dtype=np.float64)
   
    result[0] = np.dot(x, y)
    
    MPI.COMM_WORLD.Allreduce([result, MPI.DOUBLE], [global_result, MPI.DOUBLE], op=MPI.SUM)

    return global_result

# conjugate gradient solver
# solve the linear system A*x = b for x
# the matrix A is implicit in the objective function for the diffusion equation
# the value in x constitute the "first guess" at the solution
# x(N)
# ON ENTRY contains the initial guess for the solution
# ON EXIT contains the solution
def cg_solver(x, b, maxiters, tol, success): 

    global Ap, r, p, Fx, Fxold, v, xold, cg_initialized
    
    # this is the dimension of the linear system which gonna be solved
    nx = data.domain.nx
    ny = data.domain.ny

    # initialize the variables in the first iteration
    if not cg_initialized: 
        cg_init(nx,ny)
    
    # tolerance
    eps = 1.e-8
    # used for the derivative (Jacobian and Ap)
    eps_inv = 1. / eps

    xold[:] = x

    # compute Ax, the Jacobian
    # by computing the derivative using the difference
    # between two points on the grid having an epsilon difference
    operators.diffusion(x, Fxold, 0, 0)
    v = x * (1.0 + eps)
    operators.diffusion(v, Fx, 0, 0)
    
    # r = b - Ax with Ax = J(x) * deltaX
    # The derivative to get the Jacobian: - eps_inv * (Fx - Fxold) = (f(x+eps) - f(x)) / eps
    r = b - eps_inv * (Fx - Fxold)

    p[:] = r

    rold = dot(r.flatten(), r.flatten())
    rnew = rold

    success = False
    if math.sqrt(rold) < tol:
        success = True
        return None
    
    iteration = 0
    for iteration in range(0, maxiters): # maxiters

        # Ap = A*p
        # eps * p => p is the variable which decides for the size of the move
        # p is big, the gradient does a big step
        # v is then a vector pointing in the solution
        v = 1.0 * xold + eps * p
        # because A is a jacobian, we have to look into the
        # direction of v from xold and then getting 
        # the gradient between v and xold
        # which is obtained with the derivative below
        operators.diffusion(v, Fx, 0, 0)
        Ap = eps_inv * (Fx - Fxold)

        # alpha = rold / p' * Ap
        alpha = rold / dot(p.flatten(), Ap.flatten())

        # x += alpha * p
        x += alpha * p

        # r -= alpha * Ap
        r = r - alpha * Ap

        # rnew = r' * r
        rnew = dot(r.flatten(), r.flatten())

        if math.sqrt(rnew) < tol:
            success = True
            break

        # p = r + beta * p
        # p = r + (rnew/rold) * p
        p = r + (rnew/rold) * p

        # rold = rnew
        rold = rnew

    # statistics
    data.iters_cg += iteration + 1

    if not success:
        print("ERROR: CG failed to converge")

    return success


    
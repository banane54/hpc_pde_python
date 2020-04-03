from mpi4py import MPI
import numpy as np
import multiprocessing as mp
import data

# U is x_new
# S is b
# this function calculate b which is f(x_new)
# domain is the domain of the process executing the function
def diffusion(U, S, t, tt):
    
    # shortcuts
    discretization = data.discretization
    domain = data.domain
    alpha = discretization.alpha
    beta = discretization.beta
    nx = domain.nx
    ny = domain.ny

    # we initialize the strucutre for saving the requests
    # and the statuses
    statuses = [MPI.Status()] * 8
    requests = [MPI.Request()] * 8
    comm_cart = domain.comm_cart
    num_requests = 0

    # !! non-blocking communication !!

    # send the North boundary to the north neigbour 
    if domain.neighbour_north >= 0:
        # set tag to be the sender's rank
        # post receive
        requests[num_requests] = comm_cart.Irecv([data.bndN, MPI.DOUBLE], domain.neighbour_north, domain.neighbour_north)
        num_requests += 1

        # pack north buffer
        for i in range(0, nx):
            data.buffN[0][i] = U[ny-1][i]

        requests[num_requests] = comm_cart.Isend([data.buffN, MPI.DOUBLE], domain.neighbour_north, domain.rank)
        num_requests += 1
    
    # same for South
    if domain.neighbour_south >= 0:
        # set tag to be the sender's rank
        # post receive
        requests[num_requests] = comm_cart.Irecv([data.bndS, MPI.DOUBLE], domain.neighbour_south, domain.neighbour_south)
        num_requests += 1

        # pack south buffer
        for i in range(0, nx):
            data.buffS[0][i] = U[0][i]

        requests[num_requests] = comm_cart.Isend([data.buffS, MPI.DOUBLE], domain.neighbour_south, domain.rank)  
        num_requests += 1

    # same for East
    if domain.neighbour_east >= 0:
        # set tag to be the sender's rank
        # post receive
        requests[num_requests] = comm_cart.Irecv([data.bndE, MPI.DOUBLE], domain.neighbour_east, domain.neighbour_east)
        num_requests += 1

        # pack east buffer
        for j in range(0, ny):
            data.buffE[0][j] = U[j][nx-1]
    
        requests[num_requests] = comm_cart.Isend([data.buffE, MPI.DOUBLE], domain.neighbour_east, domain.rank)  
        num_requests += 1

    # same for West
    if domain.neighbour_west >= 0:
        # set tag to be the sender's rank
        # post receive
        requests[num_requests] = comm_cart.Irecv([data.bndW, MPI.DOUBLE], domain.neighbour_west, domain.neighbour_west)
        num_requests += 1

        # pack west buffer
        for j in range(0, ny):
            data.buffW[0][j] = U[j][0]

        requests[num_requests] = comm_cart.Isend([data.buffW, MPI.DOUBLE], domain.neighbour_west, domain.rank) 
        num_requests += 1

    srow = domain.ny - 1
    scol = domain.nx - 1

    # the non-blocking communication allows the communications to take places
    # and while waiting we can calculate the interior grid points for each domain
    # srow and scol NOT INCLUDED in the slice operator (work like range())
    # S is y^(l+1)
    # U is x^(l)
    # data.x_old is x^(l-1)
    S[1:srow, 1:scol] = ( -(4.0 + alpha) * U[1:srow, 1:scol] 
                        + U[1-1:srow-1, 1:scol] + U[1+1:srow+1, 1:scol] 
                        + U[1:srow, 1-1:scol-1] + U[1:srow, 1+1:scol+1]
                        + beta * U[1:srow, 1:scol] * (1.0 - U[1:srow, 1:scol]) 
                        + alpha * data.x_old[1:srow, 1:scol] )

    # wait for all communication to succeed before calculating the boundaries of each subdomain
    MPI.Request.Waitall(requests, statuses)

    # east boundary
    srow = domain.ny - 1
    scol = domain.nx - 1

    S[1:srow, scol] = ( -(4.0 + alpha) * U[1:srow, scol]
                        + U[1-1:srow-1, scol] + U[1+1:srow+1, scol] 
                        + U[1:srow, scol-1] + data.bndE[0, 1:srow]
                        + beta * U[1:srow, scol] * (1.0 - U[1:srow, scol])
                        + alpha * data.x_old[1:srow, scol]  )

    srow = domain.ny - 1
    scol = 0
    
    # west boundary
    S[1:srow, scol] = ( -(4.0 + alpha) * U[1:srow, scol]
                        + U[1:srow, scol+1] + U[1-1:srow-1, scol] + U[1+1:srow+1, scol]
                        + alpha * data.x_old[1:srow, scol] + data.bndW[0, 1:srow]
                        + beta * U[1:srow, scol] * (1.0 - U[1:srow, scol]) )
    
    # North boundary
    srow = domain.ny - 1
    
    # NW corner
    scol = 0
    S[srow, scol] = ( -(4.0 + alpha) * U[srow, scol]
                        + U[srow-1, scol] + data.bndN[0][scol] 
                        + data.bndW[0, srow] + U[srow, scol+1]
                        + beta * U[srow, scol] * (1.0 - U[srow, scol])
                        + alpha * data.x_old[srow, scol])
    
    # north boundary
    scol = domain.nx - 1
    S[srow, 1:scol] = ( -(4.0 + alpha) * U[srow, 1:scol]
                        + U[srow, 1-1:scol-1] + U[srow, 1+1:scol+1] + U[srow-1, 1:scol]
                        + alpha * data.x_old[srow, 1:scol] + data.bndN[0, 1:scol]
                        + beta * U[srow, 1:scol] * (1.0 - U[srow, 1:scol]) )

    # NE corner
    scol = domain.nx - 1
    S[srow, scol] = ( -(4.0 + alpha) * U[srow, scol]
                        + U[srow, scol-1] + U[srow-1, scol]
                        + alpha * data.x_old[srow, scol] + data.bndE[0, srow] + data.bndN[0, scol]
                        + beta * U[srow, scol] * (1.0 - U[srow, scol]) )

    # South boundary
    srow = 0
    
    # SW corner
    scol = 0
    S[srow, scol] = ( -(4.0 + alpha) * U[srow, scol]
                        + U[srow, scol+1] + U[srow+1, scol]
                        + alpha * data.x_old[srow, scol] + data.bndW[0, srow] + data.bndS[0, scol]
                        + beta * U[srow, scol] * (1.0 - U[srow, scol]) )
    
    # south boundary
    scol = domain.nx - 1
    S[srow, 1:scol] = ( -(4.0 + alpha) * U[srow, 1:scol]
                        + U[srow, 1-1:scol-1] + U[srow, 1+1:scol+1] + U[srow+1, 1:scol]
                        + alpha * data.x_old[srow, 1:scol] + data.bndS[0, 1:scol]
                        + beta * U[srow, 1:scol] * (1.0 - U[srow, 1:scol]) )

    # SE corner
    scol = domain.nx - 1
    S[srow, scol] = ( -(4.0 + alpha) * U[srow, scol]
                        + U[srow, scol-1] + U[srow+1, scol]
                        + alpha * data.x_old[srow, scol] + data.bndE[0, srow] + data.bndS[0, scol]
                        + beta * U[srow, scol] * (1.0 - U[srow, scol]) )

    # Statistics
    # Update the flop counts
    # 8 flops per point
    data.flops_count += (
            + 12 * (nx - 2) * (ny - 2)  # interior points
            + 11 * (nx - 2  +  ny - 2)  # all boundaries points
            + 11 * 4 )                  # corner points

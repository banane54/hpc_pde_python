from mpi4py import MPI
import time
import math

# It is very important to understand that each processor (subdomain)
# has it's own version of all these variables below
# -----------------------------------------------
# statistics
flops_count = 0
iters_cg = 0 
iters_newton = 0
verbose_output = False
interactive_output = False
printed_output = False
printed_matrix = False
custom_init = False
# -----------------------------------------------
# fields that holds the solution 
x_new = 0 # 2d
x_old = 0 # 2d
# -----------------------------------------------
# fields that hold the boundary points 
bndN = 0 # 1d
bndE = 0 # 1d
bndS = 0 # 1d
bndW = 0 # 1d
# -----------------------------------------------
# buffers used during boundary halo communication
buffN = 0 # 1d
buffE = 0 # 1d
buffS = 0 # 1d
buffW = 0 # 1d
# -----------------------------------------------
# data structures (python object)
# discretization keeps the informations of the full grids
# domain keeps the per domain informations (dimensions, coordinates in the cartesian plan...)
domain = None 
discretization = None

class Discretization: 
    
    def __init__(self):
        self.nx = 0         # x dimension (Number of HORIZONTAL grid points) (int)
        self.ny = 0         # y dimension (Number of VERTICAL grids points) (int)
        self.nt = 0         # number of time steps (int)
        self.dx = 0         # distance between grid points (double)
        self.dt = 0         # time step size (double)
        self.r = 1000.0     # R model parameter (double)
        self.d = 1.0        # D model parameter (double)
        self.alpha = 0      # dx^2/(D*dt) (double)
        self.beta = 0       # R * delta(x^2)/D (double)

        self.points = 0     # manually entering initial conditions

class SubDomain: 

    # generate dimensions of the cartesian plan given a number of processors
    def create_dim(self, size):
        dividers = []
        for i in range(size - 1 , 0, -1):
            if (size%i == 0 and i != 1):
                dividers.append(i)
        if (dividers == []):
            return [size, 1]
        else: 
            divider = dividers[len(dividers) // 2]
            return [size // divider, divider]
    
    # constructor and init function
    def __init__(self, rank, size, discretization, communicator):

        # dimension of the cartesian plan
        dims = self.create_dim(size)
        
        # dimension on axis
        self.ndomy = dims[0]
        self.ndomx = dims[1]

        # generate cartesian plan given the dimensions
        comm_cart = communicator.Create_cart(dims, [False, False], False)
        
        self.comm_cart = comm_cart

        # get coordinates of the current processor on the newly generated cartesian plan
        coords = self.comm_cart.Get_coords(rank)
        
        # coordinates of each subdomain
        self.domy = coords[0] # (int)
        self.domx = coords[1] # (int)
    
        # Column direction is 0 (y-direction)
        south_north = self.comm_cart.Shift(0, 1) # tuple
        # Row direction is 1 (x-direction)
        west_east = self.comm_cart.Shift(1, 1) # tuple
    
        # the rank of neighbouring domains
        self.neighbour_south = south_north[0] # (int)
        self.neighbour_north = south_north[1] # (int)
        self.neighbour_west = west_east[0] # (int)
        self.neighbour_east = west_east[1] # (int)

        # x and y dimension in grid points of the sub-domain
        self.nx = discretization.nx // (self.ndomx) # (int)
        self.ny = discretization.ny // (self.ndomy) # (int)

        # the starting coordinates in the grid of the current subdomain
        self.startx = self.domx * self.nx # (int)
        self.starty = self.domy * self.ny # (int)

        # adjust for grid dimensions that, potentially, 
        # do not divided evenly between the sub-domains
        if self.domx == (self.ndomx - 1):
            self.nx = discretization.nx - self.startx
        if self.domy == (self.ndomy - 1):
            self.ny = discretization.ny - self.starty
        
        # the ending coordinates in the grid of the current subdomain
        self.endx = self.startx + self.nx - 1 # (int)
        self.endy = self.starty + self.ny - 1 # (int)

        # total number of grid points of the current subdomain
        self.n_total = self.nx * self.ny # (int)

        # mpi values for the subdomain
        self.rank = rank
        self.size = size

    # for printing the caracteristics of the cartesian division of the processors
    def print(self):
        for i in range(0, self.size):
            if i == self.rank:  
                print("Rank " + str(self.rank) + "/" + str(self.size))
                print("At index (" + str(self.domy) + "," + str(self.domx) + ")")
                print("Neigh N:S " + str(self.neighbour_north) + ":" + str(self.neighbour_south))
                print("Neigh E:W " + str(self.neighbour_east) + ":" + str(self.neighbour_west))
                print("Startx:endx  " + str(self.startx) + ":" + str(self.endx))
                print("Starty:endy  " + str(self.starty) + ":" + str(self.endy))
                print("Local dims " + str(self.nx) + " x " + str(self.ny))
                print("")
            
            MPI.COMM_WORLD.Barrier()
            
            # for the welcome output to not be polluated
            time.sleep(0.1)

        return None
    

      

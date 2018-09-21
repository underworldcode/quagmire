import numpy as np
from quagmire.mesh import TriMesh
from quagmire import tools as meshtools
from mpi4py import MPI
comm = MPI.COMM_WORLD

class Diffusion(TriMesh):
    """
    Diffusion class extends the TriMesh structure to facilitate diffusion
    problems with explicit timestepping.

    Mesh points should be uniformly-spaced (by using lloyd mesh improvement
    or poisson disc sampling) to avoid degeneracies in the solution.
    """
    def __init__(self, dm):
        """
        Initialise TriMesh from the DM
        """
        super(Diffusion, self).__init__(dm)
        self.history = []

    def initial_conditions(self, t, u):
        """
        Evaluate initial conditions on vector u
        """
        if t != 0.0:
            raise ValueError("Initial conditions only for t=0.0")

        pts = self.tri.points
        self.lvec.setArray(np.exp(-0.025*(pts[:,0]**2 + pts[:,1]**2)**2) + 0.0001)
        self.dm.localToGlobal(self.lvec, u)

        # append to history
        self.history.append((0, 0.0, u.array.copy()))


    def monitor(self, i, t, u):
        """
        i    = number of timestep
        t    = time at timestep
        u    = state vector
        """
        last_i, last_t, last_u = self.history[-1]
        if i - last_i == 10:
            uu = u.copy()
            self.history.append((i, t, uu.array))
            print("cpu {}  - step = {:3d},  time = {:2.3f}".format(comm.rank, i, t))


    def plotHistory(self):
        """
        Moves global informaton to the root processor to plot timesteps.
        """
        self._gather_root()
        x = self.root_x
        y = self.root_y

        if comm.rank == 0:
            import matplotlib.pyplot as plt
            import matplotlib.tri as mtri

            # re-triangulate
            triang = mtri.Triangulation(x, y)

        for i,t,u in self.history:
            self.gvec.setArray(u)
            self.tozero.scatter(self.gvec, self.zvec)

            if comm.rank == 0:
                print("plotting timestep {:3d}, t = {:2.3f}".format(i, t))
                fig = plt.figure(1)
                ax = fig.add_subplot(111, xlim=[x.min(), x.max()], \
                                          ylim=[y.min(), y.max()])
                im = ax.tripcolor(triang, self.zvec.array, vmin=0, vmax=1)
                fig.colorbar(im)
                plt.savefig('diffusion_{}.png'.format(i))
                plt.cla()
                plt.clf()


    def diffusion_rate(self, t, u, kappa, bval, f):
        self.dm.globalToLocal(u, self.lvec)

        tstep = (self.area/kappa).min()
        u_x, u_y = self.derivative_grad(self.lvec.array)

        flux_x = kappa * u_x
        flux_y = kappa * u_y
        flux_y[~self.bmask] = bval
        flux_x[~self.bmask] = bval

        d2u = self.derivative_div(flux_x, flux_y)

        self.lvec.setArray(d2u)
        self.dm.localToGlobal(self.lvec, f)

        return tstep


# Setup distributed mesh
minX, maxX = -5., 5.
minY, maxY = -5., 5.
dx, dy = 0.1, 0.1

x, y, bmask = meshtools.square_mesh(minX, maxX, minY, maxY, dx, dy, 10000, 100)
x, y = meshtools.lloyd_mesh_improvement(x, y, bmask, iterations=3)
dm = meshtools.create_DMPlex_from_points(x, y, bmask, refinement_steps=0)

ode = Diffusion(dm)
f = ode.gvec.duplicate()
d2f = ode.gvec.duplicate()

time = 0.0
ode.initial_conditions(time, f)

# kappa can be spatially variable or constant
kappa = 1.0

steps = 100
for step in range(0,steps):
    tstep = ode.diffusion_rate(time, f, kappa, 0.0, d2f)
    
    d2f.scale(tstep)
    f.axpy(1., d2f)

    time += tstep

    ode.monitor(step, time, f)

ode.plotHistory()

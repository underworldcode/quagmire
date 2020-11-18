# ---
# jupyter:
#   jupytext:
#     formats: Notebooks/IdealisedExamples//ipynb,Examples/IdealisedExamples//py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Orogenic Landscape Modelling

# We investigate the drainage network dynamics and the steady-state drainage patterns that emerge from erosion of an uplifting mountain.

import quagmire as qg

qg.nd = qg.scaling.non_dimensionalise

u = qg.scaling._scaling.u

scaling_coefficients = qg.scaling._scaling.get_coefficients()

scaling_coefficients["[length]"] = 80 * u.km
scaling_coefficients["[time]"] = 1000 * u.years

# ## Utilities



# # Create Mesh

# +
from quagmire import QuagMesh
from quagmire import tools as meshtools
from quagmire import function as fn
from quagmire import equation_systems as systems
import quagmire
import numpy as np
import matplotlib.pyplot as plt
from time import time

# %matplotlib inline

# +
minX, maxX = 0.0, qg.nd(80. * u.km)
minY, maxY = 0.0, qg.nd(40. * u.km)
dx, dy = qg.nd(500 * u.m), qg.nd(500 * u.m)

x1, y1, simplices = meshtools.square_mesh(minX, maxX, minY, maxY, dx, dy, random_scale=1.0)
DM = meshtools.create_DMPlex(x1, y1, simplices, boundary_vertices=None)
mesh = QuagMesh(DM, verbose=True, tree=True)
# -

print( "\nNumber of points in the triangulation: {}".format(mesh.npoints))
print( "Downhill neighbour paths: {}".format(mesh.downhill_neighbours))

boundary_mask_fn = fn.misc.levelset(mesh.mask)

# ### Initial topography

with mesh.deform_topography():
    mesh.topography.data = 0.

with mesh.deform_topography():
    new_elevation = qg.nd(100.*u.meter) * mesh.mask
    mesh.topography.data = new_elevation.evaluate(mesh)

# ### Rainfall Function

rainfall_fn = mesh.add_variable(name="rainfall")
rainfall_fn.data = qg.nd(1.*u.m / u.year)

# ### Uplift function

uplift_rate_fn = mesh.add_variable(name="uplift")
uplift_rate_fn = qg.nd(1.0 * u.mm / u.year) * mesh.mask

# ### Stream Power Law

# +
# vary these and visualise difference
m = fn.parameter(0.5)
n = fn.parameter(1.0)
K = fn.parameter(qg.nd(5.0e-6 / u.year))

# create stream power function
upstream_precipitation_integral_fn = mesh.upstream_integral_fn(rainfall_fn)
stream_power_fn = K*upstream_precipitation_integral_fn**m * mesh.slope**n * boundary_mask_fn

# evaluate on the mesh
sp = stream_power_fn.evaluate(mesh)
# -

# ### Diffusion and Transport Solvers

# +
import quagmire.equation_systems as systems

## Set up diffusion solver
diffusion_solver = systems.DiffusionEquation(mesh=mesh)
diffusion_solver.neumann_x_mask = fn.misc.levelset(mesh.mask, invert=True)
diffusion_solver.neumann_y_mask = fn.misc.levelset(mesh.mask, invert=True)
diffusion_solver.dirichlet_mask = fn.parameter(0.0)
diffusion_solver.diffusivity = fn.parameter(qg.nd(0.8 * u.m**2 / u.year))
diffusion_solver.verify() # Does nothing but is supposed to check we have everything necessary

# not needed to run
hillslope = diffusion_solver.phi
hillslope.data = mesh.topography.data

## Set up transport solver
transport_solver = systems.ErosionDepositionEquation(mesh=mesh, m=0.5, n=1.0)
transport_solver.rainfall = rainfall_fn
transport_solver.verify()
# -

# ## Timestepping

# +
mesh.verbose = False
save_fields = True

efficiency = fn.parameter(qg.nd(5.0e-6 / u.year))

h5_filename = "fields_{:06d}.h5"
stats = "Step {:04d} | dt {:.5f} | time {:.4f} | min/mean/max height {:.3f}/{:.3f}/{:.3f} | step walltime {:.3f}"
sim_time = 0.0
steps = 20

for i in range(steps):
    
    t = time()
    
    topography0 = mesh.topography.copy()
    
    # get timestep size   
    dt = min(diffusion_solver.diffusion_timestep(), transport_solver.erosion_deposition_timestep())
    
    # build diffusion, erosion + deposition
    diffusion_rate = diffusion_solver.diffusion_rate_fn(mesh.topography).evaluate(mesh)
    erosion_rate, deposition_rate = transport_solver.erosion_deposition_local_equilibrium(efficiency)
    uplift_rate = uplift_rate_fn.evaluate(mesh)
    dhdt = diffusion_rate - erosion_rate + uplift_rate #+ deposition_rate
    
    # do not rebuilt downhill matrix at half timestep
    mesh.topography.unlock()
    mesh.topography.data = mesh.topography.data + 0.5*dt*dhdt
    mesh.topography.lock()
    
    # get timestep size
    dt = min(diffusion_solver.diffusion_timestep(), transport_solver.erosion_deposition_timestep())
    
    # build diffusion, erosion + deposition
    diffusion_rate = diffusion_solver.diffusion_rate_fn(mesh.topography).evaluate(mesh)
    erosion_rate, deposition_rate = transport_solver.erosion_deposition_local_equilibrium(efficiency)
    uplift_rate = uplift_rate_fn.evaluate(mesh)
    dhdt = diffusion_rate - erosion_rate + uplift_rate#+ deposition_rate
    
    # now take full timestep
    with mesh.deform_topography():
        mesh.topography.data = topography0.data + dt*dhdt
        
    if save_fields:
        mesh.save_mesh_to_hdf5(h5_filename.format(i))
        mesh.save_field_to_hdf5(h5_filename.format(i), topo=mesh.topography.data)
        quagmire.tools.generate_xdmf(h5_filename.format(i))
        
    sim_time += dt
    
    if i/steps*100 in list(range(0,100,10)):
        topo_scaled = qg.scaling.dimensionalise(mesh.topography.data, u.meter)
        simulation_time = qg.scaling.dimensionalise(sim_time, u.year)
        print(stats.format(i, dt, simulation_time, topo_scaled.min(), topo_scaled.mean(),
                           topo_scaled.max(), time() - t))

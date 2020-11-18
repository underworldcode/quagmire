# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python2
#     name: python2
# ---

# %% [markdown]
# # REF01 Model

# %%
from quagmire import QuagMesh
from quagmire import tools as meshtools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

# %% [markdown]
# # Scaling

# %%
import unsupported.scaling as sca

# %%
u = sca.UnitRegistry
nd = sca.nonDimensionalize

# Characteristic values of the system
uplift_rate = (0.1 * u.millimeter / u.year).to(u.meter / u.second)
model_length = 100. * u.kilometer
model_width = 100. * u.kilometer

KL_meters = model_length
KT_seconds = KL_meters / uplift_rate
KM_kilograms = 1. * u.kilogram
Kt_degrees = 1. * u.degC
K_substance = 1. * u.mole

sca.scaling["[time]"] = KT_seconds
sca.scaling["[length]"] = KL_meters

# %%
nx =   100
ny =   100
xl = nd(100. * u.kilometer)   
yl = nd(100. * u.kilometer)  
  
#dt = nd(100000.0 * u.year, scaling)    
nstep = 200
nfreq = 200
 
m = 0.5   
n = 1.0
K = nd(1e-5 / u.year)
 
precipitation =    nd(1.0 * u.meter / u.year)    
 
# Initial topography
initial_topography =    nd(5.0 * u.meter)    
 
# Uplift rate
uplift_rate =   nd(0.1 * u.millimeter / u.year)


# %% [markdown]
# Each of the erosion-depositon models require the stream power...

# %%
def compute_stream_power(self, m=1, n=1, critical_slope=None):
    """
    Stream power law (q_s)
    """
    
    if critical_slope != None:
        slope = np.maximum(self.slope, critical_slope)
    else:
        slope = self.slope
    
    rainflux = self.rainfall_pattern
    rainfall = self.area * rainflux
    cumulative_rain = self.cumulative_flow(rainfall, use3path=True)
    cumulative_flow_rate = cumulative_rain # / self.area
    stream_power = cumulative_flow_rate**m * slope**n
    
    return stream_power


# %% [markdown]
# ## 1. Local equilibrium
#
# The assumption of the stream power law is that sediment transport is in a state of local equilibrium in which the transport rate is (less than or) equal to the local carrying capacity. If we neglect suspended-load transport for a moment and assume only bed-load transport then the local deposition is the amount of material that can be eroded from upstream.

# %%
def erosion_deposition_1(mesh, stream_power, efficiency=0.1, critical_slope=1000.0):
    """
    Local equilibrium model
    """
    
    erosion_rate = efficiency*stream_power
    full_capacity_sediment_load = stream_power
                                  # Could / should be a more complicated function than this !
         
    # Total sediment load coming from upstream 
    cumulative_eroded_material = mesh.cumulative_flow(erosion_rate*mesh.area) 
        
    # We might need to iterate this for landscapes which steepen after shallowing

    erosion_rate[np.where(cumulative_eroded_material > full_capacity_sediment_load)] = 0.0
    cumulative_eroded_material = mesh.cumulative_flow(erosion_rate*mesh.area) 

    # Local equilibrium assumes all this material is dumped on the spot   
    deposition_rate = np.maximum(0.0,(cumulative_eroded_material - full_capacity_sediment_load)) / mesh.area 
        
    # However, for stability purposes, it is useful to smooth this within the stream bed
    # (The saltation length model can smooth this over a specific length / time downstream)
    
    erosion_rate[~mesh.bmask] = 0.0
    deposition_rate[~mesh.bmask] = 0.0

    # Fix any negative values in the deposition. 
    
    depo_sum = deposition_rate.sum()
    deposition_rate = np.clip(deposition_rate, 0.0, 1.0e99)
    deposition_rate *= deposition_rate.sum() / (depo_sum + 1e-12)
   
    # Smoothing

    erosion_rate    = mesh.streamwise_smoothing(erosion_rate, 3, centre_weight=0.75)
    deposition_rate = mesh.downhill_smoothing(deposition_rate, 10, centre_weight=0.75)

    # Patch low points, undershoots and smooth flat spots
    
    mesh.height = np.clip(sp.height, 0.0, 1.0e99)
    
    low_points = sp.identify_low_points()
    
    if len(low_points):
        deposition_rate[low_points] = 0.0
        sp.height[low_points] = sp.height[sp.neighbour_cloud[low_points,0:10]].mean(axis=1)
   

    flat_spots = sp.identify_flat_spots()

    if len(flat_spots):
        smoothed_deposition_rate = deposition_rate.copy()
        smoothed_deposition_rate[np.invert(flat_spots)] = 0.0   
        for i in range(0,5):
            smoothed_deposition_rate = mesh.rbf_smoother(smoothed_deposition_rate)     
        deposition_rate  += smoothed_deposition_rate
    
    # Update the slope to account for those fixes (maybe height ?)
    
    hx, hy = mesh.derivative_grad(mesh.height)
    slope = np.hypot(hx, hy)
    slope = np.minimum(mesh.slope, critical_slope)
    
    return erosion_rate, deposition_rate

# %% [markdown]
# ## Time evolution
#
#

# %%
minX, maxX = -xl/2.0, xl / 2.0
minY, maxY = -yl/2.0, yl/2.0
dx, dy = xl / nx, yl / ny

x, y, bmask = meshtools.square_mesh(minX, maxX, minY, maxY, dx, dy, nx * ny, 1000)
DM = meshtools.create_DMPlex_from_points(x, y, bmask)

sp = QuagMesh(DM)
points, simplices, bmask = sp.get_local_mesh()
x = points[:,0]
y = points[:,1]

# Initial Topography
height = nd(0.1 * u.meter) * np.sin((x-minX) * np.pi /(maxX-minX)) # Add some slope
#height += np.random.random(height.size) * nd(1. * u.centimeter) # random noise
height += initial_topography

sp.update_height(height)

sp.verbose=False
rain = np.ones_like(sp.height) * precipitation # uniform precipitation

sp.update_surface_processes(rain, np.zeros_like(rain))
sp.verbose=False

time = nd(0.0 * u.year)
step = 0
steps = 50

viz_time= 0.0
vizzes = 0

kappa = nd(1.0e-3 * u.metre**2 / u.year)   # Diffusion coeficient
critical_slope = 5.0      # Critical value - assume slides etc take over to limit slope
lowest_slope   = 1.0e-3   # The slope where we cut off the erosion / deposition algorithm
base = 0.0
totalSteps = 0

experiment_name = "Ref-v1"

# %%
fig = plt.figure(1)
ax = fig.add_subplot(111, xlim=(minX, maxX), ylim=(minY, maxY))
im = ax.tripcolor(x, y, sp.tri.simplices, sca.Dimensionalize(sp.height, u.metre).magnitude, cmap='terrain')
fig.colorbar(im, ax=ax, label='height')
plt.show()

# %%
import time as systime

walltime = systime.clock()
typical_l = np.sqrt(sp.area)
running_average_uparea = sp.cumulative_flow(sp.area * sp.rainfall_pattern)

for step in range(0,steps):
    
    ###############################
    ## Compute erosion / deposition
    ###############################
    slope = np.minimum(sp.slope, critical_slope)
    stream_power = compute_stream_power(sp, m=m, n=n, critical_slope=critical_slope)

    erosion_rate, deposition_rate = erosion_deposition_1(sp, stream_power, efficiency=K, 
                                                         critical_slope=critical_slope)    
    erosion_deposition_rate = erosion_rate - deposition_rate
    erosion_timestep    = ((slope + lowest_slope) * typical_l / (np.abs(erosion_rate)+0.000001)).min()
    deposition_timestep = ((slope + lowest_slope) * typical_l / (np.abs(deposition_rate)+0.000001)).min()
    
    ################
    ## Diffusion
    ################
    diffDz, diff_timestep =  sp.landscape_diffusion_critical_slope(kappa, critical_slope, True)
        
    ## Mid-point method. Update the height and use this to estimate the new rates of 
    ## Change. Note that we have to assume that the flow pattern doesn't change for this 
    ## to work. This means we can't call the methods which do a full update ! 
    timestep = min(erosion_timestep, deposition_timestep, diff_timestep)
    time = time + timestep
    viz_time = viz_time + timestep

    # Height predictor step (at half time)
    height0 = sp.height.copy()
    sp.height -= 0.5 * timestep * (erosion_deposition_rate - diffDz )
    sp.height += 0.5 * uplift_rate * timestep
    sp.height = np.clip(sp.height, base, 1.0e99)   
    
    # Deal with internal drainages (again !)
    sp.height = sp.handle_low_points(base, 5)  
    gradZx, gradZy = sp.derivative_grad(sp.height)
    sp.slope = np.hypot(gradZx,gradZy)   
    
    # Recalculate based on mid-point values
    erosion_rate, deposition_rate = erosion_deposition_1(sp, stream_power, efficiency=K, 
                                                         critical_slope=critical_slope)    
    
    erosion_deposition_rate = erosion_rate - deposition_rate
    erosion_timestep    = ((slope + lowest_slope) * typical_l / (np.abs(erosion_rate)+0.000001)).min()
    deposition_timestep = ((slope + lowest_slope) * typical_l / (np.abs(deposition_rate)+0.000001)).min()
   
    diffDz, diff_timestep =  sp.landscape_diffusion_critical_slope(kappa, critical_slope, True)
 
    timestep = min(erosion_timestep, deposition_timestep, diff_timestep)
    time = time + timestep
    
    # Now take the full timestep

    height0 -= timestep * (erosion_deposition_rate - diffDz )
    height0 += uplift_rate * timestep
    sp.height = np.clip(height0, base, 1.0e9)  
    sp.height = sp.handle_low_points(base, 5)

    sp.update_height(sp.height)
    # sp.update_surface_processes(rain, np.zeros_like(rain))
    
    running_average_uparea = 0.5 * running_average_uparea + 0.5 * sp.cumulative_flow(sp.area * sp.rainfall_pattern)
 
    if totalSteps%10 == 0:
        print "{:04d} - ".format(totalSteps), \
          " dt - {:.5f} ({:.5f}, {:.5f}, {:.5f})".format(
                            sca.Dimensionalize(timestep, u.year),
                            sca.Dimensionalize(diff_timestep, u.year),
                            sca.Dimensionalize(erosion_timestep, u.year),
                            sca.Dimensionalize(deposition_timestep, u.year)), \
          " time - {:.4f}".format(sca.Dimensionalize(time, u.year)), \
          " Max slope - {:.3f}".format(sp.slope.max()), \
          " Step walltime - {:.3f}".format(systime.clock()-walltime)
            
              
    # Store data
#     if( viz_time > 0.1 or step==0):

#         viz_time = 0.0
#         vizzes = vizzes + 1

#         delta = height-sp.height
#         smoothHeight = sp.local_area_smoothing(sp.height, its=2, centre_weight=0.75)
         
#         #if step == 0: 
#         #    sp.save_mesh_to_hdf5("{}-Mesh".format(experiment_name))
            
#         #sp.save_field_to_file("{}-Data-{:f}".format(experiment_name, totalSteps), 
#         #                      bmask=sp.bmask,
#         #                      height=sp.height, 
#         #                      deltah=delta, 
#         #                      upflow=running_average_uparea, erosion=erosion_deposition_rate)


    ## Loop again 
    totalSteps += 1



# %%
# Plot the stream power, erosion and deposition rates
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(50,15))
for ax in [ax1, ax2, ax3]:
    ax.axis('equal')
    ax.axis('off')


#dhmax = np.abs(delta).mean() * 3.0
ermax = np.abs(erosion_deposition_rate).mean() * 3.0
    
points, simplices, bmask = sp.get_local_mesh()
x = points[:,0]
y = points[:,1]    
    
#im1 = ax1.tripcolor(x, y, sp.tri.simplices, delta, cmap=plt.cm.RdBu, vmin=-dhmax, vmax=dhmax)    
im1 = ax1.tripcolor(x, y, sp.tri.simplices, sp.height, cmap=plt.cm.terrain)
im2 = ax2.tripcolor(x, y, sp.tri.simplices, erosion_deposition_rate, cmap='RdBu')
im3 = ax3.tripcolor(x, y, sp.tri.simplices, deposition_rate, cmap='Reds', vmax=deposition_rate.mean()*3.0)

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)
plt.show()

# %%
fig = plt.figure(1)
ax = fig.add_subplot(111, xlim=(minX, maxX), ylim=(minY, maxY))
im = ax.tripcolor(x, y, sp.tri.simplices, sca.Dimensionalize(sp.height, u.meter).magnitude, cmap='terrain')
fig.colorbar(im, ax=ax, label='height')
plt.show()

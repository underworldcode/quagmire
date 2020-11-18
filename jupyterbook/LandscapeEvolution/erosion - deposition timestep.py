# ---
# jupyter:
#   jupytext:
#     formats: ../../Notebooks/LandscapeEvolution//ipynb,py:light
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

# # Long-range erosion and deposition models - time evolution
#
# Now we assume that for any of the three models, we integrate through time to determine how the localization pattern progresses.

# +
from quagmire import QuagMesh, QuagMesh
from quagmire import tools as meshtools
from quagmire import function as fn
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

# +
minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,
dx, dy = 0.02, 0.02

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, dx, dy)

DM = meshtools.create_DMPlex_from_points(x, y, bmask=None)
spmesh = QuagMesh(DM, verbose=False)

boundary_mask_fn = fn.misc.levelset(spmesh.mask, 0.5)

# +
radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x) + 0.1

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 
height  += 0.5 * (1.0-0.2*radius)
heightn  = height + np.random.random(height.size) * 0.01 # random noise

with spmesh.deform_topography():
    spmesh.topography.data = height
# -


rainfall_fn = (spmesh.topography**2.0)
upstream_precipitation_integral_fn = spmesh.upstream_integral_fn(rainfall_fn)
stream_power_fn = upstream_precipitation_integral_fn**1.5 * spmesh.slope**1.0 * boundary_mask_fn


# Each of the erosion-depositon models require the stream power...

# ## 1. Local equilibrium
#
# The assumption of the stream power law is that sediment transport is in a state of local equilibrium in which the transport rate is (less than or) equal to the local carrying capacity. If we neglect suspended-load transport for a moment and assume only bed-load transport then the local deposition is the amount of material that can be eroded from upstream.

# +
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

    erosion_rate    = mesh._streamwise_smoothing(erosion_rate, 3, centre_weight=0.75)
    deposition_rate = mesh._downhill_smoothing(deposition_rate, 10, centre_weight=0.75)

    # Patch low points, undershoots and smooth flat spots
    
    mesh.topography.unlock()
    mesh.topography.data = np.clip(mesh.topography.data, 0.0, 1.0e99)
    mesh.topography.lock()
    
    low_points = mesh.identify_low_points()
    
    if len(low_points):
        deposition_rate[low_points] = 0.0
        mesh.topography.unlock()
        mesh.topography.data[low_points] = mesh.topography.data[mesh.neighbour_cloud[low_points,0:10]].mean(axis=1)
        mesh.topography.lock()
   
    flat_spots = mesh.identify_flat_spots()

    if len(flat_spots):
        smoothed_deposition_rate = deposition_rate.copy()
        smoothed_deposition_rate[np.invert(flat_spots)] = 0.0   
        for i in range(0,5):
            smoothed_deposition_rate = mesh.rbf_smoother(smoothed_deposition_rate)     
        deposition_rate  += smoothed_deposition_rate
    
    # Update the slope to account for those fixes (maybe height ?)
    
    slope = mesh.slope.evaluate(mesh)
    slope = np.minimum(slope.data, critical_slope)
    
    return erosion_rate, deposition_rate

stream_power = stream_power_fn.evaluate(spmesh)
erosion_rate1, deposition_rate1 = erosion_deposition_1(spmesh, stream_power, efficiency=0.1)

print (erosion_rate1.max(), erosion_rate1.sum())
print (deposition_rate1.max(), deposition_rate1.sum())


deposition_rate1 = spmesh._streamwise_smoothing(deposition_rate1, 10, centre_weight=0.75)

print(deposition_rate1.max(), deposition_rate1.sum())





# -

#
# ## 2. Saltation length
#
# This model relates the length of time it takes for a grain to settle to a material property, $L_s$.
# From Beaumont et al. 1992, Kooi & Beaumont 1994, 1996 we see a linear dependency of deposition flux to stream capacity:
#
# $$
# \frac{dh}{dt} = \frac{dq_s}{dl} = \frac{D_c}{q_c} \left(q_c - q_s \right)
# $$
#
# where
#
# $$
# \frac{D_c}{q_c} = \frac{1}{L_s}
# $$
#
# $D_c$ is the detachment capacity, $q_c$ is the carrying capacity, $q_s$ is the stream capacity, and $L_s$ is the erosion length scale (a measure of the detachability of the substrate). When the flux equals capacity, $q_c = q_s$, no erosion is possible.

# +
def erosion_deposition_2(self, stream_power, efficiency=0.1, length_scale=10.):
    """
    Saltation length from Beaumont et al. 1992
    """
    
    erosion_rate = efficiency*stream_power
    
    cumulative_eroded_material = self.cumulative_flow(erosion_rate*self.area)
    cumulative_deposition_rate = cumulative_eroded_material / self.area
    
    erosion_deposition = 1.0/length_scale * (cumulative_deposition_rate - erosion_rate)
    return erosion_rate, erosion_deposition

erosion_rate2, deposition_rate2 = erosion_deposition_2(spmesh, stream_power, efficiency=0.1, length_scale=10.)


# -

# ## 3. $\xi - q$ model
#
# Davy and Lague (2009) propose a similar suspended-load model that encapsulates a range of behaviours between detachment and transport-limited end members. This model couples erodability as a function of stream power with a sedimentation term weighted by $\alpha$.
#
# $$
# \frac{dh}{dt} = -K q_r^m S^n + \frac{Q_s}{\alpha Q_w}
# $$
#
# where $Q_s$ and $Q_w$ are the sedimentary and water discharge, respectively.

# +
def erosion_deposition_3(self, stream_power, efficiency=0.1, alpha=1.):
    """
    xi - q model from Davy and Lague 2009
    """
    rainflux = rainfall_fn.evaluate(self)
    rainfall = self.area * rainflux
    cumulative_rain = self.cumulative_flow(rainfall)
    cumulative_flow_rate = cumulative_rain / self.area
    erosion_rate = efficiency*stream_power
    
    cumulative_eroded_material = self.cumulative_flow(erosion_rate*self.area)
    cumulative_deposition_rate = cumulative_eroded_material / self.area
    
    deposition_rate = cumulative_deposition_rate / (alpha * cumulative_flow_rate)

    return erosion_rate, deposition_rate

erosion_rate3, deposition_rate3 = erosion_deposition_3(spmesh, stream_power, efficiency=0.1, alpha=1.0)
# -

# ## Time evolution
#
#

# +
height = np.exp(-0.025*(x**2 + y**2)**2) + 0.0001
height += np.random.random(height.size) * 0.0005 # random noise

rain = np.ones_like(height)
rain[np.where(height<0.98)]=0.0

sp.update_height(height)
sp.update_surface_processes(rain, np.zeros_like(rain))
sp.verbose=False

time = 0.0
step = 0
steps = 500

viz_time= 0.0
vizzes = 0

kappa = 1.0e-3
critical_slope = 5.0      # Critical value - assume slides etc take over to limit slope
lowest_slope   = 1.0e-3   # The slope where we cut off the erosion / deposition algorithm
base = 0.0
totalSteps = 0

experiment_name = "ErosionModel1-v1"

# +
import time as systime

walltime = systime.clock()

typical_l = np.sqrt(sp.area)

running_average_uparea = sp.cumulative_flow(sp.area * sp.rainfall_pattern_Variable.data)

for step in range(0,steps):
    
    delta = height-sp.heightVariable.data
    efficiency = 0.01 
    
    ###############################
    ## Compute erosion / deposition
    ###############################
    
    slope = np.minimum(sp.slopeVariable.data, critical_slope)
    stream_power = compute_stream_power(sp, m=1, n=1, critical_slope=critical_slope)

    erosion_rate, deposition_rate = erosion_deposition_1(sp, stream_power, efficiency=0.1, 
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
    
    height0 = sp.heightVariable.data.copy()
    sp.heightVariable.data -= 0.5 * timestep * (erosion_deposition_rate - diffDz )
    sp.heightVariable.data = np.clip(sp.heightVariable.data, base, 1.0e99)   
    
    # Deal with internal drainages (again !)
    
    sp.heightVariable.data = sp.low_points_local_flood_fill()
    gradZx, gradZy = sp.derivative_grad(sp.heightVariable.data)
    sp.slope = np.hypot(gradZx,gradZy)   
    
    # Recalculate based on mid-point values
    
    erosion_rate, deposition_rate = erosion_deposition_1(sp, stream_power, efficiency=0.1, 
                                                         critical_slope=critical_slope)    
    
    erosion_deposition_rate = erosion_rate - deposition_rate
    erosion_timestep    = ((slope + lowest_slope) * typical_l / (np.abs(erosion_rate)+0.000001)).min()
    deposition_timestep = ((slope + lowest_slope) * typical_l / (np.abs(deposition_rate)+0.000001)).min()
   
    diffDz, diff_timestep =  sp.landscape_diffusion_critical_slope(kappa, critical_slope, True)
 
    timestep = min(erosion_timestep, deposition_timestep, diff_timestep)
    
    # Now take the full timestep

    height0 -= timestep * (erosion_deposition_rate - diffDz )
    sp.heightVariable.data = np.clip(height0, base, 1.0e9)  
    sp.heightVariable.data = sp.low_points_local_flood_fill()

    sp.update_height(sp.heightVariable.data)
    # sp.update_surface_processes(rain, np.zeros_like(rain))
    
    running_average_uparea = 0.5 * running_average_uparea + 0.5 * sp.cumulative_flow(sp.area * sp.rainfall_pattern_Variable.data)
 
    if totalSteps%10 == 0:
        print("{:04d} - ".format(totalSteps), \
          " dt - {:.5f} ({:.5f}, {:.5f}, {:.5f})".format(timestep, diff_timestep, erosion_timestep, deposition_timestep), \
          " time - {:.4f}".format(time), \
          " Max slope - {:.3f}".format(sp.slope.max()), \
          " Step walltime - {:.3f}".format(systime.clock()-walltime))
            
              
    # Store data
    
    if( viz_time > 0.1 or step==0):

        viz_time = 0.0
        vizzes = vizzes + 1

        delta = height-sp.height
        smoothHeight = sp.local_area_smoothing(sp.height, its=2, centre_weight=0.75)
         
        if step == 0: 
            sp.save_mesh_to_hdf5("{}-Mesh".format(experiment_name))
            
        sp.save_field_to_hdf5("{}-Data-{:f}".format(experiment_name, totalSteps), 
                              bmask=sp.bmask,
                              height=sp.height, 
                              deltah=delta, 
                              upflow=running_average_uparea, erosion=erosion_deposition_rate)


    ## Loop again 
    totalSteps += 1



# +
# Plot the stream power, erosion and deposition rates
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(50,15))
for ax in [ax1, ax2, ax3]:
    ax.axis('equal')
    ax.axis('off')


dhmax = np.abs(delta).mean() * 3.0
ermax = np.abs(erosion_deposition_rate).mean() * 3.0
    
#im1 = ax1.tripcolor(x, y, sp.tri.simplices, delta, cmap=plt.cm.RdBu, vmin=-dhmax, vmax=dhmax)    
im1 = ax1.tripcolor(x, y, sp.tri.simplices, sp.height, cmap=plt.cm.terrain)
im2 = ax2.tripcolor(x, y, sp.tri.simplices, erosion_deposition_rate, cmap='RdBu', vmin=-ermax, vmax=ermax)
im3 = ax3.tripcolor(x, y, sp.tri.simplices, deposition_rate, cmap='Reds', vmax=deposition_rate.mean()*3.0)

fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)
fig.colorbar(im3, ax=ax3)
plt.show()


# -



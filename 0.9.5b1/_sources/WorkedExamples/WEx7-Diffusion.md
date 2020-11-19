---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Diffusion


```{code-cell}
import numpy as np
from quagmire import QuagMesh 
from quagmire import tools as meshtools
from quagmire import function as fn
from mpi4py import MPI

import lavavu
import stripy
comm = MPI.COMM_WORLD

import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
```

```{code-cell}
from stripy.cartesian_meshes import elliptical_base_mesh_points

epointsx, epointsy, ebmask = elliptical_base_mesh_points(10.0, 7.5, 0.1, remove_artifacts=True)
```

```{code-cell}
emesh = meshtools.elliptical_equispaced_triangulation(10,0.75, 0.1, refinement_levels=0, tree=False, remove_artifacts=True)
```

```{code-cell}
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca()
ax.axes.set_aspect('equal')
ax.scatter(epointsx, epointsy)
ax.scatter(epointsx[~ebmask], epointsy[~ebmask])
```

```{code-cell}
dm = meshtools.create_DMPlex_from_points(epointsx, epointsy, bmask=ebmask, refinement_levels=1)
```

```{code-cell}
mesh = QuagMesh(dm, downhill_neighbours=2, permute=False)

#if comm.rank == 0:
print("Number of nodes in mesh - {}: {}".format(comm.rank, mesh.npoints))
print("Number of boundary nodes - {}".format(np.count_nonzero(~mesh.bmask)))

# retrieve local mesh
x = mesh.tri.x
y = mesh.tri.y

# dm generated bmask

bmask = mesh.bmask


fig = plt.figure()
ax = fig.gca()
ax.axes.set_aspect('equal')
ax.scatter(x, y)
ax.scatter(x[~bmask], y[~bmask])
plt.show()
```

```{code-cell}
import stripy

radial = fn.math.exp(-0.1 * (fn.misc.coord(0)**2.0 + fn.misc.coord(1)**2.0)) 
sinxy  = (fn.math.sin(fn.misc.coord(0)) + fn.math.sin(fn.misc.coord(1)))
height  = radial * sinxy

# various fragments 

draddx = -0.2 * fn.misc.coord(0) * radial 
draddy = -0.2 * fn.misc.coord(1) * radial 

d2raddx2  = -0.2 * radial -0.2 * fn.misc.coord(0) * draddx
d2raddy2  = -0.2 * radial -0.2 * fn.misc.coord(1) * draddy
d2raddxdy = -0.2 * fn.misc.coord(0) * draddy

# First derivatives 

dhdx_fn = draddx * sinxy + radial * fn.math.cos(fn.misc.coord(0))
dhdy_fn = draddy * sinxy + radial * fn.math.cos(fn.misc.coord(1))

# Second derivatives

dh2dx2_fn  = d2raddx2  * sinxy + draddx * fn.math.cos(fn.misc.coord(0)) + draddx * fn.math.cos(fn.misc.coord(0)) - radial * fn.math.sin(fn.misc.coord(0))
dh2dy2_fn  = d2raddy2  * sinxy + draddy * fn.math.cos(fn.misc.coord(1)) + draddy * fn.math.cos(fn.misc.coord(1)) - radial * fn.math.sin(fn.misc.coord(1))
dh2dxdy_fn = d2raddxdy * sinxy + draddx * fn.math.cos(fn.misc.coord(1)) + draddy * fn.math.cos(fn.misc.coord(0)) + radial * 0.0
```

```{code-cell}

```

```{code-cell}

```

```{code-cell}
with mesh.deform_topography():
    mesh.topography.data = height.evaluate(mesh)
    
slope = mesh.slope.evaluate(mesh)
```

```{code-cell}
import lavavu
import stripy

vertices = np.column_stack([x, y, mesh.topography.data])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

# sa = lv.points("subaerial", colour="red", pointsize=0.2, opacity=0.75)
# sa.vertices(vertices[subaerial])

tris = lv.triangles("mesh",  wireframe=True, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
tris.values(mesh.topography.data, label="elevation")
#tris.values(shade, label="shade")
tris.colourmap('dem3')
cb = tris.colourbar()


tris2 = lv.triangles("mesh2",  wireframe=False, colour="#77ff88", opacity=1.0)
tris2.vertices(vertices)
tris2.indices(tri.simplices)
tris2.values(slope, label="slope")
tris2.colourmap('no_green', range=(0.0,0.5))
cb2 = tris2.colourbar()


lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()
```

## Let's try letting this height diffuse with time



```{code-cell}
kappa = fn.parameter(1.0)
dHdX, dHdY = fn.math.grad(mesh.topography)
del2H = fn.math.div(kappa * dHdX, kappa * dHdY)

h_predictor = mesh.add_variable(name="hstar")
dHstardX, dHstardY = fn.math.grad(h_predictor)
del2Hstar          = fn.math.div(kappa * dHstardX, kappa * dHstardY)

h = mesh.add_variable(name="h")
dhdX, dhdY = fn.math.grad(h)
del2h = fn.math.div(kappa * dhdX, kappa * dhdY)

delta_t = (0.5 * mesh.area.min() * kappa ** -1.0).evaluate(0.0,0.0)

h.data = mesh.topography.data.copy()
```

```{code-cell}
## Comparison:

mesh.tri.update_tension_factors(0.0 * mesh.topography.data, tol=0.0001)
mesh.tri.sigma.min(), mesh.tri.sigma.max(), mesh.tri.sigma.mean()

node = 10000 

# V1

dHx, dHy, dHxx, dHxy, dHyy = mesh.tri.second_gradient_local(h.data, node)
dHAx, dHAy =  dhdx_fn.evaluate(mesh)[node], dhdy_fn.evaluate(mesh)[node]

# V2

dH1x_fn, dH1y_fn     = fn.math.grad(mesh.topography)
dH1dxx_fn, dH1dxy_fn = fn.math.grad(dH1x_fn)
dH1dyx_fn, dH1dyy_fn = fn.math.grad(dH1y_fn)

dH1x  = dH1x_fn.evaluate(mesh)[node]
dH1y  = dH1y_fn.evaluate(mesh)[node]
dH1xx = dH1dxx_fn.evaluate(mesh)[node]
dH1xy = dH1dxy_fn.evaluate(mesh)[node]
dH1yy = dH1dyy_fn.evaluate(mesh)[node]

# Analytic
dHAxx, dHAxy, dHAyy = dh2dx2_fn.evaluate(mesh)[node], dh2dxdy_fn.evaluate(mesh)[node], dh2dy2_fn.evaluate(mesh)[node]
```

```{code-cell}
print("dhdx   - {:+8f} | {:+8f} | {:+8f}".format(dHAx,  dH1x,  dHx))
print("dhdy   - {:+8f} | {:+8f} | {:+8f}".format(dHAy,  dH1y,  dHy))
print("dh2dxx - {:+8f} | {:+8f} | {:+8f}".format(dHAxx, dH1xx, dHxx))
print("dh2dxy - {:+8f} | {:+8f} | {:+8f}".format(dHAxy, dH1xy, dHxy))
print("dh2dyy - {:+8f} | {:+8f} | {:+8f}".format(dHAyy, dH1yy, dHyy))
```

```{code-cell}
dHx  = np.empty(mesh.npoints)
dHy  = np.empty(mesh.npoints)
dHxx = np.empty(mesh.npoints)
dHxy = np.empty(mesh.npoints)
dHyy = np.empty(mesh.npoints)
```

```{code-cell}
for i in range(0, mesh.npoints):
    dHx[i], dHy[i], dHxx[i], dHxy[i], dHyy[i] = mesh.tri.second_gradient_local(h.data, i)
```

```{code-cell}
dH1x  = dH1x_fn.evaluate(mesh)
dH1y  = dH1y_fn.evaluate(mesh)
dH1xx = dH1dxx_fn.evaluate(mesh)
dH1xy = dH1dxy_fn.evaluate(mesh)
dH1yy = dH1dyy_fn.evaluate(mesh)
```

```{code-cell}
dhdx_fn_num = mesh.add_variable(name="dhdx_num")
dhdx_fn_num.data = dhdx_fn.evaluate(mesh)
print(dhdx_fn_num.gradient()[0])
print(dhdx_fn_num.gradient()[1])

dhdy_fn_num = mesh.add_variable(name="dhdy_num")
dhdy_fn_num.data = dhdy_fn.evaluate(mesh)
print(dhdy_fn_num.gradient()[1])
```

```{code-cell}
mesh.tri._permutation
```

```{code-cell}
import lavavu
import stripy

vertices = np.column_stack([x, y, s])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

# sa = lv.points("subaerial", colour="red", pointsize=0.2, opacity=0.75)
# sa.vertices(vertices[subaerial])

tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
tris.values(dsdx-dhdx_fn.evaluate(mesh), label="elevation")
#tris.values(shade, label="shade")
tris.colourmap('Blues')
cb = tris.colourbar()

lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()
```

```{code-cell}
3=1
```

```{code-cell}
h.data = mesh.topography.data.copy()

for step in range(0,50):
    if step%10 == 0:
        print("Step {}".format(step))
        
    h_predictor.data = del2h.evaluate(mesh) * 0.5 * delta_t + h.data
    h.data = h.data + del2Hstar.evaluate(mesh) * delta_t * mesh.bmask
      
```

```{code-cell}
print(h.data.min(), h.data.max())
print(mesh.topography.data.min(), mesh.topography.data.max())
```

```{code-cell}
# with mesh.deform_topography():
#     mesh.topography.data = h.data
```

```{code-cell}
import lavavu
import stripy

vertices = np.column_stack([x, y, h.data])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

# sa = lv.points("subaerial", colour="red", pointsize=0.2, opacity=0.75)
# sa.vertices(vertices[subaerial])

tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
tris.values(del2Hstar.evaluate(mesh), label="elevation")
#tris.values(shade, label="shade")
tris.colourmap('Blues')
cb = tris.colourbar()

lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()
```

```{code-cell}

```

```{code-cell}

```

```{code-cell}

```

```{code-cell}
import lavavu
import stripy

vertices = np.column_stack([x, y, mesh.topography.data])
tri = mesh.tri

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)


tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices)
tris.indices(tri.simplices)
# tris.values(mesh.topography.data, label="elevation")
# tris.values(slope, label="slope")
tris.values(del2H.evaluate(mesh), label="del2")
tris.colourmap('no_green')
cb = tris.colourbar()

lv.control.Panel()
lv.control.ObjectList()
# tris.control.Checkbox(property="wireframe")
lv.control.show()
```

```{code-cell}

```

```{code-cell}
import time as systime

walltime = systime.clock()

typical_l = np.sqrt(sp.area)

# running_average_uparea = sp.cumulative_flow(sp.area * sp.rainfall_pattern_Variable.data)

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
```

```{code-cell}

```

```{code-cell}

```

```{code-cell}

```

```{code-cell}

```

```{code-cell}

```

```{code-cell}
3=1
```

```{code-cell}
from quagmire.tools.cloud import quagmire_cloud_fs

quagmire_cloud_fs
quagmire_cloud_fs.listdir("/")
```

```{code-cell}
# from quagmire.tools.cloud import cloud_download, cloud_upload
# cloud_download('global_OC_8.4_topography.h5', "gtopo3.h5")
quagmire_cloud_fs.listdir('/global')
```

```{code-cell}
dm = meshtools.create_DMPlex_from_cloud_fs("global/global_OC_8.4_mesh.h5")
```

```{code-cell}
mesh = QuagMesh(dm, downhill_neighbours=2)

# Mark up the shadow zones

rank = np.ones((mesh.npoints,))*comm.rank
shadow = np.zeros((mesh.npoints,))

# get shadow zones
shadow_zones = mesh.lgmap_row.indices < 0
shadow[shadow_zones] = 1
shadow_vec = mesh.gvec.duplicate()

mesh.lvec.setArray(shadow)
mesh.dm.localToGlobal(mesh.lvec, shadow_vec, addv=True)

rawheight = mesh.add_variable(name="height", locked=False)
rainfall = mesh.add_variable(name="rain", locked=False)
runoff_var = mesh.add_variable(name="runoff", locked=False)

print("{} mesh points".format(mesh.npoints))
```

```{code-cell}
with mesh.deform_topography():
    mesh.topography.load_from_cloud_fs("global/global_OC_8.4_topography.h5")
```

```{code-cell}
low_points = mesh.identify_low_points(ref_height=6.37)
low_points.shape
```

```{code-cell}
rainfall.data = 0.0
rainfall.load_from_cloud_fs("global/global_OC_8.4_rainfall.h5", quagmire_cloud_fs)
rainfall.data
```

```{code-cell}
runoff_var.data = 0.0
runoff_var.load_from_cloud_fs("global/global_OC_8.4_runoff.h5", quagmire_cloud_fs)
runoff_var.data
```

```{code-cell}
# # runoff  "/thredds/wcs/agg_terraclimate_q_1958_CurrentYear_GLOBE.nc"

# from owslib.wcs import WebCoverageService
# # import gdal

# url = "http://thredds.northwestknowledge.net:8080/thredds/wcs/agg_terraclimate_ppt_1958_CurrentYear_GLOBE.nc"
# wcs = WebCoverageService(url, version='1.0.0')
# for layer in list(wcs.contents):
#     print("Layer Name:", layer)
#     print("Title:", wcs[layer].title, '\n')
    
# output = wcs.getCoverage(identifier=layer,
#                     service="WCS", bbox=[-180, -90, 180, 90], 
#                     resx = 1800.0 / 3600.0, resy = 1800.0 / 3600.0,
#                     format='geotiff')

# with open("GlobalRainfall.tif", "wb") as f:
#     f.write(output.read())
    
# # Read it back and reduce the size of the array

# url = "http://thredds.northwestknowledge.net:8080/thredds/wcs/agg_terraclimate_q_1958_CurrentYear_GLOBE.nc"
# wcs = WebCoverageService(url, version='1.0.0')
# for layer in list(wcs.contents):
#     print("Layer Name:", layer)
#     print("Title:", wcs[layer].title, '\n')
    
# output = wcs.getCoverage(identifier=layer,
#                     service="WCS", bbox=[-180, -90, 180, 90], 
#                     resx = 1800.0 / 3600.0, resy = 1800.0 / 3600.0,
#                     format='geotiff')

# with open("GlobalRunoff.tif", "wb") as f:
#     f.write(output.read())
    
```

```{code-cell}
# import imageio
# rain = imageio.imread("GlobalRainfall.tif")[::3,::3].astype(float)
# runoff = imageio.imread("GlobalRunoff.tif")[::3,::3].astype(float)

# [cols, rows] = rain.shape
# print([cols,rows])

# rlons = np.linspace(-180,180, rows)
# rlats = np.linspace(-180,180, cols)

# rx,ry = np.meshgrid(rlons.data, rlats.data)


# rainfall.data  = np.maximum(0.0,meshtools.map_global_raster_to_strimesh(mesh, rain[::-1,:]))
# runoff_var.data  = np.maximum(0.0,meshtools.map_global_raster_to_strimesh(mesh, runoff[::-1,:]))
```

```{code-cell}

# coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
#                            edgecolor=(1.0,0.8,0.0),
#                            facecolor="none")

# ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
#                            edgecolor="green",
#                            facecolor="blue")

# lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
#                            edgecolor="green",
#                            facecolor="blue")

# rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
#                            edgecolor="green",
#                            facecolor="blue")

# map_extent = ( -180, 180, -90, 90 )

# plt.figure(figsize=(15, 10))
# ax = plt.subplot(111, projection=ccrs.PlateCarree())
# ax.set_extent(map_extent)

# ax.add_feature(coastline, edgecolor="black", linewidth=0.5, zorder=3)
# ax.add_feature(lakes,     edgecolor="black", linewidth=1, zorder=3)
# ax.add_feature(rivers   , edgecolor="black", facecolor="none", linewidth=1, zorder=3)

# plt.imshow(rain, extent=map_extent, transform=ccrs.PlateCarree(),
#            cmap='Greens', origin='upper', vmin=0., vmax=50.)
```

```{code-cell}
latitudes_in_radians  = mesh.tri.lats
longitudes_in_radians = mesh.tri.lons 
latitudes_in_degrees  = np.degrees(latitudes_in_radians) 
longitudes_in_degrees = np.degrees(longitudes_in_radians) 

map_extent = ( -180, 180, -90, 90 )

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)

ax.add_feature(coastline, edgecolor="black", linewidth=0.5, zorder=3)
ax.add_feature(lakes,     edgecolor="black", linewidth=1, zorder=3)
ax.add_feature(rivers   , edgecolor="black", facecolor="none", linewidth=1, zorder=3)

plt.scatter(x=longitudes_in_degrees, y=latitudes_in_degrees, c=rainfall.data, transform=ccrs.PlateCarree(),
            cmap='Greens',  vmin=0., vmax=50.)
```

```{code-cell}
from quagmire import function as fn

ones = fn.parameter(1.0, mesh=mesh)
cumulative_flow_0 = np.log10(1.0e-20 + mesh.upstream_integral_fn(runoff_var).evaluate(mesh))
cumulative_flow_0[mesh.topography.data < 6.37] = 0.0

cumulative_area = np.log10(1.0e-20 + mesh.upstream_integral_fn(ones).evaluate(mesh))
cumulative_area[mesh.topography.data < 6.37] = 0.0
```

```{code-cell}
import lavavu
import stripy

# vertices0 = mesh.tri.points*mesh_height.reshape(-1,1)
vertices = mesh.tri.points*mesh.topography.data.reshape(-1,1)
tri = mesh.tri

lv = lavavu.Viewer(border=False, axis=False, background="#FFFFFF", resolution=[1000,1000], near=-20.0)

lows = lv.points("lows", colour="red", pointsize=5.0, opacity=0.75)
lows.vertices(vertices[low_points])

flowball = lv.points("flowballs", pointsize=1.5, colour="rgb(50,50,100)", opacity=0.25)
flowball.vertices(vertices*1.001)
flowball.values(np.maximum(0.0,cumulative_flow_0-11.0), label="flows")
flowball["sizeby"]="flows"

ghostball = lv.points("ghostballs", colour="rgb(50,50,50)", pointsize=0.5, opacity=0.2)
ghostball.vertices(vertices*1.001)
ghostball.values(np.maximum(0.0,cumulative_area-8.0), label="areas")
ghostball["sizeby"]="areas"

heightball = lv.points("heightballs", pointsize=1.0, opacity=1.0)
heightball.vertices(vertices)
heightball.values(mesh.topography.data, label="height")
heightball.values((mesh.topography.data > 6.370).astype(float), label="land")
heightball.colourmap('geo',  range=(6.363,6.377))  # This is a good choice of colourmap and range to make the coastlines work and the Earth look nice 
heightball["sizeby"]="land"

tris = lv.triangles("mesh",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(vertices*0.999)
tris.indices(tri.simplices)
tris.values(mesh.topography.data, label="elevation")
tris.colourmap('#999999 #222222', range=(6.363,6.377))  # This is a good choice of colourmap and range to make the coastlines work and the Earth look nice 


# lv.translation(-1.012, 2.245, -13.352)
# lv.rotation(53.217, 18.104, 161.927)

lv.control.Panel()
lv.control.ObjectList()
lv.control.show()
```

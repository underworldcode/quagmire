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

## Geoscience Australia 9s DEM Map of Tasmania

Here we show a workflow for handling the higher resolution (9 arc second) DEM of Tasmania supplied by Geoscience Australia. This has been clipped using gdaltranslate to capture the area of interest and save it as a geotiff file. This has roughly 2.4 million points on the island of Tasmania. This DEM is hydrologically enforced at the outset and therefore serves as a consistency test for the `quagmire` flow algorithms etc. 

In this notebook, we read the original DEM, check it for consistency and (SPOILER !) make a few adjustments to account for peculiarities of the DEM associated with the various dams in the hydro-schemes. 

We then save the processed DEM ... 
(TODO: parallel HDF5 would be better)

### Dependencies

  - `quagmire` 
  - `gdal`     - used to read and write geotiff files
  - `cartopy`  - to produce maps
  - `lavavu`   - for 3D visualisations

```{code-cell}
import numpy as np
import quagmire
from quagmire import function as fn
from quagmire import tools as meshtools

import gdal

%pylab inline
```

```{code-cell}
file = "data/dem9s-tassie-quagmire.tif"
ds = gdal.Open(file)
band = ds.GetRasterBand(1)
height = band.ReadAsArray()
[cols, rows] = height.shape

left, hres, n0, top, n1, vres  = ds.GetGeoTransform()
right = left+rows*hres
bottom = top+cols*vres
x,y = np.meshgrid(np.arange(left, right, hres), np.arange(top,  bottom, vres))
```

```{code-cell}
ds = gdal.Open(file)
ds.GetProjection()
```

```{code-cell}
from scipy.ndimage.filters import gaussian_filter

point_mask =  height > -0.5

#corners
point_mask[0,0] = 1.0
point_mask[0,-1] = 1.0
point_mask[-1,0] = 1.0
point_mask[-1,-1] = 1.0

xs = x[point_mask]
ys = y[point_mask]
heights = height[point_mask]
points = np.column_stack([xs, ys])

submarine = (heights <  10 )
subaerial = (heights >= 10 )
```

```{code-cell}
DM = meshtools.create_DMPlex_from_spherical_points(xs, ys, bmask=subaerial)
mesh = quagmire.QuagMesh(DM, downhill_neighbours=2)
```

```{code-cell}
with mesh.deform_topography():
    mesh.topography.data = heights                                                                 
```

```{code-cell}
low_points1 = mesh.identify_low_points()
low_point_coords1 = mesh.coords[low_points1] 
print(low_points1.shape)

cumulative_flow_1 = mesh.upstream_integral_fn(mesh.topography).evaluate(mesh)
topography_1 = mesh.topography.data[:]

outflow_points1 = np.unique(np.hstack(( mesh.identify_outflow_points(), mesh.identify_low_points())))
upstream_area1  = mesh.upstream_integral_fn(fn.misc.levelset(mesh.topography, 0.0)).evaluate(mesh)
print(mesh.identify_outflow_points().shape)
```

```{code-cell}
## plot the results

import cartopy.crs as ccrs
import cartopy.feature as cfeature

coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '10m',
                           edgecolor=(1.0,0.8,0.0),
                           facecolor="none")

ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '10m',
                           edgecolor="green",
                           facecolor="blue")

lakes = cfeature.NaturalEarthFeature('physical', 'lakes', '10m',
                           edgecolor="green",
                           facecolor="blue")

rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m',
                           edgecolor="green",
                           facecolor="blue")

map_extent = ( left, right, bottom, top)

logflow = np.log10(1.0e-3+upstream_area1)
flows1 = logflow.min() * np.ones_like(height)
flows1[point_mask] = logflow

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)

# ax.add_feature(coastline, edgecolor="black", linewidth=1, zorder=3)

ax.add_feature(lakes,     edgecolor="black", facecolor="none", linewidth=1, zorder=3)
ax.add_feature(rivers   , edgecolor="black", facecolor="none", linewidth=1, zorder=3)

# ax.scatter(xs[submarine],ys[submarine], color="#000044", s=.1)

plt.imshow(flows1, extent=map_extent, transform=ccrs.PlateCarree(),
           cmap='Blues', origin='upper')

ax.scatter(xs[outflow_points1], ys[outflow_points1], color="Green", s=5)
ax.scatter(xs[low_points1], ys[low_points1], color="Red", s=5)


plt.savefig("WEx4-Flowpath-1.png", dpi=250)
```

## Apply pit filling / local-flooding / swamp filling algorithm

The pit filling is for very small local minima where the basin filling / swamp algorithm is not appropriate. The local flooding is a simple upward height propagation from a blockage with a limit on the distance that it can propagate. 

The swamp algorithm is for extensive regions that have only internal drainage. Some changes to the identification of "erroneous" low points is needed for cases where internal drainages are expected.

At least one extra round of iteration is often helpful.

In this case, the hydrologically enforced DEM should not have any local minima but there are some issues that are associated with water bodies that are dammed and this does, as a result, need a little modification which we compute here and analyse after the fact. 

```{code-cell}
# This should not be necessary but there can be some issues with very flat regions not having sufficient relief for the flow directions
# to be recorded.

mesh.low_points_local_patch_fill(its=10, smoothing_steps=2)
topography_2 = mesh.topography.data[:]
cumulative_flow_2 = mesh.upstream_integral_fn(mesh.topography**2).evaluate(mesh)
low_points2 = mesh.identify_low_points()
low_point_coords2 = mesh.coords[low_points2] 
print("Low points - {}".format(low_points2.shape))


for i in range(0,20):
    mesh.low_points_swamp_fill(ref_height=0.0, ref_gradient=0.1)
    
    # In parallel, we can't break if ANY processor has work to do (barrier / sync issue)
    low_points3 = mesh.identify_global_low_points()
    
    print("{} : {}".format(i,low_points3[0]))
    if low_points3[0] == 0:
        break
```

```{code-cell}
cumulative_flow_3 = mesh.upstream_integral_fn(mesh.topography**2).evaluate(mesh)
upstream_area3    = mesh.upstream_integral_fn(fn.misc.levelset(mesh.topography, 0.0)).evaluate(mesh)

low_points3 = mesh.identify_low_points()
topography_3 = mesh.topography.data[:]

print("Low points - {}".format(low_points3.shape))
outflow_points3 = np.unique(np.hstack(( mesh.identify_outflow_points(), mesh.identify_low_points())))
```

```{code-cell}
logflow = np.log10(1.0e-3+upstream_area3)
flows3 = logflow.min() * np.ones_like(height)
flows3[point_mask] = logflow

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)


ax.add_feature(coastline,     edgecolor="black", facecolor="none", linewidth=1, zorder=3)
ax.add_feature(lakes,     edgecolor="black", facecolor="none", linewidth=1, zorder=3)
ax.add_feature(rivers   , edgecolor="black", facecolor="none", linewidth=1, zorder=3)

ax.scatter(xs[outflow_points3],ys[outflow_points3], color="#00FF44", s=.5, zorder=2)
ax.scatter(xs[low_points3],ys[low_points3], color="#00FF44", s=.5, zorder=3)

plt.imshow(flows3, extent=map_extent, transform=ccrs.PlateCarree(),
           cmap='Blues', origin='upper', vmin=-3.5, vmax=-1.5, zorder=1)
```

```{code-cell}
## Modify the downhill neighbour connectivity

mesh1 = quagmire.QuagMesh(DM, downhill_neighbours=1)
with mesh1.deform_topography():
    mesh1.topography.data = mesh.topography.data 
    
```

```{code-cell}
# We want to exclude from the catchments some of the triangles that go to edges or to other islands
# as these really skew the area calculations

topomask = mesh1.add_variable("topomask")
topomask.data = np.where(mesh1.topography.data > 1, 1.0, 0.0)

# large triangles associated with boundaries need to be excluded (choose by inspection)
area_threshold = np.percentile(mesh1.area, 95)
topomask.data = np.where(mesh1.area < area_threshold, topomask.data, 0.0)

area = mesh1.upstream_integral_fn(topomask).evaluate(mesh1)

outflow_points3 = np.unique(np.hstack(( mesh1.identify_outflow_points()))) # , mesh1.identify_low_points())))

# log_catchment_areas = np.sort(1.0e-10+np.log(area[outflow_points3]))[::-1]
catchment_areas = np.sort(area[outflow_points3])[::-1]
cum_catchment_areas = np.cumsum(catchment_areas)
total_area = mesh1.area.sum()

plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
ax.set_xlim(0,50)
ax.plot(100.0*cum_catchment_areas/total_area)
ax.bar(x=range(0,catchment_areas.shape[0]), height=100.0*catchment_areas/catchment_areas[0])
```

```{code-cell}
ordered_catchments = np.argsort(area[outflow_points3])[::-1]
catchments = mesh1.add_variable(name="catchments")
catchments.data = mesh1.uphill_propagation(points = outflow_points3[ordered_catchments[0:100]], values=np.indices((100,)), fill=-1.0, its=1000)
```

```{code-cell}
catch = []
for i in range(0,outflow_points3.shape[0]):
    catch.append( np.where(catchments.data==i) )
```

```{code-cell}
for i in range(0,25):
    print(catch[i][0].shape, area[outflow_points3[ordered_catchments[i]]])
```

```{code-cell}
# catch_img3 = -2.0 * np.ones_like(height)
# catch_img3[point_mask] = catchments.data

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)

ax.add_feature(coastline, edgecolor="black", linewidth=1, zorder=3)
ax.add_feature(lakes,     edgecolor="black", facecolor="none", linewidth=1, zorder=3)
ax.add_feature(rivers   , edgecolor="Yellow", facecolor="none", linewidth=1, zorder=3)

for i in range(0,15):
    ax.scatter(xs[catch[i]], ys[catch[i]], s=20, alpha=0.5)

ax.scatter(xs[outflow_points3], ys[outflow_points3], color="Green", s=1.0)
ax.scatter(xs[low_points3],     ys[low_points3], color="Red", s=25.0)

plt.imshow(flows3, extent=map_extent, transform=ccrs.PlateCarree(),
           cmap='Blues', origin='upper', vmin=-3.5, vmax=-2.5, alpha=0.5, zorder=10)

plt.savefig("WEx4-15Catchments.png", dpi=250)
```

```{code-cell}
# catch_img3 = -2.0 * np.ones_like(height)
# catch_img3[point_mask] = catchments.data

plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)

ax.add_feature(coastline, edgecolor="black", linewidth=1, zorder=30)

# for i in range(0,15):
#     ax.scatter(xs[catch[i]], ys[catch[i]], s=20, alpha=0.5)

plt.imshow(flows3, extent=map_extent, transform=ccrs.PlateCarree(),
           cmap='Greys', origin='upper', vmin=-3.0, vmax=-2.5, alpha=1.0, zorder=10)

plt.savefig("WEx4-RiversBW.png", dpi=500)
```

```{code-cell}
plt.figure(figsize=(15, 10))
ax = plt.subplot(111, projection=ccrs.PlateCarree())
ax.set_extent(map_extent)

ax.add_feature(coastline, edgecolor="black", linewidth=1, zorder=30)

for i in range(0,100):
    ax.scatter(xs[catch[i]], ys[catch[i]], s=0.05, alpha=0.5)

# plt.imshow(flows3, extent=map_extent, transform=ccrs.PlateCarree(),
#            cmap='Blues', origin='upper', vmin=-3.5, vmax=-2.5, alpha=1.0, zorder=10)

plt.savefig("WEx4-100Catchments-only.png", dpi=500)
```

```{code-cell}
catch_img = np.zeros_like(height)
catch_img[point_mask] = catchments.data
catch_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=5.0)

logflow = np.log10(1.0e-3+upstream_area3)
flows_img = logflow.min() * np.ones_like(height)
flows_img[point_mask] = logflow
flows_norm = matplotlib.colors.Normalize(vmin=-3.0, vmax=-2.5)
```

```{code-cell}
logflow.max()
```

```{code-cell}

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=4.0)
im = (0.5+0.5*cm.Greys_r(catch_norm(catch_img%5.0))) * (0.2+0.8*cm.Blues(flows_norm(flows_img)))
im[..., 0:3][~point_mask] = (0.8,0.9,1.0)

import lavavu

points = mesh.data

low_point_coords3 = points[low_points3]
outflow_point_coords3 = points[outflow_points3]

low_point_coords1 = points[low_points1]

lv = lavavu.Viewer(border=False, background=(0.8,0.9,1.0), resolution=[1200,600], near=-10.0, axis=False)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)
tri1.texture(im)

lv.control.Panel()
lv.control.ObjectList()
lv.control.show()
```

```{code-cell}
lv.image(filename="WEx4-3DFlowpathsCatchments.png", resolution=(3000,1500), quality=100)
```

```{code-cell}

```

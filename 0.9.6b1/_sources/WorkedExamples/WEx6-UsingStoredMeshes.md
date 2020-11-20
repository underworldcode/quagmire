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

## Models from the cloud


```{code-cell}
import numpy as np
from quagmire import QuagMesh 
from quagmire import tools as meshtools
from mpi4py import MPI

import lavavu
import stripy
comm = MPI.COMM_WORLD

import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# from scipy.ndimage.filters import gaussian_filter
```

```{code-cell}
# dm = meshtools.create_DMPlex_from_hdf5("global_OC_8.4_mesh.h5")
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

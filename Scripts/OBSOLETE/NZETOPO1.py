
# coding: utf-8

# # Meshing Australia
#
# In this notebook we:
#
# 1. Import a coastline from an ESRI shapefile
# 2. Sample its interior using the poisson disc generator
# 3. Resample the interior using a DEM
# 4. Create a DM object and refine a few times
# 5. Save the mesh to HDF5 file 

# In[1]:

from osgeo import gdal

import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
#get_ipython().magic('matplotlib inline')

import quagmire
from quagmire import tools as meshtools

import shapefile
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter
#from matplotlib.colors import LightSource
from petsc4py import PETSc




# ne_land = shapefile.Reader("../Notebooks/data/ne_110m_land/ne_110m_land.shp")
# land_shapes = ne_land.shapeRecords()
#
# polyList = []
# for i,s in  enumerate(ne_land.shapes()):
#     if len(s.points) < 3:
#         print "Dodgy Polygon ", i, s
#     else:
#         p = Polygon(s.points)
#         if p.is_valid:
#             polyList.append(p)
#
# pAll_ne110 = MultiPolygon(polyList)
# tas_poly_ne110 = 11
# ausmain_poly_ne110 = 21
#
# AusLandPolygon_ne110 = MultiPolygon([polyList[tas_poly_ne110], polyList[ausmain_poly_ne110]])


# In[4]:

ne_land = shapefile.Reader("../Notebooks/data/ne_50m_land/ne_50m_land.shp")
land_shapes = ne_land.shapeRecords()

polyList = []
for i,s in  enumerate(ne_land.shapes()):
    if len(s.points) < 3:
        print "Dodgy Polygon ", i, s
    else:
        p = Polygon(s.points)
        if p.is_valid:
            polyList.append(p)

NZNorthI_poly_ne50 = 96
NZSouthI_poly_ne50 = 97
NZLandPolygon_ne50 = MultiPolygon([polyList[NZNorthI_poly_ne50], polyList[NZSouthI_poly_ne50]])

LandAreaPolygon_ne50 = NZLandPolygon_ne50

# In[5]:

# AusLandPolygon_ne50


# In[6]:

from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

bounds = LandAreaPolygon_ne50.bounds
minX, minY, maxX, maxY = bounds


## All of this should be done on Rank 0 (the DM is built only on rank 0 )

if PETSc.COMM_WORLD.rank == 0 or PETSc.COMM_WORLD.size == 1:

    print "Build grid points"

#    x1, y1, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, 0.25)

    xres = 500
    yres = 500

    xx = np.linspace(minX, maxX, xres)
    yy = np.linspace(minY, maxY, yres)
    x1, y1 = np.meshgrid(xx,yy)
    x1 += np.random.random(x1.shape) * 0.2 * (maxX-minX) / xres
    y1 += np.random.random(y1.shape) * 0.2 * (maxY-minY) / yres

    x1 = x1.flatten()
    y1 = y1.flatten()

    pts = np.stack((x1, y1)).T
    mpt = MultiPoint(points=pts)

    print "Find Coastline / Interior"

    interior_mpts = mpt.intersection(NZLandPolygon_ne50)
    interior_points = np.array(interior_mpts)

    fatBoundary = LandAreaPolygon_ne50.buffer(0.5) # A puffed up zone around the interior points
    boundary = fatBoundary.difference(LandAreaPolygon_ne50)
    inBuffer = mpt.intersection(boundary)

    buffer_points = np.array(inBuffer)

    ## Make a new collection of points to stuff into a DM

    ibmask = np.ones((interior_points.shape[0]), dtype=np.bool)
    bbmask = np.zeros((buffer_points.shape[0]), dtype=np.bool)

    bmask = np.hstack((ibmask, bbmask))
    pts = np.vstack((interior_points, buffer_points))

    x1 = pts[:,0]
    y1 = pts[:,1]

    # ### 3. Create the DM
    #
    # The points are now read into a DM and refined so that we can achieve very high resolutions. Refinement is achieved by adding midpoints along line segments connecting each point.

print "Create DM"


if not (PETSc.COMM_WORLD.rank == 0 or PETSc.COMM_WORLD.size == 1):
    x1 = None
    y1 = None
    bmask = None

DM = meshtools.create_DMPlex_from_points(x1, y1, bmask, refinement_steps=1)

del x1, y1, bmask

print "Built and distributed DM"

mesh = quagmire.SurfaceProcessMesh(DM, verbose=True)
print mesh.dm.comm.rank, ": Points: ", mesh.npoints

# In[71]:

x2r = mesh.tri.x
y2r = mesh.tri.y
simplices = mesh.tri.simplices
bmaskr = mesh.bmask
coords = np.stack((y2r, x2r)).T

print "Map DEM to points"

gtiff = gdal.Open("../Notebooks/data/ETOPO1_Ice_c_geotiff.tif")

width = gtiff.RasterXSize
height = gtiff.RasterYSize
gt = gtiff.GetGeoTransform()
# minX = gt[0]
# minY = gt[3] + width*gt[4] + height*gt[5]
# maxX = gt[0] + width*gt[1] + height*gt[2]
# maxY = gt[3]

img = gtiff.GetRasterBand(1).ReadAsArray().T
# img = np.flipud(img).astype(float)

img = np.fliplr(img)

# print minX, minY, maxX, maxY

bounds = LandAreaPolygon_ne50.bounds
minX, minY, maxX, maxY = bounds

sliceLeft   = int(180+minX) * 60
sliceRight  = int(180+maxX) * 60
sliceBottom = int(90+minY) * 60
sliceTop    = int(90+maxY) * 60

LandImg = img[ sliceLeft:sliceRight, sliceBottom:sliceTop].T
LandImg = np.flipud(LandImg)

print LandImg.shape

img = LandImg

#
#
# gtiff = gdal.Open("../Notebooks/data/ausbath_09_v4.tiff")
# width = gtiff.RasterXSize
# height = gtiff.RasterYSize
# gt = gtiff.GetGeoTransform()
# minX = gt[0]
# minY = gt[3] + width*gt[4] + height*gt[5]
# maxX = gt[0] + width*gt[1] + height*gt[2]
# maxY = gt[3]
#
# img = gtiff.GetRasterBand(1).ReadAsArray()

im_coords = coords.copy()
im_coords[:,0] -= minY
im_coords[:,1] -= minX

im_coords[:,0] *= img.shape[0] / (maxY-minY)
im_coords[:,1] *= img.shape[1] / (maxX-minX)

im_coords[:,0] =  img.shape[0] - im_coords[:,0]


from scipy import ndimage

spacing = 1.0
coords = np.stack((y2r, x2r)).T / spacing

## Heights from DEM and add geoid.

meshheights = ndimage.map_coordinates(img, im_coords.T, order=3, mode='nearest')
meshheights = np.maximum(-100.0, meshheights)
meshheights = mesh.rbf_smoother(meshheights, iterations=2)
# meshheights += 40.0*(mesh.coords[:,0]-minX)/(maxX-minX) + 40.0*(mesh.coords[:,1]-minY)/(maxY-minY)

# Some bug in the bmask after refinement means we need to check this

questionable = np.logical_and(bmaskr, meshheights < 10.0)
qindex = np.where(questionable)[0]

for index in qindex:
    point = Point(mesh.coords[index])
    if not LandAreaPolygon_ne50.contains(point):
         bmaskr[index] =  False

# and this (the reverse condition)

questionable = np.logical_and(~bmaskr, meshheights > -1.0)
qindex = np.where(questionable)[0]

for index in qindex:
     point = Point(mesh.coords[index])
     if  LandAreaPolygon_ne50.contains(point):
          bmaskr[index] =  True




print "Downhill Flow"

# m v km !

mesh.downhill_neighbours=2
mesh.update_height(meshheights*0.001)


print "Flowpaths 1 - Lows included"

nits, flowpaths = mesh.cumulative_flow_verbose(mesh.area*np.ones_like(mesh.height), verbose=True, maximum_its=2500)
flowpaths = mesh.rbf_smoother(flowpaths, iterations=1)
flowpaths[~bmaskr] = -1.0


# super_smooth_topo = mesh.rbf_smoother(mesh.height, iterations=100)
# mesh.update_height(super_smooth_topo)
#
# print "Flowpaths - Smooth"
# nits, flowpathsSmooth = mesh.cumulative_flow_verbose(np.ones_like(mesh.height), verbose=True, maximum_its=1500)
# flowpathsSmooth = mesh.rbf_smoother(flowpathsSmooth, iterations=1)
# flowpathsSmooth[~bmaskr] = 0.0

new_heights=mesh.low_points_local_fill(its=2, smoothing_steps=2)
mesh._update_height_partial(new_heights)
low_points2 = mesh.identify_low_points()
print "Low Points", low_points2.shape


for i in range(0,10):
    new_heights = mesh.low_points_swamp_fill()
    mesh._update_height_partial(new_heights)
    # mesh.update_height(new_heights)
    low_points2 = mesh.identify_low_points()
    print low_points2.shape

print "Low Points", low_points2.shape

print "Flowpaths 2 - Lows patched"

raw_heights=mesh.height
mesh.update_height(new_heights)


print "Flowpaths 1 - Lows included"

nits, flowpaths2 = mesh.cumulative_flow_verbose(mesh.area*np.ones_like(mesh.height), verbose=True, maximum_its=2500)
flowpaths2 = mesh.rbf_smoother(flowpaths2, iterations=1)
flowpaths2[~bmaskr] = -1.0


print "Downhill Flow - complete"


filename = 'NZ-ETOPO.h5'

decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=meshheights*0.001,
                                  slope=mesh.slope,
                                  flowLP=np.sqrt(flowpaths),
                                  flow=np.sqrt(flowpaths1),
                                  lakes=mesh.height - raw_heights,
                                  decomp=decomp)

# to view in Paraview
meshtools.generate_xdmf(filename)


# In[ ]:

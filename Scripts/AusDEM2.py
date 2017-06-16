
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


# ## 1. Import coastline shapefile
#
# This requires pyshp to be installed. We scale the points to match the dimensions of the DEM we'll use later.

# In[2]:

# def remove_duplicates(a):
#     """
#     find unique rows in numpy array
#     <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array>
#     """
#     b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
#     dedup = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])
#     return dedup
#
# coast_shape = shapefile.Reader("../Notebooks/data/AustCoast/AustCoast2.shp")
# shapeRecs = coast_shape.shapeRecords()
# coords = []
# for record in shapeRecs:
#     coords.append(record.shape.points[:])
#
# coords = np.vstack(coords)
#
# # Remove duplicates
# points = remove_duplicates(coords)


# In[3]:

ne_land = shapefile.Reader("../Notebooks/data/ne_110m_land/ne_110m_land.shp")
land_shapes = ne_land.shapeRecords()

polyList = []
for i,s in  enumerate(ne_land.shapes()):
    if len(s.points) < 3:
        print "Dodgy Polygon ", i, s
    else:
        p = Polygon(s.points)
        if p.is_valid:
            polyList.append(p)

pAll_ne110 = MultiPolygon(polyList)
tas_poly_ne110 = 11
ausmain_poly_ne110 = 21

AusLandPolygon_ne110 = MultiPolygon([polyList[tas_poly_ne110], polyList[ausmain_poly_ne110]])


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

pAll_ne50 = MultiPolygon(polyList)
tas_poly_ne50 = 89
ausmain_poly_ne50 = 6

AusLandPolygon_ne50 = MultiPolygon([polyList[tas_poly_ne50], polyList[ausmain_poly_ne50]])


# In[5]:

# AusLandPolygon_ne50


# In[6]:

from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon

ausBounds = [110, -45 , 155, -5]
minX, minY, maxX, maxY = ausBounds


## All of this should be done on Rank 0 (the DM is built only on rank 0 )

if PETSc.COMM_WORLD.rank == 0 or PETSc.COMM_WORLD.size == 1:

    print "Build grid points"

#    x1, y1, bmask = meshtools.poisson_disc_sampler(minX, maxX, minY, maxY, 0.25)

    xx = np.linspace(minX, maxX, 500)
    yy = np.linspace(minY, maxY, 250)
    x1, y1 = np.meshgrid(xx,yy)
    x1 += np.random.random(x1.shape) * 0.05 * (maxX-minX) / 500.0
    y1 += np.random.random(y1.shape) * 0.05 * (maxY-minY) / 250.0

    x1 = x1.flatten()
    y1 = y1.flatten()

    pts = np.stack((x1, y1)).T
    mpt = MultiPoint(points=pts)

    print "Find Coastline / Interior"

    interior_mpts = mpt.intersection(AusLandPolygon_ne50)
    interior_points = np.array(interior_mpts)

    fatAus = AusLandPolygon_ne50.buffer(0.5) # A puffed up zone around the interior points
    ausBoundary = fatAus.difference(AusLandPolygon_ne50)
    inBuffer = mpt.intersection(ausBoundary)

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

DM = meshtools.create_DMPlex_from_points(x1, y1, bmask, refinement_steps=3)

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

gtiff = gdal.Open("../Notebooks/data/ausbath_09_v4.tiff")
width = gtiff.RasterXSize
height = gtiff.RasterYSize
gt = gtiff.GetGeoTransform()
minX = gt[0]
minY = gt[3] + width*gt[4] + height*gt[5]
maxX = gt[0] + width*gt[1] + height*gt[2]
maxY = gt[3]

img = gtiff.GetRasterBand(1).ReadAsArray()

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
meshheights = mesh.rbf_smoother(meshheights, iterations=10)
meshheights += 40.0*(mesh.coords[:,0]-minX)/(maxX-minX) + 40.0*(mesh.coords[:,1]-minY)/(maxY-minY)

questionable = np.logical_and(bmaskr, meshheights < 10.0)
qindex = np.where(questionable)[0]

for index in qindex:
    point = Point(mesh.coords[index])
    if not AusLandPolygon_ne50.contains(point):
         bmaskr[index] =  False


print "Downhill Flow"

# m v km !

mesh.downhill_neighbours=2
mesh.update_height(meshheights*0.001)

# print "Flowpaths "
# nits, flowpaths = mesh.cumulative_flow_verbose(np.ones_like(mesh.height), verbose=True, maximum_its=1500)
# flowpaths = mesh.rbf_smoother(flowpaths, iterations=1)
# flowpaths[~bmaskr] = 0.0

mesh.handle_low_points(its=200)

print "Flowpaths - Low point"
nits, flowpaths = mesh.cumulative_flow_verbose(np.ones_like(mesh.height), verbose=True, maximum_its=1500)
flowpaths = mesh.rbf_smoother(flowpaths, iterations=1)
flowpaths[~bmaskr] = 0.0


super_smooth_topo = mesh.rbf_smoother(mesh.height, iterations=100)
mesh.update_height(super_smooth_topo)

print "Flowpaths - Smooth"
nits, flowpathsSmooth = mesh.cumulative_flow_verbose(np.ones_like(mesh.height), verbose=True, maximum_its=1500)
flowpathsSmooth = mesh.rbf_smoother(flowpathsSmooth, iterations=1)
flowpathsSmooth[~bmaskr] = 0.0


print "Downhill Flow - complete"


filename = 'austopo-v-smooth.h5'

decomp = np.ones_like(mesh.height) * mesh.dm.comm.rank

mesh.save_mesh_to_hdf5(filename)
mesh.save_field_to_hdf5(filename, height=mesh.height,
                                  slope=mesh.slope,
                                  flowLP=np.sqrt(flowpaths),
                                  flowSmooth=np.sqrt(flowpathsSmooth),
                                  decomp=decomp)

# to view in Paraview
meshtools.generate_xdmf(filename)


# In[ ]:

# Spherical mesh of Australia

Download a GeoTiff from Geoscience Australia's online API.

import numpy as np
import quagmire
from quagmire import QuagMesh
from quagmire import function as fn
from quagmire import tools as meshtools

from scipy.interpolate import RegularGridInterpolator

import h5py
from mpi4py import MPI
comm = MPI.COMM_WORLD

%pylab inline

data_dir = "./data/"
etopo_filename = data_dir+'ETOPO1_Ice_g.h5'

extent_australia = [112, 155, -44, -10]
lonmin, lonmax, latmin, latmax = extent_australia

mlons, mlats, bmask = meshtools.generate_square_points(lonmin, lonmax, latmin, latmax, 0.1, 0.1, 15000, 800)

DM = meshtools.create_DMPlex_from_spherical_points(mlons, mlats, bmask,refinement_levels=3)
mesh = QuagMesh(DM, downhill_neighbours=2, verbose=True)

print("number of points in mesh: ", mesh.npoints)

## Read topography from HDF5

# local extent
local_extent = [mesh.coords[:,0].min(), mesh.coords[:,0].max(), mesh.coords[:,1].min(), mesh.coords[:,1].max()]

with h5py.File(etopo_filename, 'r', driver='mpio', comm=comm) as h5:
    h5_lons = h5['lons'][:]
    h5_lats = h5['lats'][:]
    
    xbuffer = np.diff(h5_lons).mean()
    ybuffer = np.diff(h5_lats).mean()
    
    i0 = np.abs(h5_lons - (local_extent[0] - xbuffer)).argmin()
    i1 = np.abs(h5_lons - (local_extent[1] + xbuffer)).argmin() + 1
    j0 = np.abs(h5_lats - (local_extent[2] - ybuffer)).argmin()
    j1 = np.abs(h5_lats - (local_extent[3] + ybuffer)).argmin() + 1
    
    aus_dem = h5['data'][j0:j1,i0:i1]


# map DEM to local mesh
interp = RegularGridInterpolator((h5_lats[j0:j1], h5_lons[i0:i1]), aus_dem, bounds_error=True)
height = interp(mesh.coords[:,::-1])

mesh.mask.unlock()
mesh.mask.data = height > 0.0
mesh.mask.lock()
mesh.mask.sync()

mesh.bmask = height > 0.0

with mesh.deform_topography():
    mesh.topography.data = height

for repeat in range(0,3): 
    
    mesh.low_points_local_patch_fill(its=3, smoothing_steps=3)
    low_points2 = mesh.identify_global_low_points(ref_height=0.0)
    if low_points2[0] <= 1:
        break

    for i in range(0,20):

        mesh.low_points_swamp_fill(ref_height=0.0, ref_gradient=1e-24)

        # In parallel, we can't break if ANY processor has work to do (barrier / sync issue)
        low_points3 = mesh.identify_global_low_points(ref_height=0.0)

        print("{} : {}".format(i,low_points3[0]))
        if low_points3[0] <= 1:
            break

outflow_points = mesh.identify_outflow_points()
low_points     = mesh.identify_low_points()

# normalise height on [0, 1]
norm_height = mesh.topography.data[:].copy()
norm_height -= norm_height.min()
norm_height /= norm_height.max()

# modify the vertical exaggeration
norm_height /= 25
norm_height += 1.0

ones = mesh.add_variable("ones")
ones.data = 1.0
cumulative_flow_1 = mesh.upstream_integral_fn(ones).evaluate(mesh)

cumulative_flow_1 *= mesh.bmask
logflow1 = np.log10(1e-10 + cumulative_flow_1)
logflow1[logflow1 < 0] = 0

import lavavu
import stripy

vertices = mesh.data
tri = mesh.tri

lv = lavavu.Viewer(border=False, axis=False, background="#FFFFFF", resolution=[1200,600], near=-10.0)

outs = lv.points("outflows", colour="green", pointsize=5.0, opacity=0.75)
outs.vertices(vertices[outflow_points])

lows = lv.points("lows", colour="red", pointsize=5.0, opacity=0.75)
lows.vertices(vertices[low_points])

flowball = lv.points("flowballs", pointsize=2.0)
flowball.vertices(vertices*1.01)
flowball.values(logflow1, label="flow1")
flowball.colourmap("rgba(255,255,255,0.0) rgba(128,128,255,0.1) rgba(25,100,225,0.2) rgba(0,50,200,0.5)")

heightball = lv.points("heightballs", pointsize=5.0, opacity=1.0)
heightball.vertices(vertices)
heightball.values(height, label="height")
heightball.colourmap('dem3')

# lv.translation(-1.012, 2.245, -13.352)
# lv.rotation(53.217, 18.104, 161.927)

lv.control.Panel()
lv.control.ObjectList()
lv.control.show()


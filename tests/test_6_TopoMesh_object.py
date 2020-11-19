import pytest
import numpy as np
import quagmire
from quagmire import function as fn
from quagmire import QuagMesh
from petsc4py import PETSc

from conftest import load_multi_mesh_DM as DM


def test_height_mesh_variable(DM):
    mesh = QuagMesh(DM)
    x, y = mesh.coords[:,0], mesh.coords[:,1]

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    with mesh.deform_topography():
        print("Update topography data array (automatically rebuilds matrices)")
        mesh.topography.data = height

    # make sure the adjacency matrices are updated
    mat_info = mesh.downhillMat.getInfo()
    mat_size = mesh.downhillMat.getSize()

    assert mat_info['nz_used'] >= mat_size[0], "{}: Downhill matrix is not initialised correctly\n{}".format(mesh.id, mat_info)


def test_downhill_neighbours(DM):
    mesh = QuagMesh(DM, downhill_neighbours=1)
    x, y = mesh.coords[:,0], mesh.coords[:,1]

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    nz = []

    # increase number of downhill neighbours from 1-3
    for i in range(1,4):
        mesh = QuagMesh(DM, downhill_neighbours=i)

        with mesh.deform_topography():
            mesh.topography.data = height

        mat_info = mesh.downhillMat.getInfo()
        nz.append(mat_info['nz_used'])

    mat_size = mesh.downhillMat.getSize()
    ascending = (np.diff(np.array(nz) - mat_size[0]) > 0).all()

    err_msg = "{}: Downhill matrix is not denser with more downhill neighbours: {}"
    assert ascending, err_msg.format(mesh.id, list(enumerate(nz)))


def test_cumulative_flow(DM):

    def identify_outflow_points(self):
        """
        Identify the (boundary) outflow points and return an array of (local) node indices
        """

        # nodes = np.arange(0, self.npoints, dtype=np.int)
        # low_nodes = self.down_neighbour[1]
        # mask = np.logical_and(nodes == low_nodes, self.bmask == False)
        #

        i = self.downhill_neighbours

        o = (np.logical_and(self.down_neighbour[i] == np.indices(self.down_neighbour[i].shape), self.bmask == False)).ravel()
        outflow_nodes = o.nonzero()[0]

        return outflow_nodes

    mesh = QuagMesh(DM)

    x, y = mesh.coords[:,0], mesh.coords[:,1]

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    with mesh.deform_topography():
        mesh.topography.data = height

    # constant rainfall everywhere
    rainfall = mesh.add_variable(name='rainfall')
    rainfall.data = np.ones(mesh.npoints)

    upstream_precipitation_integral_fn = mesh.upstream_integral_fn(rainfall)
    upstream_rain = upstream_precipitation_integral_fn.evaluate(mesh)

    # compare cumulative rainfall at the outflow nodes to the rest of the domain
    outflow_indices = identify_outflow_points(mesh)

    err_msg = "{}: cumulative rain outflow is less than the mean".format(mesh.id)
    assert upstream_rain[outflow_indices].mean() > upstream_rain.mean(), err_msg


def test_streamwise_smoothing(DM):
    mesh = QuagMesh(DM)

    x, y = mesh.coords[:,0], mesh.coords[:,1]

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    with mesh.deform_topography():
        mesh.topography.data = height

    # find streams
    mask_streams = mesh.upstream_area.data > 0.001
    stream_x = x[mask_streams]
    stream_y = y[mask_streams]

    # seed random rainfall
    rainfall = mesh.add_variable("rainfall")
    rainfall.data = np.random.random(size=mesh.npoints)

    stream = np.ndarray((stream_x.size, 2))
    stream[:, 0] = stream_x
    stream[:, 1] = stream_y
    sm1 = np.std(rainfall.evaluate(stream))

    for i in range(1,4):
        sm0 = float(sm1)
        smooth_fn = mesh.streamwise_smoothing_fn(rainfall, its=i)
        sm1 = np.std(smooth_fn.evaluate(stream))

        assert sm1 < sm0, "{}: streamwise smoothing at its={} no smoother than its={}".format(mesh.id, i, i-1)


def test_swamp_fill(DM):
    mesh = QuagMesh(DM, downhill_neighbours=2)

    x, y = mesh.coords[:,0], mesh.coords[:,1]

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    # take a chunk out of the landscape centred on this point
    xpt, ypt = 1.5, 1.5

    # find the closest index on the mesh
    index = np.abs(mesh.coords - [xpt, ypt]).sum(axis=1).argmin()
    d, idx = mesh.cKDTree.query(mesh.data[index], k=30)

    height[idx] = height[idx].min()

    with mesh.deform_topography():
        mesh.topography.data = height

    # slope at idx should be zero everywhere, and non-zero where the landscape is repaired.
    # this does not happen in practise because there is a high slope at the boundary.

    slope0 = mesh.slope.evaluate(mesh)

    mesh.low_points_swamp_fill(ref_height=0.0)

    slope1 = mesh.slope.evaluate(mesh)

    assert slope1[idx].mean() > slope0[idx].mean(), "{}: swamp fill has not filled a pit in the side of a hill".format(mesh.id)

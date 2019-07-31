import pytest
import numpy as np
import quagmire
from quagmire import function as fn
from quagmire import TopoMesh
from petsc4py import PETSc

from conftest import load_triangulated_mesh_DM


def test_height_mesh_variable(load_triangulated_mesh_DM):
    mesh = TopoMesh(load_triangulated_mesh_DM)
    x, y = mesh.tri.x, mesh.tri.y

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

    assert mat_info['nz_used'] >= mat_size[0], "Downhill matrix is not initialised correctly\n{}".format(mat_info)


def test_downhill_neighbours(load_triangulated_mesh_DM):
    mesh = TopoMesh(load_triangulated_mesh_DM, downhill_neighbours=1)
    x, y = mesh.tri.x, mesh.tri.y

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    nz = []

    # increase number of downhill neighbours from 1-3
    for i in range(1,4):
        mesh = TopoMesh(load_triangulated_mesh_DM, downhill_neighbours=i)

        with mesh.deform_topography():
            mesh.topography.data = height

        mat_info = mesh.downhillMat.getInfo()
        nz.append(mat_info['nz_used'])

    mat_size = mesh.downhillMat.getSize()
    ascending = (np.diff(np.array(nz) - mat_size[0]) > 0).all()

    err_msg = "Downhill matrix is not denser with more downhill neighbours: {}"
    assert ascending, err_msg.format(list(enumerate(nz)))


def test_cumulative_flow(load_triangulated_mesh_DM):

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

    mesh = TopoMesh(load_triangulated_mesh_DM)

    x, y = mesh.tri.x, mesh.tri.y

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

    err_msg = "cumulative rain outflow is less than the mean"
    assert upstream_rain[outflow_indices].mean() > upstream_rain.mean(), err_msg


def test_streamwise_smoothing(load_triangulated_mesh_DM):
    mesh = TopoMesh(load_triangulated_mesh_DM)

    x, y = mesh.tri.x, mesh.tri.y

    radius  = np.sqrt((x**2 + y**2))
    theta   = np.arctan2(y,x) + 0.1

    height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
    height  += 0.5 * (1.0-0.2*radius)

    with mesh.deform_topography():
        mesh.topography.data = height

    # find steeper regions
    flat_area_mask = fn.misc.levelset(mesh.slope, 0.1, invert=True)

    # seed rainfall
    rainfall = mesh.add_variable("rainfall")
    rainfall.data = flat_area_mask.evaluate(mesh)

    sm1 = np.std(rainfall.data)

    for i in range(1,4):
        sm0 = float(sm1)
        smooth_fn = mesh.streamwise_smoothing_fn(rainfall, its=i)
        sm1 = np.std(smooth_fn.evaluate(mesh))

        assert sm1 < sm0, "streamwise smoothing at its={} no smoother than its={}".format(i, i-1)
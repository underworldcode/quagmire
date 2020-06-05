
import pytest
import numpy as np
import quagmire
from quagmire import QuagMesh
from quagmire import function as fn
from quagmire.tools import meshtools

from conftest import load_triangulated_mesh_DM


def test_derivatives_and_interpolation(load_triangulated_mesh_DM):
    mesh = QuagMesh(load_triangulated_mesh_DM)

    height = mesh.add_variable(name="h(x,y)")
    height.data = mesh.tri.x**2 + mesh.tri.y**2

    dhdx = height.fn_gradient[0]
    dhdy = height.fn_gradient[1]

    # interpolate onto a straight line
    # bounding box should be [-5,5,-5,5]
    xy_pts = np.linspace(-5,5,10)

    interp_x = dhdx.evaluate(xy_pts, np.zeros_like(xy_pts))
    interp_y = dhdy.evaluate(np.zeros_like(xy_pts), xy_pts)

    ascending_x = ( np.diff(interp_x) > 0 ).all()
    ascending_y = ( np.diff(interp_y) > 0 ).all()

    assert ascending_x, "Derivative evaluation failed in the x direction"
    assert ascending_y, "Derivative evaluation failed in the y direction"


def test_smoothing(load_triangulated_mesh_DM):
    mesh = QuagMesh(load_triangulated_mesh_DM)

    noise = np.random.rand(mesh.npoints)
    smooth_noise = mesh.local_area_smoothing(noise, its=1)
    smoother_noise = mesh.local_area_smoothing(noise, its=2)

    err_msg = "Smoothing random noise using RBF is not working"
    assert np.std(noise) > np.std(smooth_noise) > np.std(smoother_noise), err_msg


def test_spherical_area():
    import stripy
    cm = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=2)
    DM = meshtools.create_spherical_DMPlex(np.degrees(cm.lons), np.degrees(cm.lats), cm.simplices)
    mesh = QuagMesh(DM)

    # adjust radius of the sphere
    # this should re-calculate pointwise area and weights
    R1 = 1.0
    R2 = 2.0

    mesh.radius = R1
    area1 = mesh.pointwise_area.sum()
    assert np.isclose(area1,4.*np.pi), "Area of the unit-sphere is incorrect, {}".format(area1)

    mesh.radius = R2
    area2 = mesh.pointwise_area.sum()
    assert np.isclose(area2,4.*np.pi*R2**2), "Area of sphere with r=2 is incorrect, {}".format(area2)
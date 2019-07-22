
import pytest
import numpy as np
import quagmire
from quagmire import FlatMesh
from quagmire import function as fn

from conftest import load_triangulated_mesh_DM


def test_derivatives_and_interpolation(load_triangulated_mesh_DM):
    mesh = FlatMesh(load_triangulated_mesh_DM)

    height = mesh.add_variable(name="h(x,y)")
    height.data = mesh.tri.x**2 + mesh.tri.y**2

    dhdx, dhdy = fn.math.grad(height)

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
    mesh = FlatMesh(load_triangulated_mesh_DM)

    noise = np.random.rand(mesh.npoints)
    smooth_noise = mesh.local_area_smoothing(noise.data, its=1)
    smoother_noise = mesh.local_area_smoothing(noise.data, its=2)

    err_msg = "Smoothing random noise using RBF is not working"
    assert np.std(noise) > np.std(smooth_noise) > np.std(smoother_noise), err_msg

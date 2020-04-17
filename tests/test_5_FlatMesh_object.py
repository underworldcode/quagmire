
import pytest
import numpy as np
import quagmire
from quagmire import FlatMesh
from quagmire import function as fn
from quagmire.tools import meshtools

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
    smooth_noise = mesh.local_area_smoothing(noise, its=1)
    smoother_noise = mesh.local_area_smoothing(noise, its=2)

    err_msg = "Smoothing random noise using RBF is not working"
    assert np.std(noise) > np.std(smooth_noise) > np.std(smoother_noise), err_msg


def test_spherical_area():
    import stripy
    cm = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=3)
    fm = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=4)

    DM_c = meshtools.create_spherical_DMPlex(cm.lons, cm.lats, cm.simplices)
    DM_f = meshtools.create_spherical_DMPlex(fm.lons, fm.lats, fm.simplices)

    mesh_c = FlatMesh(DM_c)
    mesh_f = FlatMesh(DM_f)

    area_of_earth = 510064472e6 # metres^2
    
    # re-calculate area
    mesh_c.calculate_area_weights(r1=6384.4e3, r2=6352.8e3)
    mesh_f.calculate_area_weights(r1=6384.4e3, r2=6352.8e3)

    area1 = mesh_c.area.sum() # metres^2
    area2 = mesh_f.area.sum() # metres^2

    err_msg = "Area of the earth is getting worse with refinement"
    assert abs(area_of_earth - area2) < abs(area_of_earth - area1), err_msg
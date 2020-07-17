
import pytest
import numpy as np
import numpy.testing as npt
import h5py
import quagmire
import petsc4py
from quagmire.tools import io
import quagmire.tools.cloud

from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable


dropbox_mesh_url = "https://www.dropbox.com/s/iih150l1jtm0fp6/Ex1a-circular_mesh.h5?dl=0"
dropbox_phi_url  = "https://www.dropbox.com/s/nekzkmvx5h82r2l/Ex1a-circular_mesh_phi.h5?dl=0"
dropbox_psi_url  = "https://www.dropbox.com/s/5dzujlo3ayo5s35/Ex1a-circular_mesh_psi.h5?dl=0"

google_drive_mesh_gurl = "https://drive.google.com/file/d/1zY-4XDUyUXq7So2bc7MWoiD8-F3ZdZCg/view?usp=sharing"
google_drive_phi_gurl  = "https://drive.google.com/file/d/1n-zdsKLNLtI9-ZOP5dMVTi_VEn0YYKJC/view?usp=sharing"
google_drive_psi_gurl  = "https://drive.google.com/file/d/17t8jbPFmnB8aHhyYDbxjGrzKqtq6IlCa/view?usp=sharing"



def test_dropbox_url():

    from quagmire.tools.cloud import quagmire_cloud_fs


    DMC = meshtools.create_DMPlex_from_url(dropbox_mesh_url)
    meshC = QuagMesh(DMC)

    phi = meshC.add_variable(name="PHI(X,Y)")
    psi = meshC.add_variable(name="PSI(X,Y)")

    assert( meshC.npoints == 47361)
    assert( np.fabs(meshC.area[0] - 0.00110479) < 1.0e-8)

    phi.load_from_url(dropbox_phi_url)
    psi.load_from_url(dropbox_psi_url)

    assert( np.fabs(phi.data[0] - 0.0649746471444294) < 1.0e-8)
    assert( np.fabs(psi.data[0] - 0.9016719734345492) < 1.0e-8)


    return


def test_google_drive_url():

    from quagmire.tools.cloud import quagmire_cloud_fs, google_drive_convert_link

    google_drive_mesh_url = google_drive_convert_link(google_drive_mesh_gurl)
    google_drive_phi_url  = google_drive_convert_link(google_drive_phi_gurl)
    google_drive_psi_url  = google_drive_convert_link(google_drive_psi_gurl)

    DMC = meshtools.create_DMPlex_from_url(google_drive_mesh_url)
    meshC = QuagMesh(DMC)

    phi = meshC.add_variable(name="PHI(X,Y)")
    psi = meshC.add_variable(name="PSI(X,Y)")

    assert( meshC.npoints == 47361)
    assert( np.fabs(meshC.area[0] - 0.00110479) < 1.0e-8)

    phi.load_from_url(google_drive_phi_url)
    psi.load_from_url(google_drive_psi_url)

    assert( np.fabs(phi.data[0] - 0.0649746471444294) < 1.0e-8)
    assert( np.fabs(psi.data[0] - 0.9016719734345492) < 1.0e-8)


    return

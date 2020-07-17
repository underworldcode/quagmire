
import pytest
import numpy as np
import numpy.testing as npt
import h5py
import quagmire
import petsc4py
from quagmire.tools import io
import quagmire.tools.cloud

from mpi4py import MPI
comm = MPI.COMM_WORLD

from conftest import load_triangulated_mesh_DM
from conftest import load_triangulated_spherical_mesh_DM
from conftest import load_pixelated_mesh_DM

from quagmire.tools import cloud_fs 

pyfilesystem = pytest.importorskip("fs")



def test_fs_access():
    """Verify that fs can connect to the quagmire cloud"""

    from quagmire.tools.cloud import verify_cloud_fs, quagmire_cloud_fs

    verify_cloud_fs(quagmire_cloud_fs)
    return

def test_fs_list_qcloud():
    """Obtain a directory listing and check the existence of quagmire_cloud_info.txt"""

    from quagmire.tools.cloud import quagmire_cloud_fs

    lsd = quagmire_cloud_fs.listdir("/")
    assert 'quagmire_cloud_info.txt' in lsd

    return

def test_file_dload_qcloud():

    from quagmire.tools.cloud import quagmire_cloud_fs, cloud_download

    # Needs to be in a tmp file ... don't want to pollute the repo
    # cloud_download("quagmire_cloud_info.txt", "test.txt")

    import tempfile

    f = tempfile.NamedTemporaryFile(delete=True)

    cloud_download("quagmire_cloud_info.txt", f.name)

    with open(f.name, 'r') as fp:
        content = fp.read()

    assert content.find("4SEAhkqSlTojYhv")

    return

def test_read_only_qcloud():

    import tempfile
    from quagmire.tools.cloud import quagmire_cloud_fs, cloud_upload

    # This should be true but is not reported so by cloudstor
    # assert quagmire_cloud_fs.getmeta()["read_only"]

    f = tempfile.NamedTemporaryFile(delete=True)
    with open(f.name, 'w') as fp:
        print("This file should not exist", file=fp)

    cloud_upload("written_to_read_only_fs.txt", f.name, quagmire_cloud_fs)

    lsd = quagmire_cloud_fs.listdir("/")

    assert 'written_to_read_only_fs.txt' not in lsd
 

def test_file_drop_qcloud():

    from quagmire.tools.cloud import quagmire_cloud_filedrop_fs, cloud_upload

    # Needs to be in a tmp file ... don't want to pollute the repo
    # cloud_download("quagmire_cloud_info.txt", "test.txt")

    import uuid
    import tempfile
    import datetime

    unique_filename_in_the_cloud = "test_" + str(uuid.uuid4().hex) + ".txt"

    f = tempfile.NamedTemporaryFile(delete=True)

    with open(f.name,"w") as fp:
        print("This file was created at {} ".format(datetime.datetime.now()), file=fp)

    cloud_upload(unique_filename_in_the_cloud, f.name,  cloud_fs_location_handle=quagmire_cloud_filedrop_fs)

    # There is no way to check if the file landed ok !

    return



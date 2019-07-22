
import pytest

def test_numpy_import():
    import numpy

def test_scipy_import():
    import scipy
    print("\t\t You have scipy version {}".format(scipy.__version__))

def test_mpi4py_import():
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	assert comm.rank is int, "mpi4py is not configured properly"

def test_h5py_import():
	import h5py

def test_petsc4py_import():
	from petsc4py import PETSc
	# should get a list of required external packages e.g. triangle

def test_stripy_modules():
    import stripy
    from stripy import documentation
    from stripy import spherical_meshes
    from stripy import cartesian_meshes
    from stripy import sTriangulation
    from stripy import Triangulation

def test_quagmire_modules():
	import quagmire
	from quagmire import documentation
	from quagmire import function
	from quagmire import mesh
	from quagmire import tools
	from quagmire import scaling
	from quagmire import FlatMesh
	from quagmire import TopoMesh
	from quagmire import SurfaceProcessMesh

def test_jupyter_available():
    from subprocess import check_output
    try:
        result = str(check_output(['which', 'jupyter']))[2:-3]
    except:
        assert False, "jupyter notebook system is not installed"

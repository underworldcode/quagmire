# -*- coding: utf-8 -*-

import pytest

# ==========================

def test_numpy_import():
    import numpy
    return

def test_scipy_import():
    import scipy
    print("\t\t You have scipy version {}".format(scipy.__version__))
    return

def test_sympy_import():
    import sympy
    print("\t\t You have sympy version {}".format(sympy.__version__))
    return

def test_pint_import():
    import pint
    return

def test_stripy_import():
    import stripy
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
	from quagmire import _fortran

def test_jupyter_available():
    from subprocess import check_output
    try:
        result = str(check_output(['which', 'jupyter']))[2:-3]
    except:
        print( "jupyter notebook system is not installed" )
        print( "  - This is needed for notebook examples")

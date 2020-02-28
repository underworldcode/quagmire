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
    return

def test_quagmire_modules():
    import quagmire

    from quagmire import documentation
    from quagmire import mesh
    from quagmire import scaling
    from quagmire import surfmesh
    from quagmire import topomesh
    from quagmire import tools
    from quagmire import function


def test_jupyter_available():
    from subprocess import check_output
    try:
        result = str(check_output(['which', 'jupyter']))[2:-3]
    except:
        assert False, "jupyter notebook system is not installed"

# def test_documentation_dependencies():
#     import matplotlib
#     import cartopy
#     import imageio
#     import lavavu
#     import pyproj
#
# def test_litho1pt0():
#     import litho1pt0
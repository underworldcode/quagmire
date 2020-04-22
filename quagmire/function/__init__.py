from .function_classes import *
from . import math
from . import misc
from . import stats


def check_dependency(this_fn, that_fn):
    """Does this_fn depend upon that_fn ?"""

    return id(that_fn) in this_fn.dependency_list

def check_object_is_a_q_function(fn_object):
    """Is this object a quagmire LazyEvaluation function ?"""

    return isinstance(fn_object, LazyEvaluation)

def check_object_is_a_q_function_and_raise(fn_object):
    """If this object is not a quagmire LazyEvaluation function then everything must die !"""

    if not isinstance(fn_object, LazyEvaluation):
        raise RuntimeError("Expecting a quagmire.function object")

def check_object_is_a_mesh_variable(fn_object):
	""" Is this object a quagmire MeshVariable or VectorMeshVariable ?"""

	return fn_object.mesh_data
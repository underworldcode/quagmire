# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
# 
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.

"""
Create mesh variables that interface with functions

Initialise a `MeshVariable` for scalar fields or `VectorMeshVariable` for vectors
on the mesh. These inherit the `quagmire.function` classes that exploit lazy evaluation.
"""

try: range = xrange
except: pass


## These also need to be lazy evaluation objects

from ..function import LazyEvaluation as _LazyEvaluation
from ..function import x_0 as _x_0
from ..function import x_1 as _x_1


import numpy as np

class MeshFunction(_LazyEvaluation):
    """
    A class of Lazy Evaluation Functions that require an underlying mesh.

    Parameters
    ----------
    name : str
        Assign the MeshVariable a unique identifier
    mesh : quagmire mesh object
        The supporting mesh for the variable
    """
    
    def __init__(self, name=None, mesh=None, lname=None):
        super(MeshFunction, self).__init__()

        self._mesh = mesh
        self._dm = mesh.dm
        self._name = str(name)
        self.description = self._name
        self.mesh_data = False

        if lname is not None:
            self.latex = lname+r"\left({},{}\right)".format(_x_0.latex, _x_1.latex)
        else:
            self.latex = self.description

        return


    def rename(self, name=None, lname=None):
        """
        Rename this MeshVariable
        """

        if name is None:
            return

        self._name = str(name)
        self._ldata.setName(name)
        self.description = self._name

        if lname is not None:
            self.latex = lname+r"\left({},{}\right)".format(_x_0.latex, _x_1.latex)
        else:
            self.latex = self.description


    def evaluate(self, input=None, **kwargs):
        """
        If the argument is a mesh, return the values at the nodes.
        In all other cases call the `interpolate` method.

        But note the way that interpolate calls this method with the mesh to 
        get values at the mesh nodes - this needs to be implemented correctly or 
        the interpolate method needs to be over-ridden as well.
        """

        ## This should be NotImplemented ... evaluate is to be defined 
        ## for any subclass

        import quagmire

        ## this should be replaced by an operation that returns the
        ## np.array of the correct dimension

        array = np.zeros(self._mesh.npoints)

        if input is not None:
            if isinstance(input, (quagmire.mesh.trimesh.TriMesh, 
                                  quagmire.mesh.pixmesh.PixMesh,
                                  quagmire.mesh.strimesh.sTriMesh)):
                if input == self._mesh:
                    return array
                else:
                    return self.interpolate(input.coords[:,0], input.coords[:,1], **kwargs)

            elif isinstance(input, (tuple, list, np.ndarray)):
                input = np.array(input)
                input = np.reshape(input, (-1, 2))
                return self.interpolate(input[:, 0], input[:,1], **kwargs)
        else:
            return array


    def interpolate(self, xi, yi, err=False, **kwargs):
        """
        Interpolate function to a set of coordinates

        This method just passes the coordinates to the interpolation
        methods on the mesh object.

        Parameters
        ----------
        xi : array of floats, shape (l,)
            interpolation coordinates in the x direction
        yi : array of floats, shape (l,)
            interpolation coordinates in the y direction
        err : bool (default: False)
            toggle whether to return error information
        **kwargs : keyword arguments
            optional arguments to pass through to the interpolation method

        Returns
        -------
        interp : array of floats, shape (l,)
            interpolated values of MeshVariable at xi,yi coordinates
        err : array of ints, shape (l,)
            error information to diagnose interpolation / extrapolation
        """
        ## pass through for the mesh's interpolate method
        import numpy as np

        mesh = self._mesh
        PHI = self.evaluate(input=self._mesh)

        xi_array = np.array(xi).reshape(-1,1)
        yi_array = np.array(yi).reshape(-1,1)

        i, e = mesh.interpolate(xi_array, yi_array, zdata=PHI, **kwargs)

        if err:
            return i, e
        else:
            return i

    def derivative(self, dirn):
        return self.fn_gradient(dirn)

    def fn_slope(self):
        return self.fn_gradient(-1)

    ## No appreciable acceleration for grad by itself, or div 


    def fn_gradient(self, dirn=None):
        """
        The generic mechanism for obtaining the gradient of a lazy variable is
        to evaluate the values on the mesh at the time in question and use the mesh gradient
        operators to compute the new values.
        Sub classes may have more efficient approaches. MeshVariables have
        stored data and don't need to evaluate values. Parameters have Gradients
        that are identically zero ... etc
        """

        import quagmire

        diff_mesh = self._mesh

        def new_fn_x(*args, **kwargs):
            local_array = self.evaluate(diff_mesh)
            dxy = diff_mesh.derivative_grad(local_array, nit=10, tol=1e-8)

            if len(args) == 1 and args[0] is diff_mesh:
                return dxy[:,0]

            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
                mesh = args[0]
                return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=dxy[:,0], **kwargs)
            else:
                coords = np.array(args).reshape(-1,2)
                i, ierr = diff_mesh.interpolate(coords[:,0], coords[:,1], zdata=dxy[:,0])
                return i

        def new_fn_y(*args, **kwargs):
            local_array = self.evaluate(diff_mesh)
            dxy = diff_mesh.derivative_grad(local_array, nit=10, tol=1e-8)

            if len(args) == 1 and args[0] is diff_mesh:
                return dxy[:,1]
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
                mesh = args[0]
                return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=dxy[:,1], **kwargs)
            else:
                coords = np.array(args).reshape(-1,2)
                i, ierr = diff_mesh.interpolate(coords[:,0], coords[:,1], zdata=dxy[:,1])
                return i

        def new_fn_slope(*args, **kwargs):
            local_array = self.evaluate(diff_mesh)
            dxy = diff_mesh.derivative_grad(local_array, nit=10, tol=1e-8)

            if len(args) == 1 and args[0] is diff_mesh:
                return np.hypot(dxy[:,0], dxy[:,1])
            elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
                mesh = args[0]
                return diff_mesh.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=np.hypot(dxy[:,0], dxy[:,1]), **kwargs)
            else:
                coords = np.array(args).reshape(-1,2)
                i, ierr = diff_mesh.interpolate(coords[:,0], coords[:,1], zdata=np.hypot(dxy[:,0], dxy[:,1]))
                return i

        newLazyFn_dx = MeshFunction(name="ddx-"+self._name, mesh=self._mesh)
        newLazyFn_dx.evaluate = new_fn_x
        newLazyFn_dx.description = "d({})/dX".format(self.description)
        newLazyFn_dx.latex = r"\frac{{ \partial }}{{\partial x}}{}".format(self.latex)
        newLazyFn_dx.exposed_operator = "d"

        newLazyFn_dy = MeshFunction(name="ddy-"+self._name, mesh=self._mesh)
        newLazyFn_dy.evaluate = new_fn_y
        newLazyFn_dy.description = "d({})/dY".format(self.description)
        newLazyFn_dy.latex = r"\frac{{\partial}}{{\partial y}}{}".format(self.latex)
        newLazyFn_dy.exposed_operator = "d"

        newLazyFn_slope = MeshFunction(name="slope-"+self._name, mesh=self._mesh)
        newLazyFn_slope.evaluate = new_fn_slope
        newLazyFn_slope.description = "slope({})".format(self.description)
        newLazyFn_slope.latex = r"\left| \nabla {} \right|".format(self.latex)
        newLazyFn_slope.exposed_operator = "S"


        if dirn == 0:
            return newLazyFn_dx
        elif dirn == 1:
            return newLazyFn_dy
        else:
            return newLazyFn_slope


class MeshVariable(MeshFunction):
    """
    The MeshVariable class generates a variable supported on the mesh.

    To set/read nodal values, use the numpy interface via the 'self.data' property.

    Parameters
    ----------
    name : str
        Assign the MeshVariable a unique identifier
    mesh : quagmire mesh object
        The supporting mesh for the variable
    """
    def __init__(self, name=None, mesh=None, lname=None, locked=False):
        super(MeshVariable, self).__init__(name=name, mesh=mesh, lname=lname)

        self._locked = locked
        self.mesh_data = True

        # mesh variable vector
        self._ldata = self._dm.createLocalVector()
        self._ldata.setName(name)
        return

    def __repr__(self):
        return "quagmire.MeshFunction: {}".format(self.description)
    

    def copy(self, name=None, locked=None):
        """
        Create a copy of this MeshVariable

        Parameters
        ----------
        name : str
            set a name to this MeshVariable, otherwise "copy"
            is appended to the original name
        locked : bool (default: False)
            lock the mesh variable from accidental modification

        Returns
        -------
        MeshVariable : object
            Instantiate a MeshVariable copy.
        """

        if name is None:
            name = self._name+"_copy"

        if locked is None:
            locked = self._locked

        new_mesh_variable = MeshVariable(name=name, mesh=self._mesh, locked=False)
        new_mesh_variable.data = self.data

        if locked:
            new_mesh_variable.lock()

        return new_mesh_variable

    def lock(self):
        self._locked = True

    def unlock(self):
        self._locked = False

## This is a redundancy - @property definition is nuked by the @ .getter
## LM: See this: https://stackoverflow.com/questions/51244348/use-of-propertys-getter-in-python

## Don't sync on get / set as this prevents doing a series of computations on the array and
## doing the sync when finished. I can also imagine this going wrong if sync nukes values
## in the shadow zone unexpectedly. Also get is called for any indexing operation ...  ugh !

    @property
    def data(self):
        """ View of MeshVariable data of shape (n,) """
        pass

    @data.getter
    def data(self):
        # This step is necessary because the numpy array is writeable
        # through to the petsc vector and this cannot be blocked.
        # Access to the numpy array will not automatically be sync'd and this
        # would also be a way to circumvent access to locked arrays - where such
        # locking is intended to ensure we update dependent data when the variable is
        # updated

# if self._locked:
        view = self._ldata.array[:]
        view.setflags(write=False)
        return view

#        else:
#            return self._ldata.array

    @data.setter
    def data(self, val):
        if self._locked:
            import quagmire
            if quagmire.mpi_rank == 0:
                print("quagmire.MeshVariable: {} - is locked".format(self.description))
            return

        if type(val) is float:
                self._ldata.set(val)
        else:
            self._ldata.setArray(val)



    ## For printing and other introspection we actually want to look through to the
    ## meshVariable's own description

    def __repr__(self):
        if self._locked:
            return "quagmire.MeshVariable: {} - RO".format(self.description)
        else:
            return "quagmire.MeshVariable: {} - RW".format(self.description)


    def getGlobalVector(self, gdata):
        """
        Obtain a PETSc global vector of the MeshVariable
        """
        from petsc4py import PETSc
        self._dm.localToGlobal(self._ldata, gdata, addv=PETSc.InsertMode.INSERT_VALUES)
        return


    def sync(self, mergeShadow=False):
        """
        Explicit global sync of data
        """

        from petsc4py import PETSc

        if mergeShadow:
            addv = PETSc.InsertMode.ADD_VALUES
        else:
            addv = PETSc.InsertMode.INSERT_VALUES

        # need a global vector
        gdata = self._dm.getGlobalVec()

        # self._dm.localToLocal(self._ldata, self._gdata)
        self._dm.localToGlobal(self._ldata, gdata, addv=addv)
        self._dm.globalToLocal(gdata, self._ldata)

        self._dm.restoreGlobalVec(gdata)

        return


    def xdmf(self, hdf5_filename, hdf5_mesh_filename=None, xdmf_filename=None):
        """
        Creates an xdmf file associating the saved HDF5 file with the mesh

        """
        from quagmire.tools.generate_xdmf import generateXdmf
        from quagmire import mpi_rank

        hdf5_filename = str(hdf5_filename)
        if not hdf5_filename.endswith('.h5'):
            hdf5_filename += '.h5'

        if mpi_rank == 0:
            generateXdmf(hdf5_filename, hdf5_mesh_filename, xdmf_filename)
        return


    def save(self, hdf5_filename, append=True):
        """
        Save the MeshVariable to disk.

        Parameters
        ----------
        hdf5_filename : str (optional)
            The name of the output file. Relative or absolute paths may be
            used, but all directories must exist.
        append : bool (default is True)
            Append to existing file if it exists

        Notes
        -----
        This method must be called collectively by all processes.
        """
        from petsc4py import PETSc
        import os

        hdf5_filename = str(hdf5_filename)
        if not hdf5_filename.endswith('.h5'):
            hdf5_filename += '.h5'

        if self._mesh.id.startswith("pixmesh"):
            # save mesh info to the hdf5 if DMDA object
            # this is so tiny that I/O hit is negligible
            self._mesh.save_mesh_to_hdf5(hdf5_filename)

        mode = "w"
        if append and os.path.exists(hdf5_filename):
            mode = "a"

        # need a global vector
        vname = self._ldata.getName()
        gdata = self._dm.getGlobalVec()
        gdata.setName(vname)
        self._dm.localToGlobal(self._ldata, gdata)

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(hdf5_filename, mode=mode)
        ViewHDF5(gdata)
        ViewHDF5.destroy()
        ViewHDF5 = None

        self._dm.restoreGlobalVec(gdata)

        return

    def load(self, hdf5_filename, name=None):
        """
        Load the MeshVariable from disk.

        Parameters
        ----------
        hdf5_filename: str
            The filename for the saved file. Relative or absolute paths may be
            used, but all directories must exist.

        Notes
        -----
        Provided files must be in hdf5 format, and contain a vector the same
        size and with the same name as the current MeshVariable
        """
        if self._locked:
            raise ValueError("quagmire.MeshVariable: {} - is locked".format(self.description))

        from petsc4py import PETSc
        # need a global vector
        gdata = self._dm.getGlobalVec()

        if name is None:
            gdata.setName(self._ldata.getName())
        else:
            gdata.setName(str(name))

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(str(hdf5_filename), mode='r')
        gdata.load(ViewHDF5)
        ViewHDF5.destroy()
        ViewHDF5 = None

        self._dm.globalToLocal(gdata, self._ldata)
        self._dm.restoreGlobalVec(gdata)


    def load_from_cloud_fs(self, cloud_hdf5_filename, cloud_location_handle=None, name=None):
        """
        Load the MeshVariable from a cloud_location pointed to by
        a pyfilesystem object. 

        Parameters
        ----------

        cloud_location: 

        cloud_hdf5_filename: str
            The filename for the hdf5 file in the cloud. It should be possible to do 
            a cloud_location_handle.listdir(".") to check the file names

        Notes
        -----
        Cloud files must be in hdf5 format, and contain a vector the same
        size and with the same name as the current MeshVariable 
        """

        from quagmire.tools import cloud_fs

        if not cloud_fs:
            print("Cloud services are not available - they require fs and fs.webdav packages")
            return

        from petsc4py import PETSc
        import os

        from quagmire.tools.cloud import quagmire_cloud_fs, quagmire_cloud_cache_dir_name, cloud_download

        if self._locked:
            raise ValueError("quagmire.MeshVariable: {} - is locked".format(self.description))

        if cloud_location_handle is None:
           cloud_location_handle = quagmire_cloud_fs

        local_filename = os.path.basename(cloud_hdf5_filename)
        tempfile = os.path.join(quagmire_cloud_cache_dir_name, str(local_filename))
            
        cloud_download(cloud_hdf5_filename, tempfile, cloud_location_handle)

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(tempfile, mode='r')

        gdata = self._dm.getGlobalVec()
        if name is None:
            gdata.setName(self._ldata.getName())
        else:
            gdata.setName(str(name))

        gdata.load(ViewHDF5)

        ViewHDF5.destroy()
        ViewHDF5 = None

        self._dm.globalToLocal(gdata, self._ldata)
        self._dm.restoreGlobalVec(gdata)

        return


    def load_from_url(self, url, name=None):
        """ Load an hdf5 file pointed to by a publicly accessible url """

        from quagmire.tools import cloud_fs

        from petsc4py import PETSc
        import os

        from quagmire.tools.cloud import quagmire_cloud_fs, quagmire_cloud_cache_dir_name, url_download

        if self._locked:
            raise ValueError("quagmire.MeshVariable: {} - is locked".format(self.description))

        tempfile = os.path.join(quagmire_cloud_cache_dir_name, self._name+".h5") 
        # print("creating file {}".format(tempfile))


        url_download(url, tempfile)

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(tempfile, mode='r')

        gdata = self._dm.getGlobalVec()
        if name is None:
            gdata.setName(self._ldata.getName())
        else:
            gdata.setName(str(name))
        gdata.load(ViewHDF5)

        ViewHDF5.destroy()
        ViewHDF5 = None

        self._dm.globalToLocal(gdata, self._ldata)
        self._dm.restoreGlobalVec(gdata)

        return

    def derivative(self, dirn):
        """ (Lazy) Derivative in direction given by dirn """

        if str(dirn) in "1":
            lazyFn = self.fn_gradient(1)
        else:
            lazyFn = self.fn_gradient(0)

        return lazyFn


    def gradient(self, nit=10, tol=1e-8):
        """
        Compute values of the derivatives of PHI in the x, y directions at the nodal points.
        This routine uses SRFPACK to compute derivatives on a C-1 bivariate function.

        Parameters
        ----------
        PHI : ndarray of floats, shape (n,)
            compute the derivative of this array
        nit : int optional (default: 10)
            number of iterations to reach convergence
        tol : float optional (default: 1e-8)
            convergence is reached when this tolerance is met

        Returns
        -------
        PHIx : ndarray of floats, shape (n,)
            first partial derivative of PHI in x direction
        PHIy : ndarray of floats, shape (n,)
            first partial derivative of PHI in y direction
        """
        return self._mesh.derivative_grad(self._ldata.array, nit, tol)


    def interpolate(self, xi, yi, err=False, **kwargs):
        """
        Interpolate mesh data to a set of coordinates

        This method just passes the coordinates to the interpolation
        methods on the mesh object.

        Parameters
        ----------
        xi : array of floats, shape (l,)
            interpolation coordinates in the x direction
        yi : array of floats, shape (l,)
            interpolation coordinates in the y direction
        err : bool (default: False)
            toggle whether to return error information
        **kwargs : keyword arguments
            optional arguments to pass through to the interpolation method

        Returns
        -------
        interp : array of floats, shape (l,)
            interpolated values of MeshVariable at xi,yi coordinates
        err : array of ints, shape (l,)
            error information to diagnose interpolation / extrapolation
        """
        ## pass through for the mesh's interpolate method
        import numpy as np

        mesh = self._mesh
        PHI = self._ldata.array
        xi_array = np.array(xi).reshape(-1,1)
        yi_array = np.array(yi).reshape(-1,1)

        i, e = mesh.interpolate(xi_array, yi_array, zdata=PHI, **kwargs)

        if err:
            return i, e
        else:
            return i


    def evaluate(self, *args, **kwargs):
        """
        If the argument is a mesh, return the values at the nodes.
        In all other cases call the `interpolate` method.
        """

        import quagmire

        if len(args) == 0:
            return self._ldata.array

        if (len(args) == 1 and args[0] is self._mesh):
            return self._ldata.array
        elif len(args) == 1 and isinstance(args[0], (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh) ):
            mesh = args[0]
            return self.interpolate(mesh.coords[:,0], mesh.coords[:,1], **kwargs).reshape(-1)
        else:
            coords = np.array(args).reshape(-1,2)
            return self.interpolate(coords[:,0], coords[:,1], **kwargs).reshape(-1)


    ## Basic global operations provided by petsc4py

    def max(self):
        """ Retrieve the maximum value """
        gdata = self._dm.getGlobalVec()
        self._dm.localToGlobal(self._ldata, gdata)
        idx, val = gdata.max()
        return val

    def min(self):
        """ Retrieve the minimum value """
        gdata = self._dm.getGlobalVec()
        self._dm.localToGlobal(self._ldata, gdata)
        idx, val = gdata.min()
        return val

    def sum(self):
        """ Calculate the sum of all entries """
        gdata = self._dm.getGlobalVec()
        self._dm.localToGlobal(self._ldata, gdata)
        return gdata.sum()

    def mean(self):
        """ Calculate the mean value """
        gdata = self._dm.getGlobalVec()
        size = gdata.getSize()
        self._dm.localToGlobal(self._ldata, gdata)
        return gdata.sum()/size

    def std(self):
        """ Calculate the standard deviation """
        from math import sqrt
        gdata = self._dm.getGlobalVec()
        size = gdata.getSize()
        self._dm.localToGlobal(self._ldata, gdata)
        mu = gdata.sum()/size
        gdata -= mu
        return sqrt((gdata.sum())**2) / size


class VectorMeshVariable(MeshVariable):
    """
    The VectorMeshVariable class generates a vector variable supported on the mesh.

    To set/read nodal values, use the numpy interface via the 'self.data' property.

    Parameters
    ----------
    name : str
        Assign the MeshVariable a unique identifier
    mesh : quagmire mesh object
        The supporting mesh for the variable

    Notes
    -----
    This class inherits several methods from the `MeshVariable` class.
    """
    def __init__(self, name, mesh):
        self._mesh = mesh
        self._dm = mesh.dm.getCoordinateDM()

        name = str(name)

        # mesh variable vector
        self._ldata = self._dm.createLocalVector()
        self._ldata.setName(name)
        return

    @property
    def data(self):
        """ View of MeshVariable data of shape (n,) """
        pass

    @data.getter
    def data(self):
        return self._ldata.array.reshape(-1,2)

    @data.setter
    def data(self, val):
        import numpy as np
        if type(val) is float:
            self._ldata.set(val)
        elif np.shape(val) == (self._mesh.npoints,2):
            self._ldata.setArray(np.ravel(val))
        else:
            raise ValueError("NumPy array must be of shape ({},{})".format(self._mesh.npoints,2))

    def gradient(self):
        raise TypeError("VectorMeshVariable does not currently support gradient operations")

    def interpolate(self, xi, yi, err=False, **kwargs):
        raise TypeError("VectorMeshVariable does not currently support interpolate operations")

    def evaluate(self, xi, yi, err=False, **kwargs):
        """
        If the argument is a mesh, return the values at the nodes.
        In all other cases call the `interpolate` method.
        """
        return self.interpolate(*args, **kwargs)


    def norm(self, axis=1):
        """ evaluate the normal vector of the data along the specified axis """
        import numpy as np
        return np.linalg.norm(self.data, axis=axis)


    # We should wait to do this one for global operations

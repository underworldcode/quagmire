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
Routines common to all mesh types.

<img src="https://raw.githubusercontent.com/underworldcode/quagmire/dev/docs/images/quagmire-flowchart-commonmesh.png" style="width: 321px; float:right">

`CommonMesh` implements the following functionality:

- creating Quagmire mesh variables
- setting and reading node labels on the PETSc DM
- saving the mesh and mesh variables to HDF5 files
- handling global and local synchronisation operations

Supply a `PETSc DM` object (created from `quagmire.tools.meshtools`) to initialise the object.
"""

import numpy as np
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
# comm = MPI.COMM_WORLD
from time import perf_counter

try: range = xrange
except: pass


class CommonMesh(object):
    """
    Build routines on top of a PETSc DM mesh object common to:

    - `quagmire.mesh.pixmesh.PixMesh`
    - `quagmire.mesh.trimesh.TriMesh`
    - `quagmire.mesh.strimesh.sTriMesh`

    The above classes inherit `CommonMesh` to:

    - create `quagmire.mesh.basemesh.MeshVariable` objects
    - save the mesh and mesh variables to HDF5 files
    - retrieving and setting labels on the DM
    - synchronise local mesh information to all processors

    Parameters
    ----------
    DM : PETSc DM object
        Build this mesh object using one of the functions in
        `quagmire.tools.meshtools`
    verbose : bool
        Flag toggles verbose output
    *args : optional arguments
    **kwargs : optional keyword arguments

    Attributes
    ----------
    dm : PETSc DM object
        structured Cartesian grid or unstructured Cartesian/
        spherical mesh object
    log : PETSc log object
        contains logs for performance benchmarks
    gvec : PETSc global vector
        used to synchronise vectors across multiple processors
    lvec : PETSc local vector
        used to synchromise local information to the global vector
    sizes : tuple
        size of the local and global domains
    comm : object
        MPI COMM object for controlling global communications
    rank : int
        COMM rank is hte number assigned to each processor
    """

    def __init__(self, dm, verbose=True,  *args, **kwargs):
        self.mesh_type = 'FlatMesh'

        self.timings = dict() # store times

        self.log = PETSc.Log()
        self.log.begin()

        self.verbose = verbose

        self.dm = dm
        self.gvec = dm.createGlobalVector()
        self.lvec = dm.createLocalVector()
        self.sect = dm.getDefaultSection()
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()

        self.comm = self.dm.comm
        self.rank = self.dm.comm.rank

        lgmap_r = dm.getLGMap()
        l2g = lgmap_r.indices.copy()
        offproc = l2g < 0

        l2g[offproc] = -(l2g[offproc] + 1)
        lgmap_c = PETSc.LGMap().create(l2g, comm=self.dm.comm)

        self.lgmap_row = lgmap_r
        self.lgmap_col = lgmap_c


        ## Attach a coordinate system to the mesh:

        import quagmire

        if isinstance(self, (quagmire.mesh.trimesh.TriMesh, quagmire.mesh.pixmesh.PixMesh)):
            self.coordinates = quagmire.function.coordinates.CartesianCoordinates2D()
            self.geometry    = self.coordinates
            self.coordinate_system = self.coordinates
        else:
            self.coordinates = quagmire.function.coordinates.SphericalSurfaceLonLat2D()
            self.geometry    = self.coordinates
            self.coordinate_system = self.coordinates


        return

    def __len__(self):
        return self.npoints

    def add_variable(self, name=None, lname=None, locked=False):
        """
        Create a Quagmire mesh variable.

        Parameters
        ----------
        name : str
            name for the mesh variable
        locked : bool (default: False)
            lock the mesh variable from accidental modification

        Returns
        -------
        MeshVariable : object
            Instantiate a `quagmire.mesh.basemesh.MeshVariable`.
        """
        from quagmire.mesh import MeshVariable
        return MeshVariable(name=name, mesh=self, lname=lname, locked=locked)

    def get_label(self, label):
        """
        Retrieves all points in the DM that is marked with a specific label.
        e.g. "boundary", "coarse"

        Parameters
        ----------
        label : str
            retrieve indices on the DM marked with `label`.

        Returns
        -------
        indices : list of ints
            list of indices corresponding to the label
        """
        pStart, pEnd = self.dm.getDepthStratum(0)


        labels = []
        for i in range(self.dm.getNumLabels()):
            labels.append(self.dm.getLabelName(i))

        if label not in labels:
            raise ValueError("There is no {} label in the DM".format(label))


        stratSize = self.dm.getStratumSize(label, 1)
        if stratSize > 0:
            labelIS = self.dm.getStratumIS(label, 1)
            pt_range = np.logical_and(labelIS.indices >= pStart, labelIS.indices < pEnd)
            indices = labelIS.indices[pt_range] - pStart
        else:
            indices = np.zeros((0,), dtype=np.int)

        return indices



    def set_label(self, label, indices):
        """
        Marks local indices in the DM with a label.

        Parameters
        ----------
        label : str
            mark indices on the DM with `label`.
        indices : list of ints
            indices on the DM
        """
        pStart, pEnd = self.dm.getDepthStratum(0)
        indices += pStart

        labels = []
        for i in range(self.dm.getNumLabels()):
            labels.append(self.dm.getLabelName(i))

        if label not in labels:
            self.dm.createLabel(label)
        for ind in indices:
            self.dm.setLabelValue(label, ind, 1)
        return


    def get_boundary(self, marker="boundary"):
        """
        Find the nodes on the boundary from the DM
        If marker does not exist then the convex hull is used.

        Parameters
        ----------
        marker : str (default: 'boundary')
            name of the boundary label
        
        Returns
        -------
        mask : array of bools, shape (n,)
            mask of interior nodes
        """

        pStart, pEnd = self.dm.getDepthStratum(0)
        bmask = np.ones(self.npoints, dtype=bool)


        try:
            boundary_indices = self.get_label(marker)

        except ValueError:
            self.dm.markBoundaryFaces(marker) # marks line segments
            boundary_indices = self.tri.convex_hull()
            for ind in boundary_indices:
                self.dm.setLabelValue(marker, ind + pStart, 1)


        bmask[boundary_indices] = False
        return bmask


    def save_quagmire_project(self, file):

        import h5py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        file = str(file)
        if not file.endswith('.h5'):
            file += '.h5'

        # first save the mesh
        self.save_mesh_to_hdf5(file)

        # then save the topography
        self.topography.save(file, append=True)

        # handle saving radius to file for spherical mesh
        if np.array(self._radius).size == self.npoints:
            radius_meshVariable = self.add_variable('radius')
            radius_meshVariable.data = self._radius
            radius_meshVariable.save(file, append=True)
            radius = False
        else:
            radius = self._radius

        # now save important parameters we need to reconstruct
        # data structures. For this we need to crack open the HDF5
        # file we just saved and write attributes on a 'quagmire' group

        if comm.rank == 0:
            with h5py.File(file, mode='r+') as h5:
                quag = h5.create_group('quagmire')
                quag.attrs['id']                  = self.id
                quag.attrs['verbose']             = self.verbose
                quag.attrs['radius']              = radius
                quag.attrs['downhill_neighbours'] = self.downhill_neighbours
                quag.attrs['topography_modified'] = self._topography_modified_count

        return

    def xdmf(self, hdf5_filename, xdmf_filename=None):
        """
        Creates an xdmf file associating the saved HDF5 file with the mesh

        If no xdmf filename is not provided, a xdmf file is generated from
        the hdf5 filename with a trailing `.xdmf` extension.
        """
        from quagmire.tools.generate_xdmf import generateXdmf
        from quagmire import mpi_rank

        hdf5_filename = str(hdf5_filename)
        if not hdf5_filename.endswith('.h5'):
            hdf5_filename += '.h5'

        if mpi_rank == 0:
            generateXdmf(hdf5_filename, xdmfFilename=xdmf_filename)
        return


    def save(self, hdf5_filename):
        return self.save_quagmire_project(hdf5_filename)


    def save_mesh_to_hdf5(self, hdf5_filename):
        """
        Saves mesh information stored in the DM to HDF5 filename
        If the filename already exists, it is overwritten.

        Parameters
        ----------
        filename : str
            Save the mesh to an HDF5 filename with this name
        """
        hdf5_filename = str(hdf5_filename)
        if not hdf5_filename.endswith('.h5'):
            hdf5_filename += '.h5'

        ViewHDF5 = PETSc.Viewer()
        ViewHDF5.createHDF5(hdf5_filename, mode='w')
        ViewHDF5.view(obj=self.dm)
        ViewHDF5.destroy()
        ViewHDF5 = None


        if self.id.startswith("pixmesh"):
            import h5py
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            (minX, maxX), (minY, maxY) = self.dm.getBoundingBox()
            resX, resY = self.dm.getSizes()

            if comm.rank == 0:
                with h5py.File(hdf5_filename, mode='r+') as h5:
                    geom = h5.create_group('geometry')
                    geom.attrs['minX'] = minX
                    geom.attrs['maxX'] = maxX
                    geom.attrs['minY'] = minY
                    geom.attrs['maxY'] = maxY
                    geom.attrs['resX'] = resX
                    geom.attrs['resY'] = resY


    def save_field_to_hdf5(self, hdf5_filename, *args, **kwargs):
        """
        Saves data on the mesh to an HDF5 file
        e.g. height, rainfall, sea level, etc.

        Pass these as arguments or keyword arguments for
        their names to be saved to the hdf5 file

        Parameters
        ----------
        hdf5_filename : str
            Save the mesh variables to an HDF5 file with this name
        *args : arguments
        **kwargs : keyword arguments
        """
        import os.path

        hdf5_filename = str(hdf5_filename)
        if not hdf5_filename.endswith('.h5'):
            hdf5_filename += '.h5'

        kwdict = kwargs
        for i, arg in enumerate(args):
            key = "arr_{}".format(i)
            if key in list(kwdict.keys()):
                raise ValueError("Cannot use un-named variables\
                                  and keyword: {}".format(key))
            kwdict[key] = arg

        vec = self.dm.createGlobalVec()

        if os.path.isfile(hdf5_filename):
            mode = "a"
        else:
            mode = "w"


        for key in kwdict:
            val = kwdict[key]
            try:
                vec.setArray(val)
            except:
                self.lvec.setArray(val)
                self.dm.localToGlobal(self.lvec, vec)

            vec.setName(key)
            if self.rank == 0 and self.verbose:
                print("Saving {} to hdf5".format(key))

            ViewHDF5 = PETSc.Viewer()
            ViewHDF5.createHDF5(hdf5_filename, mode=mode)
            ViewHDF5.view(obj=vec)
            ViewHDF5.destroy()
            mode = "a"

        vec.destroy()
        vec = None


    def _gather_root(self):
        """
        MPI gather operation to root process
        """
        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)


        # Gather x,y points
        pts = self.tri.points
        self.lvec.setArray(pts[:,0])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        self.root_x = self.zvec.array.copy()

        self.lvec.setArray(pts[:,1])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        self.root_y = self.zvec.array.copy()

        self.root = True # yes we have gathered everything


    def gather_data(self, data):
        """
        Gather data on root process
        """

        # check if we already gathered pts on root
        if not self.root:
            self._gather_root()

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        return self.zvec.array.copy()

    def scatter_data(self, data):
        """
        Scatter data to all processes
        """

        toAll, zvec = PETSc.Scatter.toAll(self.gvec)

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        toAll.scatter(self.gvec, zvec)

        return zvec.array.copy()

    def sync(self, vector):
        """
        Synchronise the local domain with the global domain

        Parameters
        ----------
        vector : array of floats, shape (n,)
            local vector to be synchronised

        Returns
        -------
        vector : array of floats, shape (n,)
            local vector synchronised with the global mesh
        """

        if self.dm.comm.Get_size() == 1:
            return vector
        else:

            # Is this the same under 3.10 ?

            self.lvec.setArray(vector)
            # self.dm.localToLocal(self.lvec, self.gvec)
            self.dm.localToGlobal(self.lvec, self.gvec)
            self.dm.globalToLocal(self.gvec, self.lvec)

            return self.lvec.array.copy()


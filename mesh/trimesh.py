import numpy as np
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
comm = MPI.COMM_WORLD
from time import clock


class Triangulation(object):
    """
    An abstraction for the triangle python module <http://dzhelil.info/triangle/>
    This class mimics the Qhull structure in SciPy.
    """
    def __init__(self, coords):
        import triangle
        self.points = np.array(coords)
        self.npoints = len(coords)

        d = dict(vertices=self.points)
        tri = triangle.triangulate(d)
        self.simplices = tri['triangles']


class RecoverTriangles(object):
    def __init__(self, dm):
        sect = dm.getDefaultSection()
        lvec = dm.createLocalVector()

        self.points = dm.getCoordinatesLocal().array.reshape(-1,2)
        self.npoints = self.points.shape[0]

        # Find cells, edges, vertices

        pStart, pEnd = dm.getChart()
        pRange = np.arange(pStart,pEnd, dtype=PETSc.IntType)
        dof = np.zeros(pRange.size, dtype=PETSc.IntType)
        off = np.zeros(pRange.size, dtype=PETSc.IntType)

        for i, p in enumerate(pRange):
            dof[i] = sect.getDof(p)
            off[i] = sect.getOffset(p)

        vertices = pRange[dof>0]
        cells = pRange[np.logical_and(dof==0, off==0)]
        edges = pRange[np.logical_and(dof==0, off==off[i])]

        # recover triangles
        simplices = np.empty((cells.size, 3), dtype=PETSc.IntType)
        lvec.setArray(np.arange(0,self.npoints))

        for t, cell in enumerate(cells):
            simplices[t] = dm.vecGetClosure(sect, lvec, cell)


        self.simplices = simplices



class TriMesh(object):
    """
    Creating a global vector from a distributed DM removes duplicate entries (shadow zones)
    We recommend having 1) triangle or 2) scipy installed for Delaunay triangulations.
    """
    def __init__(self, dm, verbose=True):
        from scipy.spatial import Delaunay
        self.timings = dict() # store times

        self.log = PETSc.Log()
        self.log.begin()

        self.verbose = verbose

        self.dm = dm
        self.gvec = dm.createGlobalVector()
        self.lvec = dm.createLocalVector()
        self.sect = dm.getDefaultSection()
        self.sizes = self.gvec.getSizes(), self.gvec.getSizes()
        
        lgmap_r = dm.getLGMap()
        l2g = lgmap_r.indices.copy()
        offproc = l2g < 0

        l2g[offproc] = -(l2g[offproc] + 1)
        lgmap_c = PETSc.LGMap().create(l2g, comm=comm)

        self.lgmap_row = lgmap_r
        self.lgmap_col = lgmap_c
        


        # Get local coordinates
        coords = dm.getCoordinatesLocal().array.reshape(-1,2)

        # Delaunay triangulation
        t = clock()

        try:
            tri = Triangulation(coords)
        except ImportError:
            pass
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(coords)
        except ImportError:
            tri = RecoverTriangles(dm)

        # tri = RecoverTriangles(dm)
        self.timings['triangulation'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Delaunay triangulation {}s".format(clock()-t))

        self.npoints = tri.npoints

        # Encircling vectors
        self.v1 = tri.points[tri.simplices[:,1]] - tri.points[tri.simplices[:,0]]
        self.v2 = tri.points[tri.simplices[:,2]] - tri.points[tri.simplices[:,1]]
        self.v3 = tri.points[tri.simplices[:,0]] - tri.points[tri.simplices[:,2]]

        self.triangle_area = 0.5*(self.v1[:,0]*self.v2[:,1] - self.v1[:,1]*self.v2[:,0])

        # Calculate weigths and pointwise area
        ntriw = np.zeros(tri.npoints)
        self.weight = np.zeros(tri.npoints, dtype=PETSc.IntType)

        for t, triangle in enumerate(tri.simplices):
            self.weight[triangle] += 1
            ntriw[triangle] += abs(self.v1[t][0]*self.v3[t][1] - self.v1[t][1]*self.v3[t][0])

        self.area = ntriw/6.0
        self.adjacency_weight = 2.0/3

        self.tri = tri



        # Calculate edge lengths
        t = clock()
        self.get_edge_lengths()
        self.timings['edge lengths'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Compute edge lengths {}s".format(clock()-t))


        # Find neighbours
        t = clock()
        self.construct_neighbours()
        self.timings['construct neighbours'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Construct neighbour array {}s".format(clock()-t))


        # Find boundary points
        t = clock()
        self.bmask = self.get_boundary()
        self.timings['find boundaries'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Find boundaries {}s".format(clock()-t))


        # Build smoothing operator
        t = clock()
        self._build_smoothing_matrix()
        self.timings['smoothing matrix'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Build smoothing matrix {}s".format(clock()-t))

        self.root = False



    def node_neighbours(self, point):
        """
        Returns a list of neighbour nodes for a given point in the delaunay triangulation
        """

        return self.vertex_neighbour_vertices[1][self.vertex_neighbour_vertices[0][point]:self.vertex_neighbour_vertices[0][point+1]]


    def derivative_grad(self, PHI):
        """
        Compute x,y derivatives of PHI
        """
        u_yx = self.derivative_grad_centres(PHI)

        u_x = np.zeros(self.npoints)
        u_y = np.zeros(self.npoints)

        for idx, triangle in enumerate(self.tri.simplices):
            u_x[triangle] += u_yx[idx,1]
            u_y[triangle] -= u_yx[idx,0]

        u_x /= self.weight
        u_y /= self.weight

        return u_x, u_y


    def derivative_div(self, PHIx, PHIy):
        """
        Compute second order derivative from flux fields PHIx, PHIy
        """
        u_xy = self.derivative_grad_centres(PHIx)
        u_yx = self.derivative_grad_centres(PHIy)

        u_xx = np.zeros(self.npoints)
        u_yy = np.zeros(self.npoints)

        for idx, triangle in enumerate(self.tri.simplices):
            u_xx[triangle] += u_xy[idx,1]
            u_yy[triangle] -= u_yx[idx,0]

        u_xx /= self.weight
        u_yy /= self.weight

        return u_xx + u_yy


    def derivative_grad_centres(self, PHI):
        u = PHI.reshape(-1,1)

        u1 = (u[self.tri.simplices[:,0]] + u[self.tri.simplices[:,1]]) * 0.5
        u2 = (u[self.tri.simplices[:,1]] + u[self.tri.simplices[:,2]]) * 0.5
        u3 = (u[self.tri.simplices[:,2]] + u[self.tri.simplices[:,0]]) * 0.5

        u_yx = (u1*self.v1 + u2*self.v2 + u3*self.v3)/self.triangle_area.reshape(-1,1)
        return u_yx


    def get_edge_lengths(self):
        """
        Find all edges in a triangluation and their lengths
        """
        points = self.tri.points

        i1 = np.sort([self.tri.simplices[:,0], self.tri.simplices[:,1]], axis=0)
        i2 = np.sort([self.tri.simplices[:,0], self.tri.simplices[:,2]], axis=0)
        i3 = np.sort([self.tri.simplices[:,1], self.tri.simplices[:,2]], axis=0)

        a = np.hstack([i1, i2, i3]).T

        # find unique rows in numpy array 
        # <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array>
        b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        edges = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

        edge_lengths = np.linalg.norm(points[edges[:,0]] - points[edges[:,1]], axis=1)

        self.edges = edges
        self.edge_lengths = edge_lengths


    def construct_neighbours(self):
        """
        Find neighbours from edges and store as CSR coordinates.

        This allows you to directly ask the neighbours for a given node a la Qhull,
        or efficiently construct a sparse matrix (PETSc/SciPy)
        """

        row = np.hstack([self.edges[:,0], self.edges[:,1]])
        col = np.hstack([self.edges[:,1], self.edges[:,0]])
        val = np.hstack([self.edge_lengths, self.edge_lengths])

        # sort by row
        sort = row.argsort()
        row = row[sort].astype(PETSc.IntType)
        col = col[sort].astype(PETSc.IntType)
        val = val[sort]

        nnz = np.bincount(row) # number of nonzeros
        indptr = np.insert(np.cumsum(nnz),0,0)

        self.vertex_neighbours = nnz.astype(PETSc.IntType)
        self.vertex_neighbour_vertices = indptr, col
        self.vertex_neighbour_distance = val

        # We may not need this, but constuct anyway for now!
        neighbours = [[]]*self.npoints
        closed_neighbours = [[]]*self.npoints

        for i in xrange(indptr.size-1):
            start, end = indptr[i], indptr[i+1]
            neighbours[i] = np.array(col[start:end])
            closed_neighbours[i] = np.hstack([i, neighbours[i]])

        self.neighbour_list = np.array(neighbours)
        self.neighbour_array = np.array(closed_neighbours)


    def _build_smoothing_matrix(self):

        indptr, indices = self.vertex_neighbour_vertices
        weight  = 1.0/self.weight
        nweight = weight[indices]

        lgmask = self.lgmap_row.indices >= 0


        nnz = self.vertex_neighbours[lgmask] + 1

        # smoothMat = self.dm.createMatrix()
        # smoothMat.setOption(smoothMat.Option.NEW_NONZERO_LOCATIONS, False)
        smoothMat = PETSc.Mat().create(comm=comm)
        smoothMat.setType('aij')
        smoothMat.setSizes(self.sizes)
        smoothMat.setLGMap(self.lgmap_row, self.lgmap_col)
        smoothMat.setFromOptions()
        smoothMat.setPreallocationNNZ(nnz)

        # read in data 
        smoothMat.setValuesLocalCSR(indptr.astype(PETSc.IntType), indices.astype(PETSc.IntType), nweight)
        self.lvec.setArray(weight)
        self.dm.localToGlobal(self.lvec, self.gvec)
        smoothMat.setDiagonal(self.gvec)

        smoothMat.assemblyBegin()
        smoothMat.assemblyEnd()

        self.localSmoothMat = smoothMat


    def local_area_smoothing(self, data, its=1, centre_weight=0.75):

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        smooth_data = self.gvec.copy()

        for i in xrange(0, its):
            self.localSmoothMat.mult(smooth_data, self.gvec)
            smooth_data = centre_weight*smooth_data + (1.0 - centre_weight)*self.gvec

        self.dm.globalToLocal(smooth_data, self.lvec)

        return self.lvec.array


    def get_boundary(self, marker="BC"):
        """
        Get distributed boundary information
        """
        # self.dm.markBoundaryFaces(str(marker))

        pStart, pEnd = self.dm.getDepthStratum(0)
        eStart, eEnd = self.dm.getDepthStratum(1)
        eRange = np.arange(eStart, eEnd, dtype=PETSc.IntType)

        bnd = np.zeros_like(eRange)
        for idx, e in enumerate(eRange):
            bnd[idx] = self.dm.getLabelValue(marker, e)
            
        boundary = eRange[bnd==1]
        boundary_ind = np.zeros((boundary.size, 2), dtype=PETSc.IntType)

        for idx, e in enumerate(boundary):
            boundary_ind[idx] = self.dm.getCone(e)
            
        boundary_ind -= pStart # return to local ordering

        bmask = np.ones(self.npoints, dtype=bool)
        bmask[boundary_ind] = False
        bmask[self.lgmap_row.indices < 0] = True

        return bmask

    def save_mesh_to_file(self, file):
        """
        Save mesh points to a HDF5 file.
        Requires h5py and a HDF5 built with at least the following options:
            ./configure --enable-parallel --enable-shared
        """
        import h5py

        if isinstance(file, basestring):
            if not file.endswith('.h5'):
                file = file + '.h5'

        on_proc = self.lgmap_row.indices >= 0
        indices = self.lgmap_row.indices[on_proc]

        gsize = self.gvec.getSizes()[1]

        with h5py.File(str(file), 'w', driver='mpio', comm=comm) as f:
            grp = f.create_group('dmplex')
            dset = grp.create_dataset('points', (gsize,2), dtype='f')
            dset[list(indices)] = self.tri.points[on_proc]


    def save_field_to_file(self, file, *args, **kwargs):
        """
        Save field to an HDF5 file.
        Maps the field from a local vector to global.
        """
        import h5py

        if isinstance(file, basestring):
            if not file.endswith('.h5'):
                file = file + '.h5'

        kwdict = kwargs
        for i, arg in enumerate(args):
            key = 'arr_%d' % i
            if key in kwdict.keys():
                raise ValueError("Cannot use un-named variables and keyword %s" % key)
            kwdict[key] = arg

        indices = self.lgmap_row.indices[self.lgmap_row.indices >= 0]
        gsize = self.gvec.getSizes()[1]
        lsize = self.lvec.getSizes()[1]

        with h5py.File(str(file), 'w', driver='mpio', comm=comm) as f:
            for key in kwdict:
                val = kwdict[key]
                if val.size == lsize:
                    self.lvec.setArray(val)
                    self.dm.localToGlobal(self.lvec, self.gvec)
                else:
                    self.gvec.setArray(val)
                dset = f.create_dataset(key, (gsize,), dtype='f')
                dset[list(indices)] = self.gvec.array


    def _gather_root(self):
        """
        MPI gather operation to root processor
        """
        self.tozero, self.zvec = PETSc.Scatter.toZero(self.gvec)


        # Gather x,y points
        pts = self.tri.points
        self.lvec.setArray(pts[:,0])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        x = self.zvec.array.copy()
        
        self.lvec.setArray(pts[:,1])
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        y = self.zvec.array.copy()

        if comm.rank == 0:
            # Re-triangulate with whole domain
            self.tri0 = Triangulation(np.vstack([x,y]).T)

        self.root = True # yes we have gathered everything


    def gather_data(self, data):
        """
        Gather data on root processor 
        """

        # check if we already gathered pts on root
        if not self.root:
            self._gather_root()

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.tozero.scatter(self.gvec, self.zvec)

        return self.zvec.array.copy()


    def _local_global_local(self, vector):
        """ Communicate to global then back again """
        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)
        self.dm.globalToLocal(self.gvec, self.lvec)
        return self.lvec.array.copy()

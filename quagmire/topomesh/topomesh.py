"""
Copyright 2016-2017 Louis Moresi, Ben Mather, Romain Beucher

This file is part of Quagmire.

Quagmire is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Quagmire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numpy.ma as ma
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
comm = MPI.COMM_WORLD
from time import clock

try: range = xrange
except: pass


class TopoMesh(object):
    def __init__(self, downhill_neighbours=2, *args, **kwargs):
        self.downhill_neighbours = downhill_neighbours

        # Initialise cumulative flow vectors
        self.DX0 = self.gvec.duplicate()
        self.DX1 = self.gvec.duplicate()
        self.dDX = self.gvec.duplicate()

        # Initialise mesh fields
        # self.height = self.gvec.duplicate()
        # self.slope = self.gvec.duplicate()


    def update_height(self, height):
        """
        Update height field
        """

        height = np.array(height)
        if height.size != self.npoints:
            raise IndexError("Incompatible array size, should be {}".format(self.npoints))

        t = clock()
        self.height = height.copy()
        dHdx, dHdy = self.derivative_grad(height)
        self.slope = np.hypot(dHdx, dHdy)

        # Lets send and receive this from the global space
        # self.slope[:] = self.sync(self.slope)
        self.timings['gradient operation'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Compute slopes {}s".format(self.dm.comm.rank, clock()-t))


        t = clock()
        self._build_downhill_matrix_iterate()
        self.timings['downhill matrices'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print("{} - Build downhill matrices {}s".format(self.dm.comm.rank, clock()-t))


    def _update_height_partial(self, height):
        """
        Partially update height field for specific purpose of patching topographic low points etc.
        This allows rebuilding of the Adjacency1/Downhill matrix but does not compute gradients or
        higher-order descent paths.
        """

        height = np.array(height)
        if height.size != self.npoints:
            raise IndexError("Incompatible array size, should be {}".format(self.npoints))

        self.height = height

        t = clock()

        neighbours = self.downhill_neighbours
        self.downhill_neighbours = 1

        self._build_adjacency_matrix_iterate()
        if self.verbose:
            print(" - Partial rebuild of downhill matrices {}s".format(clock()-t))

        self.downhill_neighbours = neighbours

        return



    def _sort_nodes_by_field(self, height):

        # Sort neighbours by gradient
        indptr, indices = self.vertex_neighbour_vertices
        # gradH = height[indices]/self.vertex_neighbour_distance

        self.node_high_to_low = np.argsort(height)[::-1]

        neighbour_array_lo_hi = self.neighbour_array.copy()
        neighbour_array_2_low = np.empty((self.npoints, 2), dtype=PETSc.IntType)

        for i in xrange(indptr.size-1):
            # start, end = indptr[i], indptr[i+1]
            # neighbours = np.hstack([i, indices[start:end]])
            # order = height[neighbours].argsort()
            # neighbour_array_lo_hi[i] = neighbours[order]
            # neighbour_array_2_low[i] = neighbour_array_lo_hi[i][:2]

            neighbours = self.neighbour_array[i]
            order = height[neighbours].argsort()
            neighbour_array_lo_hi[i] = neighbours[order]
            neighbour_array_2_low[i] = neighbour_array_lo_hi[i][:2]

        self.neighbour_array_lo_hi = neighbour_array_lo_hi
        self.neighbour_array_2_low = neighbour_array_2_low


        # neighbour_array_lo_hi = self.neighbour_array.copy()
        # for node in xrange(0, self.npoints):
        #     neighbours = self.neighbour_array[node]


    def _adjacency_matrix_template(self, nnz=(1,1)):

        matrix = PETSc.Mat().create(comm=comm)
        matrix.setType('aij')
        matrix.setSizes(self.sizes)
        matrix.setLGMap(self.lgmap_row, self.lgmap_col)
        matrix.setFromOptions()
        matrix.setPreallocationNNZ(nnz)

        return matrix


## This is the lowest near node / lowest extended neighbour

# hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.near_neighbours_mask)
# low_neighbours = np.argmin(hnear, axis=1)
#
# hnear = np.ma.array(mesh.height[mesh.neighbour_cloud], mask=mesh.extended_neighbours_mask)
# low_eneighbours = np.argmin(hnear, axis=1)
#
#


## This version is based on distance not mesh connectivity -
##
## Be careful - it finds the nearest low node and not the lowest
## near node.
##

    def _build_adjacency_matrix_iterate(self):

        self.adjacency = dict()
        self.down_neighbour = dict()

        data = np.empty(self.npoints)
        down_neighbour = np.empty(self.npoints, dtype=PETSc.IntType)

        indptr = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        node_range = indptr[:-1]

        # compute low neighbours
        nheight  = self.height[self.neighbour_cloud]
        # nheightm = ma.array( nheight, mask = self.neighbour_cloud.mask)

        dneighTF = nheight < self.height.reshape(-1,1)
        dneighL1  = dneighTF.argmax(axis=1)

        for i in range(1, self.downhill_neighbours+1):
            data.fill(1.0)

            dneighL = dneighTF.argmax(axis=1)
            own = dneighL == 0
            dneighL[own] = dneighL1[own]
            dneighTF[node_range, dneighL[node_range]] = False

            down_neighbour[:] = self.neighbour_cloud[node_range, dneighL[node_range]]
            data[dneighL == 0] = 0.0

            adjacency = self._adjacency_matrix_template()
            adjacency.setValuesLocalCSR(indptr, down_neighbour, data)
            adjacency.assemblyBegin()
            adjacency.assemblyEnd()
            self.adjacency[i] = adjacency.transpose()
            self.down_neighbour[i] = down_neighbour.copy()

    def _build_downhill_matrix_iterate(self):

        self._build_adjacency_matrix_iterate()
        weights = np.empty((self.downhill_neighbours, self.npoints))

        # Process weights
        for i in range(0, self.downhill_neighbours):
            down_N = self.down_neighbour[i+1]
            grad = np.abs(self.height - self.height[down_N]+1.0e-10) / (1.0e-10 + \
                   np.hypot(self.coords[:,0] - self.coords[down_N,0],
                            self.coords[:,1] - self.coords[down_N,1] ))

            weights[i,:] = np.sqrt(grad)

        weights /= weights.sum(axis=0)
        w = self.gvec.duplicate()


        # Store weighted downhill matrices
        downhill_matrices = [None]*self.downhill_neighbours
        for i in range(0, self.downhill_neighbours):
            N = i + 1
            self.lvec.setArray(weights[i])
            self.dm.localToGlobal(self.lvec, w)

            D = self.adjacency[N].copy()
            D.diagonalScale(R=w)
            downhill_matrices[i] = D


        # Sum downhill matrices
        self.downhillMat = downhill_matrices[0]
        for i in range(1, self.downhill_neighbours):
            self.downhillMat += downhill_matrices[i]
            downhill_matrices[i].destroy()


    def build_cumulative_downhill_matrix(self):
        """
        Build non-sparse, single hit matrices to propagate information downhill
        (self.sweepDownToOutflowMat and self.downhillCumulativeMat)

        This may be expensive in terms of storage so this is only done if
        self.storeDense == True and the matrices are also out of date (which they
        will be if the height field is changed)

        downhillCumulativeMat = I + D + D**2 + D**3 + ... D**N where N is the length of the graph

        """

        downSweepMat    = self.accumulatorMat.copy()
        downHillaccuMat = self.downhillMat.copy()
        accuM           = self.downhillMat.copy()   # work matrix

        DX1 = self.gvec.duplicate()
        DX0 = self.gvec.duplicate()
        DX0.set(1.0)

        err = np.array([True])
        err_proc = np.ones(comm.size, dtype=bool)

        while err_proc.any():
            downSweepMat    = downSweepMat*self.accumulatorMat  # N applications of the accumulator
            accuM           = accuM*self.downhillMat
            downHillaccuMat = downHillaccuMat + accuM
            DX0 = self.downhillMat*DX1

            err[0] = np.any(DX0)
            comm.Allgather([err, MPI.BOOL], [err_proc, MPI.BOOL])

        # add identity matrix
        I = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        J = np.arange(0, self.npoints, dtype=PETSc.IntType)
        V = np.ones(self.npoints)
        identityMat = self._adjacency_matrix_template()
        identityMat.setValuesLocalCSR(I, J, V)
        identityMat.assemblyBegin()
        identityMat.assemblyEnd()

        downHillaccuMat += identityMat

        self.downhillCumulativeMat = downHillaccuMat
        self.sweepDownToOutflowMat = downSweepMat


    def cumulative_flow_verbose(self, vector, verbose=False, maximum_its=None):

        if not maximum_its:
            maximum_its = 1000000000000


        print "DHmat**2"
        # downhillMat2 = self.downhillMat * self.downhillMat
        # downhillMat4 = downhillMat2 * downhillMat2
        # downhillMat8 = downhillMat4 * downhillMat4
        downhillMat = self.downhillMat
        print "DHmat**2 - done "

        DX0 = self.DX0
        DX1 = self.DX1
        dDX = self.dDX

        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, DX0, addv=PETSc.InsertMode.INSERT_VALUES)

        DX1.setArray(DX0)
        # DX0.assemble()
        # DX1.assemble()

        niter = 0
        equal = False

        tolerance = 1e-8 * DX1.max()[1]

        while not equal and niter < maximum_its:
            dDX.setArray(DX1)
            #dDX.assemble()

            downhillMat.mult(DX1, self.gvec)
            DX1.setArray(self.gvec)
            DX1.assemble()

            DX0 += DX1

            dDX.axpy(-1.0, DX1)
            dDX.abs()
            max_dDX = dDX.max()[1]

            equal = max_dDX < tolerance

            if self.dm.comm.rank==0 and verbose and niter%10 == 0:
                print "{}: Max Delta - {} ".format(niter, max_dDX)

            niter += 1

        self.dm.globalToLocal(DX0, self.lvec)

        return niter, self.lvec.array.copy()

    def cumulative_flow(self, vector):

        niter, cumulative_flow_vector = self.cumulative_flow_verbose(vector)
        return cumulative_flow_vector



    def downhill_smoothing(self, data, its, centre_weight=0.75, use3path=False):

        downhillMat = self.downhillMat

        norm = self.gvec.duplicate()
        smooth_data = self.gvec.duplicate()

        self.gvec.set(1.0)
        self.downhillMat.mult(self.gvec, norm)

        mask = norm.array == 0.0

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, smooth_data)
        for i in xrange(0, its):
            self.downhillMat.mult(smooth_data, self.gvec)
            smooth_data.setArray((1.0 - centre_weight) * self.gvec.array + \
                                  smooth_data.array*np.where(mask, 1.0, centre_weight))

        self.dm.globalToLocal(smooth_data, self.lvec)

        return self.lvec.array.copy()


    def uphill_smoothing(self, data, its, centre_weight=0.75, use3path=False):

        downhillMat = self.downhillMat


        norm2 = self.gvec.duplicate()
        smooth_data = self.gvec.duplicate()

        self.gvec.set(1.0)
        self.downhillMat.multTranspose(self.gvec, norm2)

        mask = norm2.array == 0.0
        norm2.array[~mask] = 1.0/norm2.array[~mask]

        self.lvec.setArray(data)
        self.dm.localToGlobal(self.lvec, smooth_data)
        for i in xrange(0, its):
            self.downhillMat.multTranspose(smooth_data, self.gvec)
            smooth_data.setArray((1.0 - centre_weight) * self.gvec.array * norm2 + \
                                  smooth_data.array*np.where(mask, 1.0, centre_weight))

        self.dm.globalToLocal(smooth_data, self.lvec)
        self.lvec *= data.mean()/self.lvec.array.mean()

        return self.lvec.array.copy()


    def streamwise_smoothing(self, data, its, centre_weight=0.75, use3path=False):
        """
        A smoothing operator that is limited to the uphill / downhill nodes for each point. It's hard to build
        a conservative smoothing operator this way since "boundaries" occur at irregular internal points associated
        with watersheds etc. Upstream and downstream smoothing operations bracket the original data (over and under,
        respectively) and we use this to find a smooth field with the same mean value as the original data. This is
        done for each application of the smoothing.
        """


        smooth_data_d = self.downhill_smoothing(data, its, centre_weight=centre_weight, use3path=use3path)
        smooth_data_u = self.uphill_smoothing(data, its, centre_weight=centre_weight, use3path=use3path)

        return 0.5*(smooth_data_d + smooth_data_u)



    def _node_lowest_neighbour(self, node):
        """
        Find the lowest node in the neighbour list of the given node
        """

        lowest = self.neighbour_array_lo_hi[node][0]

        if lowest != node:
            return lowest
        else:
            return -1



    def _node_highest_neighbour(self, node):
        """
        Find the highest node in the neighbour list of the given node
        """

        highest = self.neighbour_array_lo_hi[node][-1]

        if highest != node:
            return highest
        else:
            return -1


    def _node_walk_downhill(self, node):
        """
        Walks downhill terminating when the downhill node is already claimed
        """

        chain = -np.ones(self.npoints, dtype=np.int32)

        idx = 0
        max_idx = self.npoints
        chain[idx] = node
        low_neighbour = self._node_lowest_neighbour(node)
        junction = -1

        while low_neighbour != -1:
            idx += 1
            chain[idx] = low_neighbour
            if self.node_chain_lookup[low_neighbour] != -1:
                junction = self.node_chain_lookup[low_neighbour]
                break

            low_neighbour = self._node_lowest_neighbour(low_neighbour)

        return junction, chain[0:idx+1]


    def build_node_chains(self):
        """ NEEDS WORK
        Builds all the chains for the mesh which flow from high to low and terminate
        when they meet with an existing chain.

        The following data structures become available once this function has been called:

            self.node_chain_lookup - tells you the chain in which a given node number lies
            self.node_chain_list   - is a list of the chains of nodes (each of which is an list)

        The terminating node of a chain may be the junction with another (pre-exisiting) chain
        and will be a member of that chain. Backbone chains which run from the highest level
        to the base level or the boundary are those whose terminal node is also a member of the same chain.

        Nodes which are at a base level given by self.base, are collected separately
        into chain number 0.
        """

        self.node_chain_lookup = -np.ones(self.npoints, dtype=np.int32)
        self.node_chain_list = []


        node_chain_idx = 1

        self.node_chain_list.append([]) # placeholder for any isolated base-level nodes

        for node1 in self.node_high_to_low:
            if self.node_chain_lookup[node1] != -1:
                continue

            junction, this_chain = self._node_walk_downhill(node1)

            if len(this_chain) > 1:
                self.node_chain_list.append(this_chain)

                self.node_chain_lookup[this_chain[0:-1]] = node_chain_idx
                if self.node_chain_lookup[this_chain[-1]] == -1:
                    self.node_chain_lookup[this_chain[-1]] = node_chain_idx

                node_chain_idx += 1

            else:
                self.node_chain_list[0].append(this_chain[0])
                self.node_chain_lookup[this_chain[0]] = 0

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
from mpi4py import MPI
import sys,petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from time import clock
comm = MPI.COMM_WORLD

from scipy.spatial import cKDTree as _cKDTree


class TopoMesh(object):
    def __init__(self):
        self.build3Neighbours = False
        self.downhill_neighbours = 2

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
        self.slope[:] = self.sync(self.slope)
        self.timings['gradient operation'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]


        t = clock()
        self._build_downhill_matrix_new()
        self.timings['downhill matrices'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Build downhill matrices {}s".format(clock()-t))


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

    def _build_downhill_matrix_new(self):

        vec = self.gvec.duplicate()
        self.lvec.setArray(self.slope)
        self.dm.localToGlobal(self.lvec, vec)

        self._build_adjacency_matrix_123()


        grad1 = np.abs(self.height - self.height[self.down_neighbour1]+1.0e-10) / (1.0e-10 +
                np.hypot(self.coords[:,0] - self.coords[self.down_neighbour1,0],
                         self.coords[:,1] - self.coords[self.down_neighbour1,1] ))

        grad2 = np.abs(self.height - self.height[self.down_neighbour2]+1.0e-10) / (1.0e-10 +
                np.hypot(self.coords[:,0] - self.coords[self.down_neighbour2,0],
                         self.coords[:,1] - self.coords[self.down_neighbour2,1] ))

        w1 = np.sqrt(grad1)
        w2 = np.sqrt(grad2)

        if not self.build3Neighbours:
            w1 /= (w1+w2)
            w2  = 1.0 - w1

            downhillMat1  = self.adjacency1.copy()
            downhillMat2 = self.adjacency2.copy()

            self.lvec.setArray(w1)
            self.dm.localToGlobal(self.lvec, vec)
            downhillMat1.diagonalScale(R=vec)

            self.lvec.setArray(w2)
            self.dm.localToGlobal(self.lvec, vec)
            downhillMat2.diagonalScale(R=vec)

            self.downhillMat = downhillMat1 + downhillMat2

            downhillMat1.destroy()
            downhillMat2.destroy()

        else:
            grad3 = np.abs(self.height - self.height[self.down_neighbour3]+1.0e-15) / (1.0e-10 +
                    np.hypot(self.coords[:,0] - self.coords[self.down_neighbour3,0],
                             self.coords[:,1] - self.coords[self.down_neighbour3,1] ) )

            w3 = np.sqrt(grad3)

            w1 /= (w1+w2+w3)
            w2 /= (w1+w2+w3)
            w3 = 1.0 - (w1+w2)

            downhillMat1 = self.adjacency1.copy()
            downhillMat2 = self.adjacency2.copy()
            downhillMat3 = self.adjacency3.copy()

            self.lvec.setArray(w1)
            self.dm.localToGlobal(self.lvec, vec)
            downhillMat1.diagonalScale(R=vec)

            self.lvec.setArray(w2)
            self.dm.localToGlobal(self.lvec, vec)
            downhillMat2.diagonalScale(R=vec)

            self.lvec.setArray(w3)
            self.dm.localToGlobal(self.lvec, vec)
            downhillMat3.diagonalScale(R=vec)

            self.downhillMat = downhillMat1 + downhillMat2 + downhillMat3

            downhillMat1.destroy()
            downhillMat2.destroy()
            downhillMat3.destroy()

        return

    def _adjacency_matrix_template(self, nnz=(1,1)):

        matrix = PETSc.Mat().create(comm=comm)
        matrix.setType('aij')
        matrix.setSizes(self.sizes)
        matrix.setLGMap(self.lgmap_row, self.lgmap_col)
        matrix.setFromOptions()
        matrix.setPreallocationNNZ(nnz)

        return matrix


## This version is based on distance not mesh connectivity -

    def _build_adjacency_matrix_iterate(self):

        self.adjacency = dict()
        self.down_neighbour = dict()

        data = np.empty(self.npoints)
        down_neighbour = np.empty(self.npoints, dtype=PETSc.IntType)

        indptr = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        node_range = indptr[:-1]

        # compute low neighbours
        dneighTF = self.height[self.neighbour_cloud] < self.height.reshape(-1,1)
        dneighL1 = dneighTF.argmax(axis=1)

        for i in range(1, self.downhill_neighbours+1):
            data.fill(1.0)

            dneighL = dneighTF.argmax(axis=1)
            dneighL[dneighL == 0] = dneighL1[dneighL == 0]
            dneighTF[node_range, dneighL[node_range]] = False

            down_neighbour[:] = self.neighbour_cloud[node_range, dneighL[node_range]]
            data[dneighL == 0] = 0.0

            adjacency = self._adjacency_matrix_template()
            adjacency.assemblyBegin()
            adjacency.setValuesLocalCSR(indptr, down_neighbour, data)
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
                   np.hypot(self.tri.points[:,0] - self.tri.points[down_N,0],
                            self.tri.points[:,1] - self.tri.points[down_N,1] ))

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


    def _build_adjacency_matrix_123(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the 2nd steepest direction.

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.
        """

        # preallocate matrix entry and index arrays
        data1 = np.ones(self.npoints)
        data2 = np.ones(self.npoints)
        data3 = np.ones(self.npoints)

        down_neighbour1 = np.empty(self.npoints, dtype=PETSc.IntType)
        down_neighbour2 = np.empty(self.npoints, dtype=PETSc.IntType)
        down_neighbour3 = np.empty(self.npoints, dtype=PETSc.IntType)

        indptr  = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        node_range = indptr[:-1]

        # compute low neighbours

        dneighTF = self.height[self.neighbour_cloud] < self.height.reshape(-1,1)

        dneighL1 = dneighTF.argmax(axis=1)
        dneighTF[node_range, dneighL1[node_range]] = False

        dneighL2 = dneighTF.argmax(axis=1)
        dneighL2[dneighL2 == 0] = dneighL1[dneighL2 == 0]
        dneighTF[node_range, dneighL2[node_range]] = False

        dneighL3 = dneighTF.argmax(axis=1)
        dneighL3[dneighL3 == 0] = dneighL1[dneighL3 == 0]
        dneighTF[node_range, dneighL3[node_range]] = False

        # update column and data vectors
        down_neighbour1[:] = self.neighbour_cloud[node_range, dneighL1[node_range]]
        down_neighbour2[:] = self.neighbour_cloud[node_range, dneighL2[node_range]]
        down_neighbour3[:] = self.neighbour_cloud[node_range, dneighL3[node_range]]

        data1[dneighL1 == 0] = 0.0
        data2[dneighL2 == 0] = 0.0
        data3[dneighL3 == 0] = 0.0

        # read into matrix
        adjacency1 = self._adjacency_matrix_template()
        adjacency1.assemblyBegin()
        adjacency1.setValuesLocalCSR(indptr, down_neighbour1, data1)
        adjacency1.assemblyEnd()

        self.adjacency1 = adjacency1.transpose()
        self.down_neighbour1 = down_neighbour1

        # read into matrix
        adjacency2 = self._adjacency_matrix_template()
        adjacency2.assemblyBegin()
        adjacency2.setValuesLocalCSR(indptr, down_neighbour2, data2)
        adjacency2.assemblyEnd()

        self.adjacency2 = adjacency2.transpose()
        self.down_neighbour2 = down_neighbour2

        if self.build3Neighbours:
        # read into matrix
            adjacency3 = self._adjacency_matrix_template()
            adjacency3.assemblyBegin()
            adjacency3.setValuesLocalCSR(indptr, down_neighbour3, data3)
            adjacency3.assemblyEnd()

            self.adjacency3 = adjacency3.transpose()
            self.down_neighbour3 = down_neighbour3


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


    def cumulative_flow(self, vector):

        downhillMat = self.downhillMat

        DX0 = self.DX0
        DX1 = self.DX1
        dDX = self.dDX

        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, DX0)

        DX1.setArray(DX0)

        niter = 0
        equal = False

        tolerance = 1e-8

        while not equal:
            dDX.setArray(DX1)
            downhillMat.mult(DX1, self.gvec)
            DX1.setArray(self.gvec)
            DX0 += DX1

            dDX.axpy(-1.0, DX1)
            dDX.abs()
            max_dDX = dDX.max()[1]

            equal = max_dDX < tolerance
            niter += 1

        self.dm.globalToLocal(DX0, self.lvec)

        return self.lvec.array.copy()


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

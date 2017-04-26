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

        pass


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
        slope = np.hypot(dHdx, dHdy)

        # Lets send and receive this from the global space
        self.slope = self._local_global_local(slope)

        self.timings['gradient operation'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]

        t = clock()
        self._sort_nodes_by_field(height)
        self.timings['sort heights'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]
        if self.verbose:
            print(" - Sort nodes by field {}s".format(clock()-t))

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

        vec = self.gvec.copy()
        vec.setArray(self.slope)

        self._build_adjacency_matrix_1()
        self._build_adjacency_matrix_2()

        w1 = np.sqrt(self.slope[self.down_neighbour1])
        w2 = np.sqrt(self.slope[self.down_neighbour2])

        w1 /= (w1+w2)
        w2  = 1.0 - w1


        vec.setArray(w1)

        downhillMat  = self.adjacency1.copy()
        downhillMat2 = self.adjacency2.copy()

        vec.setArray(w1)
        downhillMat.diagonalScale(L=vec)

        vec.setArray(w2)
        downhillMat2.diagonalScale(L=vec)

        self.downhillMat = downhillMat + downhillMat2

        downhillMat.destroy()
        downhillMat2.destroy()

    def _build_downhill_matrix(self):

        Z_neighbours = self.slope[self.neighbour_array_2_low]**0.5
        Z_neighbours_sum = np.clip(Z_neighbours.sum(axis=1), 1e-12, 1e99)
        weight = Z_neighbours/Z_neighbours_sum.reshape(-1,1)

        # self._build_adjacency_matrix_1()
        indptr = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        index  = np.arange(0, self.npoints, dtype=PETSc.IntType)

        down_neighbour1 = self.neighbour_array_2_low[:,0]
        down_neighbour2 = self.neighbour_array_2_low[:,1].copy()
        data = np.ones(self.npoints)

        # read into accumulator matrix
        accumulatorMat = self._adjacency_matrix_template()
        accumulatorMat.setValuesLocalCSR(indptr, down_neighbour1, data)
        accumulatorMat.assemblyBegin()
        accumulatorMat.assemblyEnd()
        self.accumulatorMat = accumulatorMat.transpose()


        # find nodes that are their own low neighbour!
        data[np.logical_or(index==down_neighbour1, ~self.bmask)] = 0.0
        mask = index == down_neighbour2
        down_neighbour2[mask] = down_neighbour1[mask]


        # read into adjacency matrices
        adjacency1 = self._adjacency_matrix_template()
        adjacency1.setValuesLocalCSR(indptr, down_neighbour1, weight[:,0]*data)
        adjacency1.assemblyBegin()
        adjacency1.assemblyEnd()
        self.adjacency1 = adjacency1.transpose()

        adjacency2 = self._adjacency_matrix_template()
        adjacency2.setValuesLocalCSR(indptr, down_neighbour2, weight[:,1]*data)
        adjacency2.assemblyBegin()
        adjacency2.assemblyEnd()
        self.adjacency2 = adjacency2.transpose()


        # indptr = np.arange(0, self.npoints*2+2, 2, dtype=PETSc.IntType)
        # down_neighbours = np.vstack([down_neighbour1, down_neighbour2]).ravel(order='F')
        # data2 = np.vstack([data*weight, data*(1.0-weight)]).ravel(order='F')

        # downhillMat = self._adjacency_matrix_template(nnz=(2,2))
        # downhillMat.setValuesLocalCSR(indptr, down_neighbours, data2, addv=True)
        # downhillMat.assemblyBegin()
        # downhillMat.assemblyEnd()
        # self.downhillMat = downhillMat.transpose()


        # self._build_adjacency_matrix_2()

        # self.downhillMat = weight * self.adjacency1 + (1.0-weight) * self.adjacency2

        # self.downhillMat = weight * self.adjacency1
        # self.downhillMat.axpy(1.0, (1.0-weight)*self.adjacency2)
        self.downhillMat = self.adjacency1 + self.adjacency2


    def _build_downhill_matrix_neighbours(self):

        # Lets see if we can't read all neighbours in
        maxC = 0
        for row in self.neighbour_array_lo_hi:
            if row.size > maxC:
                maxC = row.size # -1 (unless it gives to itself?)


        indptr, indices = self.vertex_neighbour_vertices

        downhillMat = self._adjacency_matrix_template(nnz=(maxC,1))

        for i in xrange(0, indptr.size-1):
            neighbours = self.neighbour_array_lo_hi[i]
            heightN = self.height[neighbours]

# Benchmark - this converges
            # Find all nodes where height is less than current node
            down_neighbours = neighbours[heightN<self.height[i]]

# Benchmark - this does not
#            # Find all nodes where height is equal or less than current node
#            down_neighbours = neighbours[heightN<=self.height[i]]

            Z_neighbours = self.slope[down_neighbours]**0.5
            weight = Z_neighbours/Z_neighbours.sum()

## This is wrong - gives incompatible values
            # if i==down_neighbours[0]:
            #     weight[0] = 0

## This is probably what is meant by the above but does not really work
             # weight[np.where(down_neighbours == i)] = 0.0



            # read in downhill neighbours to downhill matrix
            downhillMat.setValuesLocal(i, down_neighbours.astype(np.int32), weight)


        downhillMat.assemblyBegin()
        downhillMat.assemblyEnd()
        self.downhillMat = downhillMat.transpose()



    def _adjacency_matrix_template(self, nnz=(1,1)):

        matrix = PETSc.Mat().create(comm=comm)
        matrix.setType('aij')
        matrix.setSizes(self.sizes)
        matrix.setLGMap(self.lgmap_row, self.lgmap_col)
        matrix.setFromOptions()
        matrix.setPreallocationNNZ(nnz)

        return matrix


## This version is based on distance not mesh connectivity

    def _build_adjacency_matrix_1(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the steepest direction.

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.
        """

        indptr  = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        down_neighbour1 = self.neighbour_array_2_low[:,0]
        data    = np.ones(self.npoints)

        down_neighbour1 = np.empty(self.npoints,dtype=PETSc.IntType)

        dneigh5  =  self.height[self.neighbour_cloud[:, 0:5]].argmin(axis=1)
        dneigh10 =  self.height[self.neighbour_cloud[:, 0:10]].argmin(axis=1)
        dneigh25 =  self.height[self.neighbour_cloud[:, 0:25]].argmin(axis=1)
        dneigh50 =  self.height[self.neighbour_cloud[:, 0:50]].argmin(axis=1)

        dneigh0 = dneigh5.copy()
        dneigh0[dneigh5==0]  = dneigh10[dneigh5==0]
        dneigh0[dneigh10==0] = dneigh25[dneigh10==0]
        dneigh0[dneigh25==0] = dneigh50[dneigh25==0]

        # Now have to disentangle the lookup table part of dneigh0

        for n in range(0,self.npoints):
            down_neighbour1[n] = self.neighbour_cloud[n,dneigh0[n]]

        hit_list = np.where(dneigh50 == 0)[0]
        data[hit_list] = 0.0

        # find nodes that are their own low neighbour!
        # data[indptr[:-1] == down_neighbour1] = 0.0

        # read into matrix
        adjacency1 = self._adjacency_matrix_template()
        adjacency1.setValuesLocalCSR(indptr, down_neighbour1, data)
        adjacency1.assemblyBegin()
        adjacency1.assemblyEnd()

        self.adjacency1 = adjacency1.transpose()
        self.down_neighbour1 = down_neighbour1

## This version is based on distance not mesh connectivity -

    def _build_adjacency_matrix_2(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the 2nd steepest direction.

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.
        """

        data    = np.ones(self.npoints)
        down_neighbour2 = self.down_neighbour1.copy()
        indptr  = np.arange(0, self.npoints+1, dtype=PETSc.IntType)

        # More efficient to do this along with the 1 neighbour above

        dneigh5  =  self.height[self.neighbour_cloud[:, 0:5]].argmin(axis=1)
        dneigh10 =  self.height[self.neighbour_cloud[:, 0:10]].argmin(axis=1)
        dneigh25 =  self.height[self.neighbour_cloud[:, 0:25]].argmin(axis=1)
        dneigh50 =  self.height[self.neighbour_cloud[:, 0:50]].argmin(axis=1)

        dneigh0 = dneigh5.copy()
        dneigh0[dneigh5==0]  = dneigh10[dneigh5==0]
        dneigh0[dneigh10==0] = dneigh25[dneigh10==0]
        dneigh0[dneigh25==0] = dneigh50[dneigh25==0]

        # Could loop over points we know to be correct

        for n in range(0,self.npoints):
            candidates = np.where(self.height[self.neighbour_cloud[n, dneigh0[n]:50]] < self.height[n])[0]
            if len(candidates) > 1:
                nindx = dneigh0[n]+candidates[1]
                down_neighbour2[n] = self.neighbour_cloud[n,nindx]   ## Copy down_neighbour1, change if valid

        hit_list = np.where(dneigh50 == 0)[0]
        data[hit_list] = 0.0

        # find nodes that are their own low neighbour!
        # data[indptr[:-1] == down_neighbour1] = 0.0

        # read into matrix
        adjacency2 = self._adjacency_matrix_template()
        adjacency2.setValuesLocalCSR(indptr, down_neighbour2, data)
        adjacency2.assemblyBegin()
        adjacency2.assemblyEnd()

        self.adjacency2 = adjacency2.transpose()
        self.down_neighbour2 = down_neighbour2


    def _build_adjacency_matrix_2_old(self):
        """
        Constructs a sparse matrix to move information downhill by one node in the
        direction of the second-steepest node (self.adjacency2)

        The downhill matrix pushes information out of the domain and can be used as an increment
        to construct the cumulative area (etc) along the flow paths.
        """
        indptr  = np.arange(0, self.npoints+1, dtype=PETSc.IntType)
        down_neighbour1 = self.neighbour_array_2_low[:,0]
        down_neighbour2 = self.neighbour_array_2_low[:,1]
        indices = down_neighbour2.copy()
        data    = np.ones(self.npoints)

        # find nodes that are their own low neighbour!
        data[indptr[:-1] == down_neighbour1] = 0.0
        mask = indptr[:-1] == down_neighbour2
        indices[mask] = down_neighbour1[mask]

        # read into matrix
        adjacency2 = self._adjacency_matrix_template()
        adjacency2.setValuesLocalCSR(indptr, indices, data)
        adjacency2.assemblyBegin()
        adjacency2.assemblyEnd()

        self.adjacency2 = adjacency2.transpose()


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

        self.lvec.setArray(vector)
        self.dm.localToGlobal(self.lvec, self.gvec)

        DX0 = self.gvec.copy()
        DX1 = self.gvec.copy()
        DX1_sum = DX1.sum()

        niter = 0
        equal = False
        while not equal:
            DX1_isum = DX1_sum
            self.downhillMat.mult(DX1, self.gvec)
            DX1.setArray(self.gvec)
            DX1_sum = DX1.sum()
            DX0 += DX1

            equal = DX1_sum == DX1_isum
            niter += 1

        self.dm.globalToLocal(DX0, self.lvec)

        return self.lvec.array.copy()


    def downhill_smoothing(self, data, its, centre_weight=0.5):

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


    def uphill_smoothing(self, data, its, centre_weight=0.5):

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


    def streamwise_smoothing(self, data, its, centre_weight=0.5):
        """
        A smoothing operator that is limited to the uphill / downhill nodes for each point. It's hard to build
        a conservative smoothing operator this way since "boundaries" occur at irregular internal points associated
        with watersheds etc. Upstream and downstream smoothing operations bracket the original data (over and under,
        respectively) and we use this to find a smooth field with the same mean value as the original data. This is
        done for each application of the smoothing.
        """

        smooth_data_d = self.downhill_smoothing(data, its, centre_weight=centre_weight)
        smooth_data_u = self.uphill_smoothing(data, its, centre_weight=centre_weight)

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

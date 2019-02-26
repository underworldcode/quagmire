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
# comm = MPI.COMM_WORLD
from time import clock

try: range = xrange
except: pass


from quagmire import function as fn
from ..mesh import MeshVariable as _MeshVariable
from ..topomesh import TopoMesh as _TopoMesh

class SurfMesh(_TopoMesh):

    def __init__(self, *args, **kwargs):
        super(SurfMesh,self).__init__(*args, **kwargs)

        # self.kappa = 1.0 # dummy value

        ## Variables that are needed by default methods

        # self.rainfall_pattern_Variable = self.add_variable(name="precipitation")
        # self.sediment_distribution_Variable = self.add_variable(name="sediment")

        # new context manager ...
        self.deform_topography = self._height_update_context_manager_generator()
        self.upstream_area     = self.add_variable(name="A(x,y)", locked=True)

    ## Not sure if it is best to inherit this manager and extend it or to
    ## redefine / over-ride it. Only the exit method has changed.

    def _height_update_context_manager_generator(self):
        """Builds a context manager on the current object to control when matrices are to be updated"""

        surfmesh = self
        topographyVariable = self.topography

        class Surfmesh_Height_Update_Manager(object):
            """Manage when changes to the height information trigger a rebuild
            of the topomesh matrices and other internal data.
            """

            def __init__(inner_self):
                inner_self.surfmesh = surfmesh
                inner_self._topovar  = topographyVariable
                return

            def __enter__(inner_self):
                # unlock
                inner_self._topovar.unlock()
                return

            def __exit__(inner_self, *args):
                inner_self.surfmesh._update_height()
                inner_self.surfmesh._update_height_for_surface_flows()
                inner_self._topovar.lock()
                return

        return Surfmesh_Height_Update_Manager


    def _update_height_for_surface_flows(self):

        from time import clock

        t = clock()

        self.upstream_area.unlock()
        self.upstream_area.data = self.cumulative_flow(self.area)
        self.upstream_area.lock()

        self.timings['Upstream area'] = [clock()-t, self.log.getCPUTime(), self.log.getFlops()]

        if self.verbose:
            print(("{} - Build upstream areas {}s".format(self.dm.comm.rank, clock()-t)))

        # Find low points
        self.low_points = self.identify_low_points()

        # Find high points
        self.outflow_points = self.identify_outflow_points()



    def low_points_local_flood_fill(self, its=99999, scale=1.0, smoothing_steps=2):
        """
        Fill low points with a local flooding algorithm.
          - its is the number of uphill propagation steps
          - scale
        """

        t = clock()
        if self.rank==0 and self.verbose:
            print("Low point local flood fill")

        my_low_points = self.identify_low_points()

        h = self.heightVariable.data

        fill_height =  (h[self.neighbour_cloud[my_low_points,1:7]].mean(axis=1)-h[my_low_points])

        new_h = self.uphill_propagation(my_low_points,  fill_height, scale=scale,  its=its, fill=0.0)
        new_h = self.sync(new_h)

        smoothed_new_height = self.rbf_smoother(new_h, iterations=smoothing_steps)
        new_height = np.maximum(0.0, smoothed_new_height) + h
        new_height = self.sync(new_height)

        self._update_height_partial(new_height)
        if self.rank==0 and self.verbose:
            print("Low point local flood fill ",  clock()-t, " seconds")

        return new_height

    def low_points_local_patch_fill(self, its=1, smoothing_steps=1):

        t = clock()
        if self.rank==0 and self.verbose:
            print("Low point local patch fill")

        for iteration in range(0,its):
            low_points = self.identify_low_points()

            self.topography.unlock()

            h = self.topography.data
            delta_height = np.zeros_like(h)


            if len(low_points) != 0:
                delta_height[low_points] =  (h[self.neighbour_cloud[low_points,1:5]].mean(axis=1) -
                                                         h[low_points])
            ## Note, the smoother has a communication barrier so needs to be called even
            ## if len(low_points==0) and there is no work to do on this process
            smoothed_height = self.rbf_smoother(h+delta_height, iterations=smoothing_steps)

            self.topography.data = np.maximum(smoothed_height, h)
            self.topography.sync()
            self.topography.lock()
            self._update_height()

        if self.rank==0 and self.verbose:
            print("Low point local patch fill ",  clock()-t, " seconds")

        ## and now we need to rebuild the surface process information
        self._update_height_for_surface_flows()

        return


    def low_points_swamp_fill(self, its=1000, saddles=True, ref_height=0.0):

        import petsc4py
        from petsc4py import PETSc
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        t0 = clock()

        my_low_points = self.identify_low_points()
        my_glow_points = self.lgmap_row.apply(my_low_points.astype(PETSc.IntType))

        t = clock()
        ctmt = self.uphill_propagation(my_low_points,  my_glow_points, its=its, fill=-999999).astype(np.int)

        if self.rank==0:
            print("Build low point catchments - ", clock() - t, " seconds")

        if saddles:  # Find saddle points on the catchment edge
            cedges = np.where(ctmt[self.down_neighbour[2]] != ctmt )[0] ## local numbering
        else:        # Fine all edge points
            ctmt2 = ctmt[self.neighbour_cloud] - ctmt.reshape(-1,1)
            ctmt3 = ctmt2 * self.near_neighbour_mask
            cedges = np.where(ctmt3.any(axis=1))[0]

        outer_edges = np.where(~self.bmask)[0]
        edges = np.unique(np.hstack((cedges,outer_edges)))

        height = self.topography.data.copy()

        ## In parallel this is all the low points where this process may have a spill-point
        my_catchments = np.unique(ctmt)

        spills = np.empty((edges.shape[0]),
                         dtype=np.dtype([('c', int), ('h', float), ('x', float), ('y', float)]))

        ii = 0
        for l, this_low in enumerate(my_catchments):
            this_low_spills = edges[np.where(ctmt[edges] == this_low)]  ## local numbering

            for spill in this_low_spills:
                spills['c'][ii] = this_low
                spills['h'][ii] = height[spill]
                spills['x'][ii] = self.coords[spill,0]
                spills['y'][ii] = self.coords[spill,1]
                ii += 1

        t = clock()

        spills.sort(axis=0)  # Sorts by catchment then height ...
        s, indices = np.unique(spills['c'], return_index=True)
        spill_points = spills[indices]

        if self.rank == 0:
            print(rank, " Sort spills - ", clock() - t)

        # Gather lists to process 0, stack and remove duplicates

        t = clock()
        list_of_spills = comm.gather(spill_points,   root=0)

        if rank == 0:
            print(rank, " Gather spill data - ", clock() - t)

        if self.rank == 0:
            t = clock()

            all_spills = np.hstack(list_of_spills)
            all_spills.sort(axis=0) # Sorts by catchment then height ...
            s, indices = np.unique(all_spills['c'], return_index=True)
            all_spill_points = all_spills[indices]

            print(rank, " Sort all spills - ", clock() - t)

        else:
            all_spill_points = None
            pass

        # Broadcast lists to everyone

        global_spill_points = comm.bcast(all_spill_points, root=0)

        height2 = np.zeros_like(height) + ref_height

        for i, spill in enumerate(global_spill_points):
            this_catchment = int(spill['c'])

            ## -ve values indicate that the point is connected
            ## to the outflow of the mesh and needs no modification
            if this_catchment < 0:
                continue

            catchment_nodes = np.where(ctmt == this_catchment)
            separation_x = (self.coords[catchment_nodes,0] - spill['x'])
            separation_y = (self.coords[catchment_nodes,1] - spill['y'])
            distance = np.hypot(separation_x, separation_y)

            height2[catchment_nodes] = spill['h'] + 0.000001 * distance  # A 'small' gradient (should be a user-parameter)

        height2 = self.sync(height2)

        new_height = np.maximum(height, height2)
        new_height = self.sync(new_height)


        # We only need to update the height not all
        # surface process information that is associated with it.
        self.topography.unlock()
        self.topography.data = new_height
        self._update_height()
        self.topography.lock()


        if self.rank==0 and self.verbose:
            print("Low point swamp fill ",  clock()-t0, " seconds")

        ## but now we need to rebuild the surface process information
        self._update_height_for_surface_flows()
        return


    def backfill_points(self, fill_points, heights, its):
        """
        Handles *selected* low points by backfilling height array.
        This can be used to block a stream path, for example, or to locate lakes
        """

        if len(fill_points) == 0:
            return self.heightVariable.data

        new_height = self.lvec.duplicate()
        new_height.setArray(heights)
        height = np.maximum(self.height, new_height.array)

        # Now march the new height to all the uphill nodes of these nodes
        # height = np.maximum(self.height, delta_height.array)

        self.dm.localToGlobal(new_height, self.gvec)
        global_dH = self.gvec.copy()

        for p in range(0, its):
            self.adjacency[1].multTranspose(global_dH, self.gvec)
            global_dH.setArray(self.gvec)
            global_dH.scale(1.001)  # Maybe !
            self.dm.globalToLocal(global_dH, new_height)

            height = np.maximum(height, new_height.array)

        return height

    def uphill_propagation(self, points, values, scale=1.0, its=1000, fill=-1):

        t0 = clock()

        local_ID = self.lvec.copy()
        global_ID = self.gvec.copy()

        local_ID.set(fill+1)
        global_ID.set(fill+1)

        identifier = np.empty_like(self.topography.data)
        identifier.fill(fill+1)

        if len(points):
            identifier[points] = values + 1

        local_ID.setArray(identifier)
        self.dm.localToGlobal(local_ID, global_ID)

        delta = global_ID.copy()
        delta.abs()
        rtolerance = delta.max()[1] * 1.0e-10

        for p in range(0, its):

            # self.adjacency[1].multTranspose(global_ID, self.gvec)
            gvec = self.uphill[1] * global_ID
            delta = global_ID - gvec
            delta.abs()
            max_delta = delta.max()[1]

            if max_delta < rtolerance:
                break

            self.gvec.scale(scale)

            if self.dm.comm.Get_size() == 1:
                local_ID.array[:] = gvec.array[:]
            else:
                self.dm.globalToLocal(gvec, local_ID)

            global_ID.array[:] = gvec.array[:]

            identifier = np.maximum(identifier, local_ID.array)
            identifier = self.sync(identifier)

        # Note, the -1 is used to identify out of bounds values

        if self.rank == 0:
            print(p, " iterations, time = ", clock() - t0)

        return identifier - 1



    def identify_low_points(self, include_shadows=False):
        """
        Identify if the mesh has (internal) local minima and return an array of node indices
        """

        # from petsc4py import PETSc

        nodes = np.arange(0, self.npoints, dtype=np.int)
        gnodes = self.lgmap_row.apply(nodes.astype(PETSc.IntType))

        low_nodes = self.down_neighbour[1]
        mask = np.logical_and(nodes == low_nodes, self.bmask == True)

        if not include_shadows:
            mask = np.logical_and(mask, gnodes >= 0)

        return nodes[mask]

    def identify_global_low_points(self, global_array=False):
        """
        Identify if the mesh as a whole has (internal) local minima and return an array of local lows in global
        index format.

        If global_array is True, then lows for the whole mesh are returned
        """

        import petsc4py
        from petsc4py import PETSc
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # from petsc4py import PETSc

        nodes = np.arange(0, self.npoints, dtype=np.int)
        gnodes = self.lgmap_row.apply(nodes.astype(PETSc.IntType))

        low_nodes = self.down_neighbour[1]
        mask = np.logical_and(nodes == low_nodes, self.bmask == True)
        mask = np.logical_and(mask, gnodes >= 0)

        number_of_lows = np.count_nonzero(mask)
        low_gnodes = self.lgmap_row.apply(low_nodes.astype(PETSc.IntType))

        # gather/scatter numbers

        list_of_nlows  = comm.gather(number_of_lows,   root=0)
        if self.rank == 0:
            all_low_counts = np.hstack(list_of_nlows)
            no_global_lows0 = all_low_counts.sum()

        else:
            no_global_lows0 = None

        no_global_lows = comm.bcast(no_global_lows0, root=0)


        if global_array:

            list_of_lglows = comm.gather(low_gnodes,   root=0)

            if self.rank == 0:
                all_glows = np.hstack(list_of_lglows)
                global_lows0 = np.unique(all_glows)

            else:
                global_lows0 = None

            low_gnodes = comm.bcast(global_lows0, root=0)

        return no_global_lows, low_gnodes


    def identify_outflow_points(self):
        """
        Identify the (boundary) outflow points and return an array of (local) node indices
        """

        # nodes = np.arange(0, self.npoints, dtype=np.int)
        # low_nodes = self.down_neighbour[1]
        # mask = np.logical_and(nodes == low_nodes, self.bmask == False)
        #

        i = self.downhill_neighbours

        o = (np.logical_and(self.down_neighbour[i] == np.indices(self.down_neighbour[i].shape), self.bmask == False)).ravel()
        outflow_nodes = o.nonzero()[0]

        return outflow_nodes


    def identify_flat_spots(self):

        smooth_grad1 = self.local_area_smoothing(self.slopeVariable.data, its=1, centre_weight=0.5)

        # flat_spot_field = np.where(smooth_grad1 < smooth_grad1.max()/100, 0.0, 1.0)

        flat_spots = np.where(smooth_grad1 < smooth_grad1.max()/1000.0, True, False)

        return flat_spots


    def stream_power_erosion_deposition_rate_old(self, efficiency=0.01, smooth_power=3, \
                                             smooth_low_points=2, smooth_erosion_rate=2, \
                                             smooth_deposition_rate=2, smooth_operator=None,
                                             centre_weight_u=0.5, centre_weight=0.5):

        """
        Function of the SurfaceProcessMesh which computes stream-power erosion and deposition rates
        from a given rainfall pattern (self.rainfall_pattern).

        In this model we assume a the carrying capacity of the stream is related to the stream power and so is the
        erosion rate. The two are related to one another in this particular case by a single contant (everywhere on the mesh)
        This does not allow for spatially variable erodability and it does not allow for differences in the dependence
        of erosion / deposition on the stream power.

        Deposition occurs such that the upstream-integrated eroded sediment does not exceed the carrying capacity at a given
        point. To conserve mass, we have to treat internal drainage points carefully and, optionally, smooth the deposition
        upstream of the low point. We also have to be careful when stream-power and carrying capacity increase going downstream.
        This produces a negative deposition rate when the flow is at capacity. We suppress this behaviour and balance mass across
        all other deposition sites but this does mean the capacity is not perfectly satisfied everywhere.

        parameters:
         efficiency=0.01          : erosion rate for a given stream power compared to carrying capacity
         smooth_power=3           : upstream / downstream smoothing of the stream power (number of cycles of smoothing)
         smooth_low_points=3      : upstream smoothing of the deposition at low points (number of cycles of smoothing)
         smooth_erosion_rate=0    : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)
         smooth_deposition_rate=0 : upstream / downstream smoothing of the computed erosion rate (number of cycles of smoothing)

        """


        if smooth_operator == None:
            smooth_operator = self.streamwise_smoothing

        # Calculate stream power

        ## Model 1 - Local equilibrium

        rainflux = self.rainfall_pattern
        rainfall = self.area * rainflux
        cumulative_rain = self.cumulative_flow(rainfall)

        cumulative_flow_rate = cumulative_rain / self.area

        stream_power = self.uphill_smoothing(cumulative_flow_rate * self.slopeVariable.data, smooth_power, centre_weight=centre_weight_u)

        #  predicted erosion rate from stream power * efficiency
        #  maximum sediment that can be transported is limited by the local carrying capacity (assume also prop to stream power)
        #  whatever cannot be passed on has to be deposited

        erosion_rate = self.streamwise_smoothing(efficiency * stream_power, smooth_erosion_rate, centre_weight=centre_weight)
        full_capacity_sediment_flux = stream_power
        full_capacity_sediment_load = stream_power * self.area
        cumulative_eroded_material = self.cumulative_flow(self.area * erosion_rate)

        # But this can exceed the carrying capacity

        transport_limited_eroded_material = np.minimum(cumulative_eroded_material, full_capacity_sediment_load)
        transport_limited_erosion_rate = transport_limited_eroded_material / self.area

        # And this therefore implies a deposition rate which reduces the total sediment in the system to capacity
        # Calculate this by substracting the deposited amounts from the excess integrated flow. We could then iterate
        # to compute the new erosion rates etc, but here we just spread the sediments around to places where
        # the deposition is positive

        excess = self.gvec.duplicate()
        deposition = self.lvec.duplicate()
        self.lvec.setArray(cumulative_eroded_material - transport_limited_eroded_material)
        self.dm.localToGlobal(self.lvec, excess)
        self.downhillMat.mult(excess, self.gvec)
        self.dm.globalToLocal(excess - self.gvec, deposition)
        depo_sum = deposition.sum()


        # Now rebalance the fact that we have clipped off the negative deposition which will need
        # to be clawed back downstream (ideally, but for now we can just make a global correction)

        deposition = np.clip(deposition.array, 0.0, 1.0e99)
        deposition *= depo_sum / (depo_sum + 1e-12)


        # The (interior) low points are a bit of a problem - we stomped on the stream power there
        # but this produces a very lumpy deposition at the low point itself and this could (does)
        # make the numerical representation pretty unstable. Instead what we can do is to take that
        # deposition at the low points let it spill into the local area


        ## These will instead be handled by a specific routine "handle_low_points" which is
        ## done once the height has been updated

        if len(self.low_points):
            deposition[self.low_points] = 0.0

        # The flat regions in the domain are also problematic since the deposition there is

        flat_spots = self.identify_flat_spots()

        if len(flat_spots):
            smoothed_deposition = deposition.copy()
            smoothed_deposition[np.invert(flat_spots)] = 0.0
            smoothed_deposition = self.local_area_smoothing(smoothed_deposition, its=2, centre_weight=0.5)
            deposition[flat_spots] = smoothed_deposition[flat_spots]

        deposition_rate = smooth_operator(deposition, smooth_deposition_rate, centre_weight=centre_weight) / self.area

        return erosion_rate, deposition_rate, stream_power



    # def landscape_diffusion_critical_slope(self, kappa, critical_slope, fluxBC):
    #     '''
    #     Non-linear diffusion to keep slopes at a critical value. Assumes a background
    #     diffusion rate (can be a vector of length mesh.tri.npoints) and a critical slope value.
    #
    #     This term is suitable for the sloughing of sediment from hillslopes.
    #
    #     To Do: The critical slope should be a function of the material (sediment, basement etc)
    #     but currently it is not.
    #
    #     To Do: The fluxBC flag is global ... it should apply to the outward normal
    #     at selected nodes but currently it is set to kill both fluxes at all boundary nodes.
    #     '''
    #
    #     inverse_bmask = np.invert(self.bmask)
    #
    #     kappa_eff = kappa / (1.01 - (np.clip(self.slopeVariable.data,0.0,critical_slope) / critical_slope)**2)
    #     self.kappa = kappa_eff
    #
    #     # get minimum timestep across the global mesh
    #     local_diffusion_timestep = np.array((self.area / kappa_eff).min())
    #     global_diffusion_timestep = np.array(0.0)
    #     self.comm.Allreduce([local_diffusion_timestep, MPI.DOUBLE], \
    #                         [global_diffusion_timestep, MPI.DOUBLE], op=MPI.MIN)
    #
    #
    #     fluxVariable = self.heightVariable.gradient(nit=3, tol=1e-3)
    #     fluxVariable.data *= kappa_eff.reshape(-1,1)
    #     if fluxBC:
    #         fluxVariable.data[inverse_bmask] = 0.0 # outward normal flux, actually
    #
    #     diffDz = self.derivative_div(*fluxVariable.data.T, nit=3, tol=1e-3)
    #     diffDz = self.sync(diffDz)
    #
    #     if not fluxBC:
    #         diffDz[inverse_bmask] = 0.0
    #
    #     return diffDz, global_diffusion_timestep
    #


    def landscape_diffusion_critical_slope(self, kappa, critical_slope, fluxBC):
        '''
        Non-linear diffusion to keep slopes at a critical value. Assumes a background
        diffusion rate (can be a vector of length mesh.tri.npoints) and a critical slope value.
        This term is suitable for the sloughing of sediment from hillslopes.
        To Do: The critical slope should be a function of the material (sediment, basement etc)
        but currently it is not.
        To Do: The fluxBC flag is global ... it should apply to the outward normal
        at selected nodes but currently it is set to kill both fluxes at all boundary nodes.
        '''

        inverse_bmask = np.invert(self.bmask)

        kappa_eff = kappa / (1.01 - (np.clip(self.slopeVariable.data,0.0,critical_slope) / critical_slope)**2)
        self.kappa = kappa_eff
        diff_timestep   =  self.area.min() / kappa_eff.max()

        # get minimum timestep across the global mesh
        local_diffusion_timestep = np.array((self.area / kappa_eff).min())
        global_diffusion_timestep = np.array(0.0)
        self.comm.Allreduce([local_diffusion_timestep, MPI.DOUBLE], \
                            [global_diffusion_timestep, MPI.DOUBLE], op=MPI.MIN)


        gradZx, gradZy = self.derivative_grad(self.heightVariable.data)
        gradZx = self.sync(gradZx)
        gradZy = self.sync(gradZy)
        flux_x = kappa_eff * gradZx
        flux_y = kappa_eff * gradZy

        if fluxBC:
            flux_x[inverse_bmask] = 0.0
            flux_y[inverse_bmask] = 0.0  # outward normal flux, actually

        diffDz = self.derivative_div(flux_x, flux_y)
        diffDz = self.sync(diffDz)

        if not fluxBC:
            diffDz[inverse_bmask] = 0.0

        return diffDz, global_diffusion_timestep



    def landscape_evolution_timestep(self, diffusion_rate, erosion_rate, deposition_rate, uplift_rate):
        """
        Calculate the change in topography for one timestep
        """

        time = 0.0
        typical_l = np.sqrt(self.area)
        critical_slope = 50.0

        slope = np.maximum(self.slopeVariable.data, critical_slope)

        erosion_deposition_rate = deposition_rate - erosion_rate

        erosion_timestep    = (self.slopeVariable.data*typical_l/(erosion_rate + 1e-12)).min()
        deposition_timestep = (self.slopeVariable.data*typical_l/(deposition_rate + 1e-12)).min()
        diffusion_timestep  = self.area.min()/np.max(self.kappa)

        local_timestep = np.array(min(erosion_timestep, deposition_timestep, diffusion_timestep))
        timestep = np.array(0.0)
        comm.Allreduce([local_timestep, MPI.DOUBLE], [timestep, MPI.DOUBLE], op=MPI.MIN)

        delta_h = timestep * (erosion_deposition_rate - diffusion_rate + uplift_rate)

        # Note this is based on local information, and must be synced
        return delta_h, timestep




    def stream_power_erosion_deposition_rate_local(self, stream_power,
                                             efficiency=0.01,
                                             smoothOperator=None,
                                             smoothPowerIts=1, smoothDepoIts=1, smoothLowIts=3 ):

        """
        Function of the SurfaceProcessMesh which computes stream-power erosion and deposition rates
        from a given rainfall pattern (self.rainfall_pattern).

        In this model we assume a the carrying capacity of the stream is related to the stream power and so is the
        erosion rate. The two are related to one another in this particular case by a single contant (everywhere on the mesh)
        This does not allow for spatially variable erodability and it does not allow for differences in the dependence
        of erosion / deposition on the stream power.

        Deposition occurs such that the upstream-integrated eroded sediment does not exceed the carrying capacity at a given
        point. To conserve mass, we have to treat internal drainage points carefully and, optionally, smooth the deposition
        upstream of the low point. We also have to be careful when stream-power and carrying capacity increase going downstream.
        This produces a negative deposition rate when the flow is at capacity. We suppress this behaviour and balance mass across
        all other deposition sites but this does mean the capacity is not perfectly satisfied everywhere.

        parameters:
         efficiency=0.01          : erosion rate for a given stream power compared to carrying capacity
         smoothOperator=None      :
         smoothPowerIts=1         : Use smoothing function (smoothOperator) n times on stream power
         smoothDepoIts=1          : Use smoothing function (smoothOperator) n times on deposition rate
         smoothLowIts=1           : Use smoothing function (smoothOperator) n times on low point patches
        """


        if smooth_operator == None:
            smooth_operator = self.rbf_smoother

        for i in range(0, smoothPowerIts):
            stream_power = smoothOperator(stream_power)

        dhdt_erosion = efficiency*stream_power

        cumulative_eroded_material = self.cumulative_flow(erosion_rate*self.area)
        full_capacity_sediment_load = stream_power   # Constant ?

        # if the sediment load exceeds the capacity, start depositing material
        # use vec.pointwisemin
        transport_limited_eroded_material = np.minimum(cumulative_eroded_material, full_capacity_sediment_load)

        excess = sp.gvec.duplicate()
        excess.setArray(cumulative_eroded_material - transport_limited_eroded_material)

        deposition = excess - self.downhillMat*excess
        depo_sum   = deposition.sum()




        # Now rebalance the fact that we have clipped off the negative deposition which will need
        # to be clawed back downstream (ideally, but for now we can just make a global correction)

        deposition = np.clip(deposition.array, 0.0, 1.0e99)
        deposition *= depo_sum / (depo_sum + 1e-12)


        # The (interior) low points are a bit of a problem - we stomped on the stream power there
        # but this produces a very lumpy deposition at the low point itself and this could (does)
        # make the numerical representation pretty unstable. Instead what we can do is to take that
        # deposition at the low points let it spill into the local area


        ## These will instead be handled by a specific routine "handle_low_points" which is
        ## done once the height has been updated

        if len(self.low_points):
            deposition[self.low_points] = 0.0

        # The flat regions in the domain are also problematic since the deposition there is

        flat_spots = self.identify_flat_spots()

        if len(flat_spots):
            smoothed_deposition = deposition.copy()
            smoothed_deposition[np.invert(flat_spots)] = 0.0
            smoothed_deposition = self.local_area_smoothing(smoothed_deposition, its=2, centre_weight=0.5)
            deposition[flat_spots] = smoothed_deposition[flat_spots]

        deposition_rate = smooth_operator(deposition, smooth_deposition_rate, centre_weight=centre_weight) / self.area

        return erosion_rate, deposition_rate, stream_power

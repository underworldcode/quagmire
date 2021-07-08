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


from quagmire import function as _fn


import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from petsc4py import PETSc

# To do ... an interface for (iteratively) dealing with
# boundaries with normals that are not aligned to the coordinates

class DiffusionEquation(object):

    ## it is helpful to have a unique ID for each instance so we
    ## can autogenerate unique names if we wish

    __count = 0

    @classmethod
    def _count(cls):
        DiffusionEquation.__count += 1
        return DiffusionEquation.__count

    @property
    def id(self):
        return self.__id

    def __init__(self,
                 mesh=None,
                 diffusivity_fn=None,
                 dirichlet_mask=None,
                 neumann_x_mask=None,
                 neumann_y_mask=None,
                 non_linear=False,
                 theta=0.5 ):

        self.__id = "diffusion_solver_{}".format(self._count())
        self.mesh = mesh

        # These we initialise to None first of all
        self._diffusivity = None
        self._neumann_x_mask = None
        self._neumann_y_mask = None
        self._dirichlet_mask = None
        self._non_linear = False
        self.theta = theta

        if diffusivity_fn is not None:
            self.diffusivity = diffusivity_fn

        if neumann_x_mask is not None:
            self.neumann_x_mask = neumann_x_mask

        if neumann_y_mask is not None:
            self.neumann_y_mask = neumann_y_mask

        if dirichlet_mask is not None:
            self.dirichlet_mask = dirichlet_mask


        # create some global work vectors
        self._RHS = mesh.dm.createGlobalVec()
        self._PHI = mesh.dm.createGlobalVec()
        self._RHS_outer = mesh.dm.createGlobalVec()

        # store this flag to make sure verify has been called before timestepping
        self._verified = False


    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, meshobject):
        self._mesh = meshobject
        self._phi   = meshobject.add_variable(name="phi_{}".format(self.id))
        return

    @property
    def phi(self):
        return self._phi
    # No setter for this ... it is defined via the mesh.setter

    @property
    def diffusivity(self):
        return self._diffusivity

    @diffusivity.setter
    def diffusivity(self, fn_object):
        self._diffusivity = fn_object

        if _fn.check_dependency(self._diffusivity, self._phi):
            # Set the flag directly, bypassing .setter checks
            self._non_linear = True

        return

    @property
    def neumann_x_mask(self):
        return self._neumann_x_mask

    @neumann_x_mask.setter
    def neumann_x_mask(self, fn_object):
        _fn.check_object_is_a_q_function_and_raise(fn_object)
        self._neumann_x_mask = fn_object
        return

    @property
    def neumann_y_mask(self):
        return self._neumann_y_mask

    @neumann_y_mask.setter
    def neumann_y_mask(self, fn_object):
        _fn.check_object_is_a_q_function_and_raise(fn_object)
        self._neumann_y_mask = fn_object
        return

    @property
    def dirichlet_mask(self):
        return self._dirichlet_mask

    @dirichlet_mask.setter
    def dirichlet_mask(self, fn_object):
        _fn.check_object_is_a_q_function_and_raise(fn_object)
        self._dirichlet_mask = fn_object
        return

    @property
    def non_linear(self):
        return self._non_linear

    @non_linear.setter
    def non_linear(self, bool_object):
        # Warn if non-linearity exists
        # This flag is set whenever the diffusivity function is changed
        # and will need to be over-ridden each time.
        self._non_linear = bool_object
        if _fn.check_dependency(self._diffusivity, self._phi):
            print("Over-riding non-linear solver path despite diffusivity being non-linear")

        return

    @property
    def theta(self):
        return self._theta
    @theta.setter
    def theta(self, val):
        assert val <= 1 and val >= 0, "theta must be within the range [0,1]"
        self._theta = float(val)

        # trigger an update to the LHS / RHS matrices?
    

    def construct_matrix(self):

        mesh = self._mesh
        kappa = self.diffusivity.evaluate(mesh)

        nnz = mesh.natural_neighbours_count.astype(PETSc.IntType)

        # initialise matrix object
        mat = PETSc.Mat().create(comm=comm)
        mat.setType('aij')
        mat.setSizes(mesh.sizes)
        mat.setLGMap(mesh.lgmap_row, mesh.lgmap_col)
        mat.setFromOptions()
        mat.setPreallocationNNZ(nnz)
        # mat.setOption(mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        area = mesh.pointwise_area.evaluate(mesh)
        neighbour_area = area[mesh.natural_neighbours]
        neighbour_area[~mesh.natural_neighbours_mask] = 0.0 # mask non-neighbours
        total_neighbour_area = neighbour_area.sum(axis=1)


        # vectorise along the matrix stencil
        mat.assemblyBegin()

        nodes = np.arange(0, mesh.npoints+1, dtype=PETSc.IntType)

        # need this to prevent mallocs
        mat.setValuesLocalCSR(nodes, nodes[:-1], np.zeros(mesh.npoints))

        for col in range(1, mesh.natural_neighbours.shape[1]):
            node_neighbours = mesh.natural_neighbours[:,col].astype(PETSc.IntType)

            node_distance = np.linalg.norm(mesh.data - mesh.data[node_neighbours], axis=1)
            node_distance[node_distance == 0] += 0.000001 # do not divide by zero
            weight = 3*area[node_neighbours]/(total_neighbour_area[node_neighbours]/3)

            vals = weight*0.5*(kappa + kappa[node_neighbours])/node_distance**2

            mat.setValuesLocalCSR(nodes, node_neighbours, vals)

        mat.assemblyEnd()

        diag = mat.getRowSum()
        diag.scale(-1.0)
        mat.setDiagonal(diag)

        ## NOTE: This matrix can be used to solve steady state diffusion
        ## we augment it for timestepping (and to specify Dirichlet BCs)

        return mat


    def _set_boundary_conditions(self, matrix, rhs):

        mesh = self._mesh

        dirichlet_mask = self.dirichlet_mask.evaluate(mesh).astype(bool)
        # neumann_mask   = self.neumann_x_mask.evaluate(mesh).astype(bool)
        # neumann_mask  += self.neumann_y_mask.evaluate(mesh).astype(bool)

        if dirichlet_mask.any():
            # augment the matrix

            # put zeros where we have the Dirichlet mask
            mesh.lvec.setArray(np.invert(dirichlet_mask).astype(np.float))
            mesh.dm.localToGlobal(mesh.lvec, mesh.gvec)
            matrix.diagonalScale(mesh.gvec)

            # now put ones along the diagonal
            diag = matrix.getDiagonal()
            mesh.dm.globalToLocal(diag, mesh.lvec)
            lvec_data = mesh.lvec.array
            lvec_data[dirichlet_mask] = 1.0
            mesh.lvec.setArray(lvec_data)
            mesh.dm.localToGlobal(mesh.lvec, diag)
            matrix.setDiagonal(diag)


            # set the RHS vector
            lvec_data.fill(0.0)
            lvec_data[dirichlet_mask] = self.phi.data[dirichlet_mask]
            mesh.lvec.setArray(lvec_data)
            mesh.dm.localToGlobal(mesh.lvec, rhs)


        # neumann BCs are by design of the matrix,
        # so nothing needs to be done with those masks.


    def _initialise_solver(self, matrix, ksp_type='gmres', atol=1e-20, rtol=1e-50, pc_type=None, **kwargs):
        """
        Initialise linear solver object
        """

        ksp = PETSc.KSP().create(comm)
        ksp.setType(ksp_type)
        ksp.setOperators(matrix)
        ksp.setTolerances(atol, rtol)
        if pc_type is not None:
            pc = ksp.getPC()
            pc.setType(pc_type)
        ksp.setFromOptions()
        self.ksp = ksp


    def verify(self, **kwargs):
        """Verify solver is ready for launch"""

        # Check to see we have provided everything in the correct form

        # store the matrix
        self.matrix = self.construct_matrix()

        self._verified = True
        return


    def diffusion_rate_fn(self, lazyFn):
        ## !! create stand alone function

        from quagmire import function as fn

        gradl = self._mesh.geometry.grad(lazyFn)
        kappa_dx_fn  = fn.misc.where(self.neumann_x_mask,
                                     self.diffusivity  * gradl[0],
                                     fn.parameter(0.0))
        kappa_dy_fn  = fn.misc.where(self.neumann_y_mask,
                                     self.diffusivity  * gradl[1],
                                     fn.parameter(0.0))

        dPhi_dt_fn = fn.misc.where(self.dirichlet_mask, \
                                   self._mesh.geometry.div(fn.vector_field(kappa_dx_fn, kappa_dy_fn)), \
                                   fn.parameter(0.0))

        return dPhi_dt_fn


    def diffusion_timestep(self):

        local_diff_timestep = (self._mesh.area / self._diffusivity.evaluate(self._mesh)).min()

        # synchronise ...

        local_diff_timestep = np.array(local_diff_timestep)
        global_diff_timestep = np.array(0.0)
        comm.Allreduce([local_diff_timestep, MPI.DOUBLE], [global_diff_timestep, MPI.DOUBLE], op=MPI.MIN)

        return global_diff_timestep


    def _get_scaled_matrix(self, scale):
        mat = self.matrix.copy()
        mat.scale(scale)
        diag = mat.getDiagonal()
        diag += 1.0
        mat.setDiagonal(diag)
        return mat



    def time_integration(self, timestep, steps=1, Delta_t=None, feedback=None):

        if Delta_t is not None:
            steps = Delta_t // timestep
            timestep = Delta_t / steps

        if not self._verified:
            self.verify()


        # Create LHS and RHS matrices
        scale_LHS = 0.5*timestep*self.theta
        scale_RHS = 0.5*timestep*(1.0 - self.theta)

        mat_LHS = self._get_scaled_matrix(-scale_LHS) # LHS matrix
        mat_RHS = self._get_scaled_matrix(scale_RHS) # RHS matrix


        PHI = self._PHI
        RHS = self._RHS
        BCS = self._RHS_outer

        # set boundary conditions
        self._set_boundary_conditions(mat_LHS, BCS)
        self._set_boundary_conditions(mat_RHS, BCS)
        BCS.set(0.0)

        dirichlet_mask = self.dirichlet_mask.evaluate(self._mesh).astype(bool)
        dirichlet_BC = self.phi.data[dirichlet_mask].copy()


        # initialise solver
        self._initialise_solver(mat_LHS)

        self._mesh.dm.localToGlobal(self.phi._ldata, PHI)

        elapsed_time = 0.0
        
        for step in range(0, int(steps)):

            # enforce Dirichlet BCs
            self._mesh.dm.globalToLocal(PHI, self._mesh.lvec)
            self._mesh.lvec.array[dirichlet_mask] = dirichlet_BC
            self._mesh.dm.localToGlobal(self._mesh.lvec, PHI)


            # evaluate right hand side
            mat_RHS.mult(PHI, RHS)
            RHS += BCS # add BCs

            # find solution inside domain
            self.ksp.solve(RHS, PHI)

            elapsed_time += timestep

            if feedback is not None and step%feedback == 0 or step == steps:
                print("{:05d} - t = {:.3g}".format(step, elapsed_time))


        self._mesh.dm.globalToLocal(PHI, self.phi._ldata)

        return steps, elapsed_time


    def steady_state(self, feedback=None):

        if not self._verified:
            self.verify()

        mat_LHS = self.matrix.copy()
        RHS = self._RHS
        PHI = self._PHI


        # set boundary conditions
        self._set_boundary_conditions(mat_LHS, RHS)

        # initialise solver
        self._initialise_solver(mat_LHS)


        self._mesh.dm.localToGlobal(self.phi._ldata, PHI)


        self.ksp.solve(RHS, PHI)
        self._mesh.dm.globalToLocal(PHI, self.phi._ldata)
        return self.phi.data



    def time_integration_explicit(self, timestep, steps=1, Delta_t=None, feedback=None):

        from quagmire import function as fn

        if Delta_t is not None:
            steps = Delta_t // timestep
            timestep = Delta_t / steps

        elapsed_time = 0.0

        for step in range(0, int(steps)):

            ## Non-linear loop will need to go here
            ## and update timestep somehow.

            dPhi_dt_fn   = self.diffusion_rate_fn(self.phi)


            phi0 = self.phi.copy()
            self.phi.data = self.phi.data  +  0.5 * timestep * dPhi_dt_fn.evaluate(self._mesh)
            self.phi.data = phi0.data +  timestep * dPhi_dt_fn.evaluate(self._mesh)

            elapsed_time += timestep

            if feedback is not None and step%feedback == 0 or step == steps:
                print("{:05d} - t = {:.3g}".format(step, elapsed_time))



        return steps, elapsed_time

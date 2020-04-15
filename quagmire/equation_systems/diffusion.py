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
                 non_linear=False ):

        self.__id = "diffusion_solver_{}".format(self._count())
        self.mesh = mesh

        # These we initialise to None first of all
        self._diffusivity = None
        self._neumann_x_mask = None
        self._neumann_y_mask = None
        self._dirichlet_mask = None
        self._non_linear = False

        if diffusivity_fn is not None:
            self.diffusivity = diffusivity_fn

        if neumann_x_mask is not None:
            self.neumann_x_mask = neumann_x_mask

        if neumann_y_mask is not None:
            self.neumann_y_mask = neumann_y_mask

        if dirichlet_mask is not None:
            self.dirichlet_mask = dirichlet_mask


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



    def verify(self):
        """Verify solver is ready for launch"""

        # Check to see we have provided everything in the correct form

        return


    def diffusion_rate_fn(self, lazyFn):
        ## !! create stand alone function

        from quagmire import function as fn

        dx_fn, dy_fn = fn.math.grad(lazyFn)
        kappa_dx_fn  = fn.misc.where(self.neumann_x_mask,
                                     self.diffusivity  * dx_fn,
                                     fn.parameter(0.0))
        kappa_dy_fn  = fn.misc.where(self.neumann_y_mask,
                                     self.diffusivity  * dy_fn,
                                     fn.parameter(0.0))

        dPhi_dt_fn   = fn.misc.where(self.dirichlet_mask, fn.math.div(kappa_dx_fn, kappa_dy_fn), fn.parameter(0.0))

        return dPhi_dt_fn


    def diffusion_timestep(self):

        local_diff_timestep = (self._mesh.area / self._diffusivity.evaluate(self._mesh)).min()

        # synchronise ...

        local_diff_timestep = np.array(local_diff_timestep)
        global_diff_timestep = np.array(0.0)
        comm.Allreduce([local_diff_timestep, MPI.DOUBLE], [global_diff_timestep, MPI.DOUBLE], op=MPI.MIN)

        return global_diff_timestep



    def time_integration(self, timestep, steps=1, Delta_t=None, feedback=None):

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

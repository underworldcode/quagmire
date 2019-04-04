"""
Copyright 2016-2019 Louis Moresi, Ben Mather, Romain Beucher

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
comm = MPI.COMM_WORLD

from .. import function as fn

# To do ... an interface for (iteratively) dealing with
# boundaries with normals that are not aligned to the coordinates

class ErosionDepositionEquation(object):

    ## it is helpful to have a unique ID for each instance so we
    ## can autogenerate unique names if we wish

    __count = 0

    @classmethod
    def _count(cls):
        ErosionDepositionEquation.__count += 1
        return ErosionDepositionEquation.__count

    def __init__(self):
        self.__id = self._count()

    @property
    def id(self):
        return self.__id

    def __init__(self,
                 mesh=None,
                 rainfall_fn=None,
                 m=1.0,
                 n=1.0 ):

        self.__id = self._count()

        # These should become properties ...


        self.mesh = mesh
        self.rainfall = rainfall_fn
        self.m = m
        self.n = n

        # This one can be generated and regenerated if mesh changes


    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, meshobject):
        self._mesh = meshobject
        self._sp   = meshobject.add_variable(name="qs_{}".format(self.id))
        self._ed   = meshobject.add_variable(name="ed_{}".format(self.id))
        return

    @property
    def stream_power(self):
        return self._sp
    # No setter for this ... it is defined via the mesh.setter
    @property
    def erosion_deposition(self):
        return self._ed
    # No setter for this ... it is defined via the mesh.setter

    @property
    def rainfall(self):
        return self._rainfall

    @rainfall.setter
    def rainfall(self, fn_object):
        # Should verify it is a function
        self._rainfall = fn_object
        return

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, scalar):
        # Should verify it is a function
        self._m = scalar
        return

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, fn_object):
        # Should verify it is a function
        self._n = fn_object
        return

    def verify(self):

        # Check to see we have provided everything in the correct form

        return


    def stream_power_law(self):
        """
        Compute the stream power

        qs = UpInt(rainfall)^m * (grad H(x,y))^2
        """
        rainfall_fn = self._rainfall
        m = self._m
        n = self._n

        # integrate upstream rainfall
        upstream_precipitation_integral_fn = self._mesh.upstream_integral_fn(rainfall_fn)

        # create stream power function
        stream_power_fn = upstream_precipitation_integral_fn**m * self._mesh.slope**n * self._mesh.mask

        self.stream_power.data = stream_power_fn.evaluate(self._mesh)
        return self.stream_power


    def erosion_deposition_model_1(self, efficiency):
        """
        Local equilibrium model
        """
        rainfall_fn = self._rainfall
        m = self._m
        n = self._n

        # integrate upstream rainfall
        upstream_precipitation_integral_fn = self._mesh.upstream_integral_fn(rainfall_fn)

        # create stream power function
        stream_power_fn = upstream_precipitation_integral_fn**m * self._mesh.slope**n * self._mesh.mask

        # we always want to store stream power (?)
        # self.stream_power.data = stream_power_fn.evaluate(self._mesh)







        stream_power = self.stream_power_law()

        erosion_rate_fn = efficiency*stream_power

        full_capacity_sediment_load = stream_power



        return

    def erosion_deposition_model_2(self):
        """
        Saltation length
        """
        return

    def erosion_deposition_model_3(self, efficiency, alpha):
        """
        Transport-limited
        """

        rainfall_fn = self._rainfall
        m = self._m
        n = self._n
        area = self._mesh.pointwise_area

        # integrate upstream / calculate rate
        upstream_precipitation_integral_fn = self._mesh.upstream_integral_fn(rainfall_fn)
        upstream_precipitation_rate_fn = upstream_precipitation_integral_fn / area

        # construct stream power function
        stream_power_fn = upstream_precipitation_integral_fn**m * self._mesh.slope**n * self._mesh.mask

        # only some is eroded
        erosion_rate_fn = stream_power_fn*efficiency

        # integrate upstream / calculate rate
        upstream_eroded_material_integral_fn = self._mesh.upstream_integral_fn(erosion_rate_fn)
        upstream_eroded_material_rate_fn = upstream_eroded_material_integral_fn / area

        deposition_rate_fn = upstream_precipitation_rate_fn / (alpha * upstream_eroded_material_rate_fn)


        dHdt_fn = -erosion_rate_fn + deposition_rate_fn

        self.erosion_deposition.data = dHdt_fn.evaluate(self._mesh)
        return


    def diffusion_timestep(self):

        local_diff_timestep = (0.5 * self._mesh.area / self._diffusivity.evaluate(self._mesh)).min()

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

            dx_fn, dy_fn = fn.math.grad(self.phi)
            kappa_dx_fn  = fn.misc.where(self.neumann_x_mask,
                                         self.diffusivity  * dx_fn,
                                         fn.parameter(0.0))
            kappa_dy_fn  = fn.misc.where(self.neumann_y_mask,
                                         self.diffusivity  * dy_fn,
                                         fn.parameter(0.0))

            dPhi_dt_fn   = fn.misc.where(self.dirichlet_mask, fn.math.div(kappa_dx_fn, kappa_dy_fn), fn.parameter(0.0))


            phi0 = self.phi.copy()
            self.phi.data = self.phi.data  +  0.5 * timestep * dPhi_dt_fn.evaluate(self._mesh)
            self.phi.data = phi0.data +  timestep * dPhi_dt_fn.evaluate(self._mesh)

            elapsed_time += timestep

            if feedback is not None and step%feedback == 0 or step == steps:
                print("{:05d} - t = {:.3g}".format(step, elapsed_time))



        return steps, elapsed_time

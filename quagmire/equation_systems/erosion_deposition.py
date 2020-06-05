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


import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from quagmire.function import LazyEvaluation as _LazyEvaluation

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
        self._erosion_rate   = meshobject.add_variable(name="qs_{}".format(self.id))
        self._deposition_rate   = meshobject.add_variable(name="ed_{}".format(self.id))
        return

    @property
    def erosion_rate(self):
        return self._erosion_rate
    # No setter for this ... it is defined via the mesh.setter
    @property
    def deposition_rate(self):
        return self._deposition_rate
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

    @property
    def erosion_deposition_fn(self):
        return self._erosion_deposition_fn

    @erosion_deposition_fn.setter
    def erosion_deposition_fn(self, fn_object):
        if hasattr(self, fn_object):
            self._erosion_deposition_fn = fn_object
        else:
            raise ValueError("Choose a valid erosion-deposition function")
    

    def verify(self):

        # Check to see we have provided everything in the correct form

        return


    def stream_power_fn(self):
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
        stream_power_law_fn = upstream_precipitation_integral_fn**m * self._mesh.slope**n * self._mesh.mask
        return stream_power_law_fn


## Built-in erosion / deposition functions

    def erosion_deposition_local_equilibrium(self, efficiency):
        """
        Local equilibrium model
        """

        stream_power_fn = self.stream_power_fn()
        erosion_rate_fn = efficiency*stream_power_fn

        # store erosion rate so we do not have to evaluate it
        # again to compute the deposition rate
        erosion_rate = self._erosion_rate
        erosion_rate.unlock()
        erosion_rate.data = erosion_rate_fn.evaluate(self._mesh)
        erosion_rate.lock()

        deposition_rate_fn = self._mesh.upstream_integral_fn(erosion_rate)

        # might as well store deposition rate too
        deposition_rate = self._deposition_rate
        deposition_rate.unlock()
        deposition_rate.data = deposition_rate_fn.evaluate(self._mesh)
        deposition_rate.lock()

        # erosion_deposition_fn = deposition_rate - erosion_rate
        return erosion_rate.data, deposition_rate.data


    def fn_local_equilibrium(self, efficiency):

        import quagmire

        def new_fn(*args, **kwargs):
            
            erosion_rate, deposition_rate = self.erosion_deposition_local_equilibrium(efficiency)
            dHdt = deposition_rate - erosion_rate
            # dHdt = self._mesh.sync(dHdt)

            if len(args) == 1 and args[0] == self._mesh:
                return dHdt
            elif len(args) == 1 and quagmire.mesh.check_object_is_a_q_mesh_and_raise(args[0]):
                mesh = args[0]
                return self.interpolate(mesh.coords[:,0], mesh.coords[:,1], zdata=dHdt, **kwargs)
            else:
                xi = np.atleast_1d(args[0])  # .resize(-1,1)
                yi = np.atleast_1d(args[1])  # .resize(-1,1)
                i, e = self.interpolate(xi, yi, zdata=dHdt, **kwargs)
                return i

        newLazyFn = _LazyEvaluation(mesh=self._mesh)
        newLazyFn.evaluate = new_fn
        newLazyFn.description = "dH/dt"
        newLazyFn.dependency_list = set([self.erosion_rate, self.deposition_rate])

        return newLazyFn


    def erosion_deposition_saltation_length(self):
        """
        Saltation length
        """
        raise NotImplementedError("Check back again soon!")


    def erosion_deposition_transport_limited_flow(self, efficiency, alpha):
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

        erosion_rate = self._erosion_rate
        erosion_rate.unlock()
        erosion_rate.data = erosion_rate_fn.evaluate(self._mesh)
        erosion_rate.lock()

        deposition_rate = self._deposition_rate
        deposition_rate.unlock()
        deposition_rate.data = deposition_rate_fn.evaluate(self._mesh)
        deposition_rate.lock()

        return erosion_rate, deposition_rate


    def erosion_deposition_timestep(self):

        from quagmire import function as fn

        # mesh variables
        erosion_rate = self._erosion_rate
        deposition_rate = self._deposition_rate
        slope = self._mesh.slope

        # protect against dividing by zero
        min_slope = fn.parameter(1e-3)
        min_depo  = fn.parameter(1e-6)
        typical_l = fn.math.sqrt(self._mesh.pointwise_area)

        erosion_timestep    = ((slope + min_slope) * typical_l / (erosion_rate + min_depo))
        deposition_timestep = ((slope + min_slope) * typical_l / (deposition_rate + min_depo))

        dt_erosion_local     = (erosion_timestep.evaluate(self._mesh)).min()
        dt_deposition_local  = (deposition_timestep.evaluate(self._mesh)).min()
        dt_erosion_global    = np.array(1e12)
        dt_deposition_global = np.array(1e12)

        comm.Allreduce([dt_erosion_local, MPI.DOUBLE], [dt_erosion_global, MPI.DOUBLE], op=MPI.MIN)
        comm.Allreduce([dt_deposition_local, MPI.DOUBLE], [dt_deposition_global, MPI.DOUBLE], op=MPI.MIN)

        return min(dt_erosion_global, dt_deposition_global)



    def time_integration(self, timestep, steps=1, Delta_t=None, feedback=None):

        from quagmire import function as fn

        topography = self._mesh.topography

        if Delta_t is not None:
            steps = Delta_t // timestep
            timestep = Delta_t / steps

        elapsed_time = 0.0

        for step in range(0, int(steps)):

            # deal with local drainage
            # mesh.low_points_local_flood_fill()

            erosion_rate, deposition_rate = self.erosion_deposition_local_equilibrium()

            # half timestep
            topography0 = topography.copy()
            topography.unlock()
            topography.data = topography.data + 0.5*timestep*(deposition_rate - erosion_rate)
            topography.lock()

            # full timestep
            erosion_rate, deposition_rate = self.erosion_deposition_local_equilibrium()

            with self._mesh.deform_topography():
                # rebuild downhill matrix structure
                topography.data = topography0.data + timestep*(deposition_rate - erosion_rate)

            elapsed_time += timestep

            if feedback is not None and step%feedback == 0 or step == steps:
                print("{:05d} - t = {:.3g}".format(step, elapsed_time))



        return steps, elapsed_time

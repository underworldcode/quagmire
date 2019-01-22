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

try: range = xrange
except: pass


class MeshVariable(object):
    """
    Mesh variables live on the global mesh
    Every time its data is called a local instance is returned
    """
    def __init__(self, name, mesh):
        self._mesh = mesh
        self._dm = mesh.dm

        name = str(name)

        # mesh variable vector
        self._ldata = self._dm.createLocalVector()
        self._ldata.setName(name)
        return


## This is a redundancy - @property definition is nuked by the @ .getter
## LM: See this: https://stackoverflow.com/questions/51244348/use-of-propertys-getter-in-python

## Don't sync on get / set as this prevents doing a series of computations on the array and
## doing the sync when finished. I can also imagine this going wrong if sync nukes values
## in the shadow zone unexpectedly.

    @property
    def data(self):
        pass

    @data.getter
    def data(self):
        return self._ldata

    @data.setter
    def data(self, val):
        if type(val) is float:
            self._ldata.set(val)
        else:
            from petsc4py import PETSc
            self._ldata.setArray(val)


    def getGlobalVector(self, gdata):
        from petsc4py import PETSc

        self._dm.localToGlobal(self._ldata, gdata, addv=PETSc.InsertMode.INSERT_VALUES)

        return


    def sync(self, mergeShadow=False):

        """ Explicit global sync of data """

        from petsc4py import PETSc

        if mergeShadow:
            addv = PETSc.InsertMode.ADD_VALUES
        else:
            addv = PETSc.InsertMode.INSERT_VALUES

        # need a global vector
        gdata = self._dm.getGlobalVec()

        # self._dm.localToLocal(self._ldata, self._gdata)
        self._dm.localToGlobal(self._ldata, gdata, addv=addv)
        self._dm.globalToLocal(gdata, self._ldata)

        self._dm.restoreGlobalVec(gdata)

        return

    ## For printing and other introspection we actually want to look through to the
    ## numpy array not the petsc vector description

    def __str__(self):
        return "{}".format(self._ldata.array.__str__())

    def __repr__(self):
        return "MeshVariable({})".format(self._ldata.array.__str__())

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
    def __init__(self, name, dm):
        self._dm = dm
        name = str(name)

        # mesh variable vector
        self._gdata = dm.createGlobalVector()
        self._ldata = dm.createLocalVector()

        self._gdata.setName(name)
        self._ldata.setName(name)

## This is a redundancy - @property definition is nuked by the @ .getter
    @property
    def data(self):
        # return self._gdata
        pass

    @data.getter
    def data(self):
        print "getter"
        self._dm.globalToLocal(self._gdata, self._ldata)
        return self._ldata

    @data.setter
    def data(self, val):
        print "setter"
        if type(val) is float:
            self._ldata.set(val)
            self._gdata.set(val)
        else:
            self._ldata.setArray(val)
            self._dm.localToGlobal(self._ldata, self._gdata)

    @data.deleter
    def data(self):
        print "deleter"
        self._ldata.destroy()
        self._gdata.destroy()

    def getGlobal(self):
        print "global"
        return self._gdata

    def getLocal(self):
        print "local"
        return self.data

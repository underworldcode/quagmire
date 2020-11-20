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

"""
Tools for creating and saving meshes
"""

import warnings

from .meshtools import *
from .generate_xdmf import generateXdmf as generate_xdmf
from .io import *


## Here we can provide a check to see if even the basic 
## quagmire cloud functionality can be made to work. 
## If cloudstor is not available (fs and fs-webdav) then
## we can avoid adding cloud capability to various classes 
## that provide it (e.g. mesh variables and mesh loading)

cloud_fs = False

try:
    import fs
    from webdavfs.webdavfs import WebDAVFS
    cloud_fs = True
except:
    pass 

## Do not explicitly import cloud modules here as they are optional


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

import pkg_resources as _pkg_resources
from distutils import dir_util as _dir_util
import os


def install_documentation(path="./Quagmire-Notebooks"):
    """Install the example notebooks for quagmire in the given location

    WARNING: If the path exists, the Notebook files will be written into the path
    and will overwrite any older, existing files with which they collide. The default
    path ("./Quagmire-Notebooks") is chosen to make collision less likely / problematic

    The documentation for quagmire is in the form of jupyter notebooks.

    Some dependencies exist for the notebooks to be useful:

       - matplotlib: for some diagrams
       - lavavu: for plotting 3D interactive surfaces

    Quagmire dependencies may be explicitly imported into the notebooks including:

       - stripy (for triangulation / interpolation)
       - numpy
       - scipy (for k-d tree point location)

    """

   ## TODO: download / clone the repo on demand to the require path

   print("Documentation is availble at https://github.com/underworld-community/quagmire-examples-and-workflows")

   return

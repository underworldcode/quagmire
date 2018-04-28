#! /usr/bin/env python2
#

from subprocess import call
import os

## Build local files into the www location to be served by jupyter

sitePath = os.path.normpath(os.path.dirname((os.path.join(os.getcwd(),os.path.dirname(__file__) ) ) ) )
siteDir = os.path.join(sitePath,"www")

# Copy Notebooks to the user space (leaving originals unchanged)
# Make symlinks for Data and Documentation so that these are visible
# to the web server started in the www build directory

print ("Building {:s}".format(siteDir))

## mkdocs to build the site
call("cd {:s} && mkdocs build --theme united --clean".format(sitePath), shell=True)

## copy pdf documentation and other helpful information that is not in mkdocs
call("ln -s {:s}/Documentation/ {:s}".format(sitePath, siteDir), shell=True)

## All the notebook examples are  installed from module calls (should use os.join / os.path etc)
call("mkdir -p {:s}/Notebooks".format(siteDir), shell=True)

print ("Adding {:s}".format(os.path.join(siteDir,"Notebooks","Stripy")))
import stripy
stripy.documentation.install_documentation(os.path.join(siteDir,"Notebooks","Stripy"))

print ("Adding {:s}".format(os.path.join(siteDir,"Notebooks","Quagmire")))

import quagmire
quagmire.documentation.install_documentation(os.path.join(siteDir,"Notebooks","Quagmire"))

## copy pdf documentation and other helpful information that is not in mkdocs
call("ln -s {:s}/quagmire/Examples/data {:s}/Notebooks".format(sitePath, siteDir), shell=True)


## Any data to copy over ?

# call("ln -s {:s}/Data/ {:s}".format(sitePath, siteDir), shell=True)
# call("cp -r {:s}/Notebooks/ {:s}/Notebooks".format(sitePath,siteDir), shell=True)


## Trust all notebooks
call("find {} -name \*.ipynb  -print0 | xargs -0 jupyter trust".format(siteDir), shell=True)

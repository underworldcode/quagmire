#! /usr/bin/env python
#

from subprocess import call
import os

sitePath = os.path.normpath(os.path.dirname((os.path.join(os.getcwd(),os.path.dirname(__file__) ) ) ) )
notebookPath = os.path.join(sitePath,"Notebooks")

# Copy Notebooks to the user space (leaving originals unchanged)
# Make symlinks for Data and Documentation so that these are visible
# to the web server started in the www build directory

# MULTIUSER: the "root" user is in www and has a symlink to the notebooks
# which allows that user to modify the master copies of the notebooks.
# Use a different theme for this one !!

# Note, such changes are not effected unless the site is rebuilt, but also,
# the rebuild does not nuke changes by the root user.

siteDir = os.path.join(sitePath,"www")

print "Building {:s}".format(siteDir)

call("cd {:s} && mkdocs build --theme united --clean".format(sitePath), shell=True)
call("ln -s {:s}/Data/ {:s}".format(sitePath, siteDir), shell=True)
call("ln -s {:s}/Notebooks/ {:s}".format(sitePath, siteDir), shell=True)
call("ln -s {:s}/Documentation/ {:s}".format(sitePath, siteDir), shell=True)


userList = [ str(i) for i in range(1,13) ]

for user in userList:
    siteDir = os.path.join(sitePath,"build","www"+user)
    print "Building {:s}".format(siteDir)
    call("cd {:s} && mkdocs build --site-dir {:s}".format(sitePath, siteDir), shell=True)
    call("ln -s {:s}/Data/ {:s}".format(sitePath, siteDir), shell=True)
    call("ln -s {:s}/Documentation/ {:s}".format(sitePath, siteDir), shell=True)
    call("cp -r {:s}/Notebooks {:s}/Notebooks".format(sitePath,siteDir), shell=True)  # Hey ! should using os.path.join !!

call("find . -name \*.ipynb  -print0 | xargs -0 jupyter trust", shell=True)

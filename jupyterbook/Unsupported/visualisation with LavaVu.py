# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# %% [markdown]
# # Visualising with LavaVu
#
# [LavaVu](https://github.com/OKaluza/LavaVu) is a lightweight, automatable visualisation and analysis viewing utility.
#
# > This notebook assumes you have run `distributed_mesh.py` in the tests folder with one or more processors.
#
# How to load HDF5 data with h5py and display with LavaVu...

# %%
import glucifer
import h5py
import numpy

# %%
f = h5py.File('../Scripts/AusFlow1a.h5', 'r')

#Print some info about the data structures
lavavu.printH5(f)

# %%
f.keys()

# %%
for field in f['fields']:
    print(field)
    
for label in f['labels']:
    print(label)

# %%
topography = f['fields']['height']
vertices = f['geometry']['vertices']
cells = f['viz']['topology']['cells']

boundary = f['labels']['boundary']['1']['indices']

# %%
#print topography[0:10],topography.shape
#print vertices[0:10],vertices.shape
#print cells[0:10],cells.shape
#print boundary[0:10],topography.shape

# %% [markdown]
# **Create a viewer**

# %%
lv = lavavu.Viewer(border=False, background="darkgrey", resolution=(1000,1000))

# %% [markdown]
# **Plot a triangle surface**
#
# Can be vertices only (3 per tri) or vertices (shared) with indices (3 per tri)

# %%
#Convert the vertex & vector arrays to 3d with numpy
verts = numpy.reshape(vertices, (-1,2))
#Insert the topography layer in Y axis
verts = numpy.insert(verts, 2, values=topography, axis=1)

# %%
#Plot the triangles
tris = lv.triangles("surface")
tris.vertices(verts)
tris.indices(cells)

# %%
lv.defaultcolourmaps()

# %% [markdown]
# **Add values, can be used to colour and filter the data**

# %%
#Use topography value to colour the surface
tris.values(topography, 'topography')
#tris.values(boundary, "boundary") #Load another field
#tris.values(tin["discharge"])
cm = tris.colourmap("etopo1") #Apply a built in colourmap
cb = tris.colourbar() #Add a colour bar

#Filter by min height value
tris["zmin"] = 0.011

# %% [markdown]
# **Apply an initial rotation and display an interactive viewer window**
#
# Along with viewer window, controls can be added to adjust properties dynamically

# %%
lv.rotate('x', -60)

lv.window()
lv.control.Range(command='translate x ', range=(-5,5), step=0.1)
tris.control.Checkbox('wireframe')
tris.control.Range('zmin', range=(0,1), step=0.001)

lv.control.Range(command='background', range=(0,1), step=0.1, value=1)
lv.control.ObjectList()
lv.control.show()



# %%
tris.reload()
lv.redisplay()

# %% [markdown]
# **Saving camera position, images and state**
#
#

# %%
lv.camera() #Get current camera as set in viewer

# %%
#Plot a static image with a saved camera setup
lv.translation(0.00558684254065156, 1.16282057762146, -8.33703231811523)
lv.rotation(-0.487509340047836, 0.172793388366699, 0.269237726926804, 0.807206630706787)
lv.display(resolution=[480,320])
lv.image('saved.png', resolution=[640,480], transparent=True) #Save to disk

# %%
#State of properties and view can be loaded and saved to a file
lv.save("state.json")
#lv.file("state.json")


# %%
# lv.export(file="testdb", format="dbz")


# %% language="sh"
#
# ls -trl

# %%

# %%

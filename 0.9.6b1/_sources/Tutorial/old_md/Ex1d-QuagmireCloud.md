---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# quagmire.tools.cloud

The `quagmire.tools.cloud` module helps to download and upload data (particularly meshes and meshvariables) from various cloud services.
By default, this module provides read-only access to a central repository of meshes and datasets 

The quagmire cloud uses [PyFilesystem](https://www.pyfilesystem.org/) to establish connections to cloud directories via webdav. This is what `PyFilesystem` promises:

   > Filesystem Abstraction for Python.

   > Work with files and directories in archives, memory, the cloud etc. as easily as your local drive. Write code now, decide later where the data will be stored; unit test without writing real files; upload files to the cloud without learning a new API; sandbox your file writing code; etc.
   
Storage for the cloud access is provided by [cloudstor](https://www.aarnet.edu.au/network-and-services/cloud-services/cloudstor/) which is hosted by [AARNET](https://www.aarnet.edu.au/). 
    
    
In these examples, we will show how to access mesh and meshvariable data stored in the cloud. The examples we use are taken from the earlier examples.

```{code-cell} ipython3
from quagmire.tools import meshtools
from quagmire import QuagMesh
from quagmire.mesh import MeshVariable
import numpy as np  
```

In the `Ex1-Creating-Meshes` notebook we learned how to save a mesh file and how to read it back to re-create the mesh.


```python
filename = "Ex1-refined_mesh.h5"

# save from QuagMesh object:
# mesh2.save_mesh_to_hdf5(filename)

# save from meshtools:
meshtools.save_DM_to_hdf5(DM_r2, filename)

# load DM from file
DM_r2 = meshtools.create_DMPlex_from_hdf5(filename)

mesh2 = QuagMesh(DM_r2)

print(mesh2.npoints)
print(mesh2.area)

```

    1333
    [0.1003038  0.09579325 0.07775106 ... 0.05819681 0.06067121 0.06067121]


This mesh is available in the Quagmire cloud and we can create a new PETSc DM and a QuagMesh object directly from the cloud as follows (and it seems to create a valid mesh)

```{code-cell} ipython3
from quagmire.tools.cloud import quagmire_cloud_fs

DM_r2 = meshtools.create_DMPlex_from_cloud_fs("Examples/Tutorial/Ex1-refined_mesh.h5")
mesh2 = QuagMesh(DM_r2)

print()
print(mesh2.npoints)
print(mesh2.area)
```

```{code-cell} ipython3
DMC = meshtools.create_DMPlex_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh.h5")
meshC = QuagMesh(DMC)

print(meshC.npoints)
print(meshC.area)

phi = meshC.add_variable(name="PHI(X,Y)")
psi = meshC.add_variable(name="PSI(X,Y)")
```

```{code-cell} ipython3
phi.load_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh_phi.h5")
```

## My cloud ?

The quagmire cloud is intended for the developers to share examples and pre-computed meshes with users but it is read only. If you wish to use cloud-storage for your own files, you will need to use the full functionality of the quagmire cloud interface.

For example, the functionality for loading the mesh variables has a more general form:


```python

from quagmire.tools.cloud import cloudstor

cloudstor_fs = cloudstor(url="https://cloudstor.aarnet.edu.au/plus/s/4SEAhkqSlTojYhv", password="8M7idzp2Q7DXLMz()()()()()")
cloud_dir = cloudstor_fs.opendir('/')

phi.load_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh_phi.h5", cloud_location_handle=cloud_dir)

```

This is using the `PyFilesystem` webdav interface for accessing public urls on cloudstor. If you have a cloudstor account, you can use obtain a public link to share any folder, set your own password and create your own cloud access. 

## My cloud, but not cloudstor ?

The interface that we provide to cloudstor can be generalised some more:

``` python
import fs

username = "4SEAhkqSlTojYhv"
password = "8M7idzp2Q7DXLMz()()()()()" 
webdav_url = "webdav://{}:{}@cloudstor.aarnet.edu.au:443/plus/public.php/webdav/".format(username, password)
cloud_dir = fs.open_fs(webdav_url)

phi.load_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh_phi.h5", cloud_location_handle=cloud_dir)
```

As long as you can obtain a valid fs.open_fs object that points to a folder somewhere, then meshes and mesh variables can be loaded from those locations without explicitly having to download those files first. You will need to dig into your cloud providers webdav interface to make this work for you.

```{code-cell} ipython3
username = "4SEAhkqSlTojYhv"
password = "8M7idzp2Q7DXLMz()()()()()" 
webdav_url = "webdav://{}:{}@cloudstor.aarnet.edu.au:443/plus/public.php/webdav/".format(username, password)
cloud_dir = fs.open_fs(webdav_url)

phi.load_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh_phi.h5", cloud_location_handle=cloud_dir)
```

## I want to share a file for others to access

If you can provide a url to a file that can be accessed, then the quagmire cloud tools allow you to load a mesh or mesh variable from that link. Examples include providing a dropbox or google drive public link to a file.

Google drive provides urls for file sharing that need some manipulation to work in a python script. For a google URL that you copy from the web interface, use `url = quagmire.tools.cloud.google_drive_convert_link(g_url)`

```{code-cell} ipython3
psi.data = 0.0
psi.load_from_url("https://www.dropbox.com/s/5dzujlo3ayo5s35/Ex1a-circular_mesh_psi.h5?dl=0")
print(psi.data)

# Note the dropbox link is to the file only, regardless of the tail of the URL:
psi.data = 0.0
psi.load_from_url("https://www.dropbox.com/s/5dzujlo3ayo5s35")
print(psi.data)

# Verification
psi.data = 0.0
psi.load_from_cloud_fs("Examples/Tutorial/Ex1a-circular_mesh_psi.h5", cloud_location_handle=cloud_dir)
print(psi.data)
```

```{code-cell} ipython3
from quagmire.tools.cloud import google_drive_convert_link

psi.data = 0.0
gurl = "https://drive.google.com/file/d/17t8jbPFmnB8aHhyYDbxjGrzKqtq6IlCa/view?usp=sharing"
url = google_drive_convert_link(gurl)
print(url)
psi.load_from_url(url)
print(psi.data)
```

## Upload / download tools

The `quagmire.tools.cloud` functions `cloud_upload` and `cloud_download` wrap PyFilesystem calls to make sure that they check the validity of the filesystem objects and work well within a parallel environment (only the root processor will download the file).

The `quagmire.tools.cloud` function `url_download` provides similar capability for a standard http or https request to download a file from a public link and also ensures this is done only once in a parallel environment.

```{code-cell} ipython3

```

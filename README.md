# Quagmire

Quagmire is a Python surface process framework for building erosion and deposition models on highly parallel, decomposed structured and unstructured meshes.

## Dependencies

Running this code requires the following packages to be installed:

- Python 2.7.x and above
- Numpy 1.9 and above
- [mpi4py](http://pythonhosted.org/mpi4py/usrman/index.html)
- [petsc4py](https://pythonhosted.org/petsc4py/usrman/install.html)
- h5py (optional - for saving parallel mesh data)

Building unstructured meshes (using the routines in the *tools* folder) require Delaunay triangulations found in either of the following dependencies:

- SciPy 0.15.0 and above
- [Triangle](http://dzhelil.info/triangle/) (faster)

### PETSc installation

PETSc is used extensively via the Python frontend, petsc4py. It is required that PETSc be configured and installed on your local machine prior to using Quagmire. You can use pip to install petsc4py and its dependencies.

```
[sudo] pip install numpy mpi4py
[sudo] pip install petsc petsc4py
```

If this does not work you must compile these manually.
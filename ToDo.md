## TO DO

### Docker environments

 - [x] Dockerfile:  needs mpirun in the path
 - [x] Dockerfile:  needs to have the proper token stuff for Jupyter
 - [ ] Dockerfile:  add the help script etc.
 - [ ] Separate binder version

### MeshVariable implementation:

 - [x] MeshVariable: add print function to output array value `__str__` and `__repr__`
 - [x] MeshVariable: add gradient and interpolation / evaluate methods (DONE)
 - [x] mesh.add_variable ... cf underworld

### Meshing
 - [ ] Pixmesh: add an interpolate method.

### Scaling

  - [ ] Integrate scaling into MeshVariable view / getter
  - [ ] Integrate scaling into meshes
  - [ ] Save / Load ? Scaling dictionary ?

### Equation solvers

  - Explicit
    - [ ] Diffusion
    - [ ] Erosion-deposition

  - Semi-implicit (?)

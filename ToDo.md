## TO DO

### Docker environments

 - [x] Dockerfile:  needs mpirun in the path (0.5)
 - [x] Dockerfile:  needs to have the proper token stuff for Jupyter (0.5)
 - [ ] Dockerfile:  add the help script etc.
 - [x] Separate binder version (branch) (0.6)

### MeshVariable implementation:

 - [x] MeshVariable: add print function to output array value `__str__` and `__repr__` (0.5)
 - [x] MeshVariable: add gradient and interpolation / evaluate methods (0.5)
 - [x] mesh.add_variable ... cf underworld (0.5)


### Functions

 - [x] eliminate need for mesh in parameter functions (0.6)
 - [ ] expose DM labels as functions / meshVariables
 - [x] dependency tracking (useful for non-linearity detection in solvers) (0.7)
 - [x] human-readable naming for fns (meshes / solvers too) (0.7)
 - [ ] store values at each evaluate statement to reuse (?)

### Meshing

 - [ ] Pixmesh: add an interpolate method.
 - [ ] Spherical meshing framework (0.7)

### Scaling

  - [x] Add scaling module - i.e. `from quagmire import scaling`  (0.6)
  - [ ] Integrate scaling into MeshVariable view / getter
  - [ ] Integrate scaling into meshes
  - [ ] Save / Load ? Scaling dictionary ?


### Input / output

 - [x] Save xmf files for Paraview (0.7)
 - [ ] Save xmf time series (0.7)
 - [ ] Save / load project file (`.quag` ?) (0.7)

### Equation solvers

  - Explicit
    - [x] Diffusion (0.6)
      - [x] Linear, constant diffusivity + benchmark (0.6)
      - [x] Spatially variable diffusity + example (0.6)
      - [ ] Non-linear diffusivity
      - [ ] Timestep for non-linear case (meaning of provided step ?)
      - [ ] Semi-implicit (?)
    - [ ] Erosion-deposition (0.7)
      - [ ] Erosion / deposition governing equations + example (0.7)
      - [ ] Linear, constant erodability (0.7)
      - [ ] Spatially verying erodability (0.7)
    - [ ] Integrated erosion-deposition-diffusion time stepping


### Testing

  - [x] Example notebooks with reasonable coverage - all passing
  - [ ] Unit test scripts (serial)
  - [ ] Unit test scripts (parallel)


### Visualisation

  - [x] lavavu examples (0.3)
  - [ ] lavavu parallel wrappers (cf gLucifer)
  - [ ] lavavu mesh variable wrappers (cf gLucifer)
  - [ ] lavavu + shapely for maps (?)

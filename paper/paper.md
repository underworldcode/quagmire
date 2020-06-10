---
title: 'Quagmire: a parallel Python framework for modelling surface processes'
tags:
  - Python
  - numpy
  - parallel
  - petsc4py
  - surface-processes
  - geodynamics
  - geomorphology
authors:
  - name: Ben Mather
    orcid: 0000-0003-3566-1557
    affiliation: "1, 2"
  - name: Louis Moresi
    orcid: 0000-0003-3685-174X
    affiliation: 3
  - name: Romain Beucher
    orcid: 0000-0003-3891-5444
    affiliation: 3
affiliations:
 - name: EarthByte Group, School of Geosciences, The University of Sydney, NSW, 2006 Australia 
   index: 1
 - name: Sydney Informatics Hub, The University of Sydney, NSW, 2006 Australia
   index: 2
 - name: Research School of Earth Sciences, Australian National University, ACT, 2601 Australia
   index: 3
date: 22 May 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Journal of Open Source Software
---

# Summary

Landscape evolution modelling simulates the erosion of mountain belts and deposition of sediment into basins in response to tectonic driving forces.
Our work is driven by the challenge of interpreting how deep-seated and long-lived plate-scale geodynamic processes are expressed in the geological record that we can directly sample in the upper few kilometres of the crust.
We present an efficient, parallel approach to modelling the physical evolution of topography.

Quagmire is a Python surface process framework for building erosion and deposition models on structured and unstructured meshes distributed in parallel.
Quagmire is built atop [PETSc](https://www.mcs.anl.gov/petsc/) using the `petsc4py` Python wrapper, which handles the partitioning of vectors and matrices that are mesh-dependent across multiple processors.
Quagmire provides a parallel-safe `function` interface to avoid much of the synchronisation issues typically encountered within parallel computing environments.
These functions provide the building pieces for users to construct their workflows, such as evaluating derivatives and integrating information on a mesh.
In this way, Quagmire is a highly flexible and extensible framework for solving landscape evolution problems.


# Mathematical background

The contributing processes to landscape evolution depend on local hill slope diffusion, long-range transport, and tectonic uplift,

$$
  \frac{\mathrm{d}h}{\mathrm{d}t} =  \dot{h}_\textrm{local} 
           + \dot{h}_\textrm{incision} 
           + \dot{h}_\textrm{deposition}   
           + \dot{h}_\textrm{basement}
$$

where $h$ is the surface height at each point and the $\dot{h}$ terms are time derivatives with respect to height.
Summed together, the change in height from local diffusion, fluvial incision, deposition, and regional basement uplift/subsidence describe the processes that govern landscape morphology.


## Local evolution rate

The local evolution rate represents small-scale, hill-slope dependent processes which can be represented as a non-linear diffusion equation. 

$$
  \dot{h}(\mathbf{x})_\textrm{local} = \nabla \left[\kappa(h,\mathbf{x}) \nabla h(\mathbf{x}) \right]
$$

$\kappa$ is a non-linear diffusion coefficient which can, for example, be used to enforce a critical hill slope value if it is a strongly increasing function of the local gradient.
Evaluating spatial derivatives is trivial on a regular grid and are outsourced to [stripy](https://github.com/underworldcode/stripy) for triangulated meshes [@Moresi:2019].
Derivatives on the mesh can be evaluated using functions interface, e.g.

```python
fn_dhdx, fn_dhdy = fn.math.grad(height)
fn_dhdx.evaluate(mesh) # evaluate on the mesh 
```


## Incision and deposition

The fluvial incision rate includes the effect of cumulative rainfall run-off across the landscape. This term encapsulates the available energy of rivers which in turn is related to both the discharge flux at any given point and the local stream-bed slope. The incision rate may be written in "stream power" form,

$$
  \dot{h}(\mathbf{x})_\textrm{incision} = 
      K(\mathbf{x}) q_r(\mathbf{x})^m \left| \nabla h(\mathbf{x}) \right|^n
$$

where $K$, $m$, $n$ are constants, $q_r$ is the runoff flux, and $\left| \nabla h \right|$ is the downhill bed slope.

$$
  q_r(\mathbf{x}) = \int\limits_{\textrm{upstream}} {\mathcal{R}} (\xi) d \xi
$$

This integral computes the accumulated run-off for all of the areas which lie upstream of the point $\mathbf{x}$.
The cumulative power of any given stream is intrinsically controlled by the inter-connectivity of nodes in the mesh that describe the network of tributaries.
In this way, the discretisation of a landscape imparts a significant role on the integrated rainfall flux of a catchment.


### Integrating upstream rainfall

The network graph of tributaries can be represented by a matrix, $\mathbf{D}$, for any given landscape.
Multipling this matrix with a rainfall vector, $\mathbf{\mathcal{R}}$, shifts information by one increment downstream.
If this operation is applied recursively, then all information is propagated to an outflow point or a local minimum in the landscape.

$$
  \mathbf{q_r} = \sum_{i=0}^N \mathbf{D} \mathbf{\mathcal{R}}_i
$$

The sparsity of $\mathbf{D}$ is controlled by the number of river pathways in the landscape.
Commonly, multiple descent pathways are desired to allow stream splitting.
This is where rainfall run-off is partitioned from a donor node to more than one recipient nodes in its local vicinity.
Quagmire lets the user control how many downhill neighbours to partition flow by setting the `downhill_neighbours` keyword on initialisation (default is 2),

```python
mesh = quagmire.QuagMesh(DM, downhill_neighbours=2)
```
The upstream integral may be calculated using the function interface in Quagmire,

```python
fn_integral = mesh.upstream_integral_fn(rainfall)
fn_integral.evaluate(mesh) # evaluate on the mesh
```

# Usage

Quagmire supports three types of mesh objects that may be constructed using functions within the `quagmire.tools.meshtools` module:

```python
# cartesian mesh on a grid
DM = create_DMDA(minX, maxX, minY, maxY, resX, resY)

# unstructured cartesian mesh
DM = create_DMPlex(x, y, simplices, boundary_vertices=None, refinement_levels=0)

# unstructured mesh on the sphere
DM = create_spherical_DMPlex(lons, lats, simplices, boundary_vertices=None, refinement_levels=0)
```

These `DM` objects may be passed to `QuagMesh`, which automatically determines what type of mesh it has received and initialises the appropriate data structures.

```python
mesh = quagmire.QuagMesh(DM, downhill_neighbours=2):
```

## Adding topography

Mesh variables can be created to safely handle global operations in parallel.
One such mesh variable is topography, which may be updated via its context manager:

```python
with mesh.deform_topography():
    mesh.topography.data = height
```

This triggers an update of the stream networks represented by the downhill matrix, $\mathbf{D}$, and integrates the upstream area.

Some algorithms are provided to process the topography in order to render it suitable for simulating erosion and deposition.
A commonly faced issue in using Digital Elevation Models (DEMs) is the existence of __flat regions__ over a large area which have no easily identifiable stream pathway, or __local minima__ which truncate the stream network.
To address this, the `low_points_swamp_fill` routine adds a small gradient to flat regions up to a spill point, and `low_points_local_patch_fill` patches local minima to smooth artifacts in the slope.

## Functions interface

Operator overloading. Lazy evaluation of functions that only get evaluated when explicitly asked.

# Acknowledgments

Development of Quagmire is financially supported by AuScope as part of the Simulation Analysis Modelling platform (SAM) and the NSW Department of Industry grant awarded through the Office of the Chief Scientist and Engineer.

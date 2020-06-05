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

Quagmire is a Python surface process framework for building erosion and deposition models on stuctured and unstructured meshes distributed in parallel.


# Erosion, deposition and transport models

The contributing processes to landscape evolution depend on local hillslope diffusion, long-range transport, and tectonic uplift,

$$
\begin{equation}
  \frac{Dh}{Dt} =  \dot{h}_\textrm{local} 
           + \dot{h}_\textrm{incision} 
           + \dot{h}_\textrm{deposition}   
           + \dot{h}_\textrm{basement}
\end{equation}
$$

where $h$ is the surface height at each point and the $\dot{h}$ terms are time derivatives with respect to height. Summed together, the change in height from local diffusion, fluvial incision, deposition, and regional basement uplift/subsidence describe the processes that govern landscape morphology.


## Local evolution rate

The local evolution rate represents small-scale, hill-slope dependent processes which can be represented as a non-linear diffusion equation. 

$$
\begin{equation}
  \dot{h}(\bm{x})_\textrm{local} = \nabla \left[\kappa(h,\bm{x}) \nabla h(\bm{x}) \right]
\end{equation}
$$

$\kappa$ is a non-linear diffusion coefficient which can, for example, be used to enforce a critical hill slope value if it is a strongly increasing function of the local gradient.
Evaluating spatial derivatives is trivial on a regular grid and are outsourced to [stripy](https://github.com/underworldcode/stripy) for triangulated meshes [@Moresi:2019].


## Incision and deposition

The fluvial incision rate includes the effect of cumulative rainfall runoff across the landscape. This term encapsulates the available energy of rivers which in turn is related to both the discharge flux at any given point and the local stream-bed slope. The incision rate may be written in "stream power" form,

$$
\begin{equation}
  \dot{h}(\bm{x})_\textrm{incision} = 
      K(\bm{x}) q_r(\bm{x})^m \left| \nabla h(\bm{x}) \right|^n
\end{equation}
$$

where $K$, $m$, $n$ are constants, $q_r$ is the runoff flux, and $\left| \nabla h \right|$ is the downhill bed slope.

$$
  \begin{equation}
    q_r(\bm{x}) = \int\displaylimits_{\textrm{upstream}} \!\!\!\! {\cal R} (\xi) d \xi
  \end{equation}
$$

This integral computes the accumulated run off for all of the areas which lie upstream of the point $\bm{x}$.
The cumulative power of any given stream is intrinsically controlled by the interconnectivity of nodes in the mesh. Thus, the discretisation of a landscape imparts a significant role on the integrated rainfall flux or a catchment.
The cumulative power of any given stream is intrinsically controlled by the interconnectivity of nodes in the mesh that describe the network of tributaries.
In this way, the discretisation of a landscape imparts a significant role on the integrated rainfall flux or a catchment.


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

We acknowledge the AuScope Simulation, Anaylsis & Modelling programme funded by the Australian Government through the National Collaborative Research Infrastructure Strategy (NCRIS).

# References
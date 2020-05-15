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

Quagmire is a Python surface process framework for building erosion and deposition models on highly parallel, decomposed structured and unstructured meshes.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

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
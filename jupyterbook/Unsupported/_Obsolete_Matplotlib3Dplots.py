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

# %%
import ipyvolume.pylab as p3
import numpy as np

# %%
s = 1/2**0.5
# 4 vertices for the tetrahedron
x = np.array([1.,  -1, 0,  0])
y = np.array([0,   0, 1., -1])
z = np.array([-s, -s, s,  s])
# and 4 surfaces (triangles), where the number refer to the vertex index
triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1,3,2)]

# %%
p3.figure()
# we draw the tetrahedron
p3.plot_trisurf(x, y, z, triangles=triangles, color='orange')
# and also mark the vertices
p3.scatter(x, y, z, marker='sphere', color='blue')
p3.xyzlim(-2, 2)
p3.show()

# %%

# f(u, v) -> (u, v, u*v**2)
a = np.arange(-5, 5)
U, V = np.meshgrid(a, a)
X = U
Y = V
Z = X*Y**2
p3.figure()
p3.plot_surface(X, Z, Y, color="orange")
p3.plot_wireframe(X, Z, Y, color="black")
p3.show()

# %%
X = np.arange(-5, 5, 0.25*1)
Y = np.arange(-5, 5, 0.25*1)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = 0.1*np.sin(R)

# %%

from matplotlib import cm
colormap = cm.coolwarm
znorm = Z - Z.min()
znorm /= znorm.ptp()
znorm.min(), znorm.max()
color = colormap(znorm)

# %%
p3.figure()
mesh = p3.plot_surface(X, Z, Y, color=color[...,0:3])
p3.show()

# %%

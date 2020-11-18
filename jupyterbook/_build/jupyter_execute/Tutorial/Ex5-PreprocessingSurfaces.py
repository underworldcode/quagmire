#### Example 5 - Preprocessing a surface


Pit filling and swamp filling ... 

Take the previous mesh with random noise 

`Quagmire` allows the user to specify the number of downhill pathways to model landscape evolution. This is set using:

```python
mesh.downhill_neighbours = 1
mesh.update_height(height)
```

where an integer specifies the number of downhill neighbour nodes (receipients) that will receive a packet of information from a donor node. The `QuagMesh` object can also be initialised with:

```python
mesh = QuagMesh(DM, downhill_neighbours=1)
```

to specify the number of downhill neighbours (default is 2).

In this notebook we use a landscape function with many outflow points to examine the effect of varying the number of recipient nodes on catchment area, stream lengths, and outflow fluxes.


#### Notebook contents

- [1-2-3 downhill neighbours](#1-2-3-downhill-neighbours)
- [Upstream propogation](#Upstream-propogation)
- [Dense downhill matrices](#Dense-downhill-matrices)

import matplotlib.pyplot as plt
import numpy as np
from quagmire import tools as meshtools
from quagmire import function as fn
%matplotlib inline

from quagmire import QuagMesh 
from quagmire import QuagMesh # all routines we need are within this class
from quagmire import QuagMesh


minX, maxX = -5.0, 5.0
minY, maxY = -5.0, 5.0,

spacingX = 0.05
spacingY = 0.05

x, y, simplices = meshtools.elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, 1.)

DM = meshtools.create_DMPlex(x, y, simplices, refinement_levels=2)

mesh = QuagMesh(DM)

x = mesh.coords[:,0]
y = mesh.coords[:,1]

print( "\nNumber of points in the triangulation: {}".format(mesh.npoints))

# topography

radius  = np.sqrt((x**2 + y**2))
theta   = np.arctan2(y,x)+0.1

height  = np.exp(-0.025*(x**2 + y**2)**2) + 0.25 * (0.2*radius)**4  * np.cos(5.0*theta)**2 ## Less so
height  += 0.5 * (1.0-0.2*radius)
height  -= height.min()

## Add some pits 

hrand0 = np.where(np.random.random(height.shape)>0.995, -0.3, 0.0)


## Add smoothed random noise to make some "lakes" 

rbf_smoother = mesh.build_rbf_smoother(0.05, iterations=3)
h0 = mesh.add_variable(name="h0")
h0.data = np.where(np.random.random(height.shape)>0.995, -1.0, 0.0)

hrand1 = 25.0 * rbf_smoother.smooth_fn(h0, iterations=25).evaluate(mesh)


# randpts1 = np.where(np.random.random(height.shape)>0.995, -1.0, 0.0)
# hrand1   = 10.0 * rbf_smoother.smooth(randpts1, iterations=10)

heightn = height + hrand0 + hrand1

with mesh.deform_topography():
    
    mesh.downhill_neighbours = 2
    mesh.topography.data = heightn

# let's use a rainfall proportional to height (any choice is ok)

rainfall_fn = mesh.topography**2

low_points = mesh.identify_low_points()
low_point_coords = mesh.coords[low_points] 
print(low_points.shape)

h0.data = mesh.topography.data

import lavavu

lowsxyz = np.column_stack([mesh.tri.points[low_points], height[low_points]])
xyz  = np.column_stack([mesh.tri.points, height])
xyz2 = np.column_stack([mesh.tri.points, heightn])


lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[1000,600], near=-10.0)

nodes = lv.points("nodes", pointsize=3.0, pointtype="shiny", colour="#448080", opacity=0.75)
nodes.vertices(lowsxyz)

tris = lv.triangles("triangles",  wireframe=False, colour="#77ff88", opacity=1.0)
tris.vertices(xyz)
tris.indices(mesh.tri.simplices)
tris.values(mesh.topography.evaluate(mesh), label="height")
tris.values(rainfall_fn.evaluate(mesh), label="rainfall")
tris.values(heightn-height, label="perturbation")

# tris.colourmap("#990000 #FFFFFF #000099")
tris.colourmap("elevation")
cb = tris.colourbar()

tris2 = lv.triangles("triangles2",  wireframe=False, colour="#77ff88", opacity=1.0)
tris2.vertices(xyz2)
tris2.indices(mesh.tri.simplices)
tris2.values(heightn, label="heightn")
tris2.colourmap("elevation")
cb = tris2.colourbar()

# view the pole

# lv.translation(0.0, 0.0, -3.0)
# lv.rotation(-20, 0.0, 0.0)

lv.control.Panel()
lv.control.Range('specular', range=(0,1), step=0.1, value=0.4)
lv.control.Checkbox(property='axis')
lv.control.ObjectList()
tris.control.Checkbox(property="wireframe")
tris.control.List(options=["height", "rainfall", "perturbation"], property="colourby", value="orginal", command="redraw", label="Display:")
lv.control.show()

rainfall_fn = mesh.topography**2
flowrate_fn = mesh.upstream_integral_fn(rainfall_fn)
stream_power_fn = flowrate_fn ** 2.0 * mesh.slope ** 2.0 * fn.misc.levelset(mesh.mask, 0.5)

cumulative_rain_n1 = mesh.upstream_integral_fn(rainfall_fn).evaluate(mesh)

import lavavu

points = np.column_stack([mesh.tri.points, height])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)
tri1.values(mesh.slope.evaluate(mesh), "slope")
tri1.values(flowrate_fn.evaluate(mesh), "flow rate")
tri1.values(stream_power_fn.evaluate(mesh), "stream power")

tri1.colourmap("drywet", range=[0.0,1.0])
tri1.colourbar()

lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["slope", 
                   "flow rate",
                   "stream power"
                  ], property="colourby", value="slope", command="redraw")
lv.control.show()

## Pit filling algorithm in quagmire

mesh1p = QuagMesh(DM)
rainfall_fn_1p = mesh1p.topography**2


with mesh1p.deform_topography():
    mesh1p.topography.data = mesh.topography.data

mesh1p.low_points_local_patch_fill(its=5, smoothing_steps=1)

cumulative_rain_n1p = mesh1p.upstream_integral_fn(rainfall_fn_1p).evaluate(mesh1p)

# cumulative_rain_n1p = mesh1p.cumulative_flow(mesh.rainfall_pattern_Variable.data * mesh.area)
# stream_power_n1p    = compute_stream_power(mesh1p, mesh1p.slopeVariable.data, m=1, n=1)

import lavavu

points = np.column_stack([mesh.tri.points, height])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)

tri1.values(mesh1p.topography.data-mesh.topography.data,  "delta h pits")
tri1.values(mesh.slope.evaluate(mesh),           "slope (rough)")
tri1.values(mesh1p.slope.evaluate(mesh1p),       "slope (unpitted)")
# tri1.values(cumulative_rain,    "cumulative rain")
# tri1.values(cumulative_rain_n,  "cum-rain-rough")
# tri1.values(cumulative_rain_n1p,"cum-rain-unpitted")

tri1.colourmap("drywet")
tri1.colourbar()

lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["delta h pits", 
                           "slope (unpitted)",
                           "slope (rough)"], property="colourby", value="slope (unpitted)", command="redraw")
lv.control.show()

## Quagmire also has a swamp filling algorithm
## NOTE this is much more efficient if it follows the pit filling

mesh1s = QuagMesh(DM)
rainfall_fn_1s = mesh1s.topography**2

with mesh1s.deform_topography():
    mesh1s.topography.data = mesh.topography.data

for i in range(0,50):
    mesh1s.low_points_swamp_fill(ref_height=-0.01)
    
    # In parallel, we can't break if ANY processor has work to do (barrier / sync issue)
    low_points2 = mesh1s.identify_global_low_points()
    
    print("{} : {}".format(i,low_points2[0]))
    if low_points2[0] == 0:
        break

cumulative_rain_n1s = mesh1s.upstream_integral_fn(rainfall_fn_1s).evaluate(mesh1s)
# stream_power_n1s    = compute_stream_power(mesh1s, mesh1p.slopeVariable.data, m=1, n=1)

import lavavu

points = np.column_stack([mesh.tri.points, heightn])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)

tri1.values(np.log(cumulative_rain_n1s),"cum-rain-swamp")
tri1.values(np.log(cumulative_rain_n1p),"cum-rain-pits")
tri1.values(np.log(cumulative_rain_n1), "cumulative rain")

tri1.colourmap("#BBEEBB #889988 #000099")
tri1.colourbar()

## Swamped

points = np.column_stack([mesh1s.tri.points, mesh1s.topography.data-0.01])

tri2 = lv.triangles("triangles2", wireframe=False)
tri2.vertices(points)
tri2.indices(mesh1s.tri.simplices)

tri2.values(mesh1s.topography.data-mesh.topography.data,"swamps")
tri2.values(np.ones_like(mesh1s.topography.data), "blank")
tri2.values(np.log(cumulative_rain_n1s), "cum-rain-swamp")

tri2.colourmap("#003366:0.5, #000099:0.8, #000099")
tri2.colourbar()

lv.translation(0.0, 0.0, -19.915)
lv.rotation(-51.21, -1.618, -3.573)

lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["cum-rain-swamp",
                   "cum-rain-pits", 
                   "cumulative rain"
                   ], property="colourby", command="redraw")

tri2.control.List(options=["blank", "swamps", 
                   "cum-rain-swamp"], property="colourby", command="redraw")


lv.control.show()

## Stream power / slope where the lakes / swamps are located:

import lavavu

points = np.column_stack([mesh.tri.points, heightn])

lv = lavavu.Viewer(border=False, background="#FFFFFF", resolution=[600,600], near=-10.0)

tri1 = lv.triangles("triangles", wireframe=False)
tri1.vertices(points)
tri1.indices(mesh.tri.simplices)

tri1.values(mesh.slope.evaluate(mesh),    "slope (rough)")
tri1.values(mesh1s.slope.evaluate(mesh1s),  "slope (swamp)")
tri1.values(mesh.slope.evaluate(mesh)-mesh1s.slope.evaluate(mesh1s),  "delta slope")

tri1.colourmap("#444444 #777777 #FF8800", range=[0,1.0])
tri1.colourbar()

## Swamped

points = np.column_stack([mesh1s.tri.points, mesh1s.topography.data-0.01])

tri2 = lv.triangles("triangles2", wireframe=False)
tri2.vertices(points)
tri2.indices(mesh1s.tri.simplices)

tri2.values(mesh1s.topography.data-mesh.topography.data,   "swamps")
tri2.values(np.ones_like(mesh1s.topography.data), "blank")
tri2.values((cumulative_rain_n1s), "cum-rain-swamp")

tri2.colourmap("#003366:0.5, #000099:0.8, #000099")
tri2.colourbar()


lv.control.Panel()
lv.control.ObjectList()
tri1.control.List(options=["slope (rough)",
                   "slope (swamp)", 
                   "delta slope" 
                   ], property="colourby", command="redraw")

tri2.control.List(options=["blank", "swamps", 
                   "cum-rain-swamp"], property="colourby", command="redraw")


lv.control.show()


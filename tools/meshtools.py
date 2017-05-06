"""
Mesh generation functions

"""

import numpy as np

try: range = xrange
except: pass

class _Triangulation(object):
    """
    An abstraction for the triangle python module <http://dzhelil.info/triangle/>
    This class mimics the Qhull structure in SciPy.
    """
    def __init__(self, coords):
        import triangle
        self.points = coords
        self.npoints = len(coords)

        d = dict(vertices=self.points)
        tri = triangle.triangulate(d)
        self.simplices = tri['triangles']


class _ConvexHull(object):
    """
    An abstraction for calculating a convex hull using triangle
    This class mimics the ConvexHull structure in SciPy
    """
    def __init__(self, coords):
        from triangle import convex_hull
        from numpy import unique
        self.points = coords
        self.simplices = convex_hull(coords)
        self.vertices = unique(self.simplices)


class _RecoverTriangles(object):
    def __init__(self, dm):
        sect = dm.getDefaultSection()
        lvec = dm.createLocalVector()

        self.points = dm.getCoordinatesLocal().array.reshape(-1,2)
        self.npoints = self.points.shape[0]

        # find cells in the DAG
        cStart, cEnd = dm.getDepthStratum(2)

        # recover triangles
        simplices = np.empty((cEnd-cStart, 3), dtype=PETSc.IntType)
        lvec.setArray(np.arange(0,self.npoints))

        for t, cell in enumerate(range(cStart, cEnd)):
            simplices[t] = dm.vecGetClosure(sect, lvec, cell)

        self.simplices = simplices


def _points_to_edges(tri, boundary):
    """
    Finds the edges connecting any combination of points in boundary
    """
    i1 = np.sort([tri.simplices[:,0], tri.simplices[:,1]], axis=0)
    i2 = np.sort([tri.simplices[:,0], tri.simplices[:,2]], axis=0)
    i3 = np.sort([tri.simplices[:,1], tri.simplices[:,2]], axis=0)

    a = np.hstack([i1, i2, i3]).T

    # find unique rows in numpy array 
    # <http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array>
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    edges = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

    ix = np.in1d(edges.ravel(), boundary).reshape(edges.shape)
    boundary2 = ix.sum(axis=1)
    # both points are boundary points that share the line segment
    boundary_edges = edges[boundary2==2]
    return boundary_edges


def create_DMPlex_from_points(x, y, bmask=None, refinement_steps=0):
    """
    Triangulates x,y coordinates on rank 0 and creates a PETSc DMPlex object
    from the cells and vertices to distribute among processors.

    Parameters
    ----------
     x : array of floats, shape (n,)
        x coordinates
     y : array of floats, shape (n,)
        y coordinates
     bmask : array of bools, shape (n,)
        boundary mask where points along the boundary
        equal False, and the interior equal True
        if bmask=None (default) then the convex hull of points is used
     refinement_steps : int
        number of iterations to refine the mesh (default: 0)

    Returns
    -------
     DM : object
        PETSc DMPlex object

    Notes
    -----
     x and y are shuffled on input to aid triangulation efficiency

     Refinement adds the midpoints of every line segment to the DM.
     Boundary markers are automatically updated with each iteration.

    """
    from petsc4py import PETSc
    from stripy import Triangulation

    if PETSc.COMM_WORLD.rank == 0 or PETSc.COMM_WORLD.size == 1:
        reshuffle = np.random.permutation(x.size)
        x = x[reshuffle]
        y = y[reshuffle]
        if bmask is not None:
            bmask = bmask[reshuffle]
        tri = Triangulation(x,y)
        coords = tri.points
        cells  = tri.simplices
    else:
        coords = np.zeros((0,2), dtype=float)
        cells  = np.zeros((0,3), dtype=PETSc.IntType)

    dim = 2
    dm = PETSc.DMPlex().createFromCellList(dim, cells, coords)

    origSect = dm.createSection(1, [1,0,0]) # define one DoF on the nodes
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    pStart, eEnd = dm.getDepthStratum(0) # points in DAG

    # Label boundary points and edges
    dm.createLabel("boundary")

    if PETSc.COMM_WORLD.rank == 0:
        if bmask is None:
            ## mark edges
            dm.markBoundaryFaces("boundary")
            ## mark points
            # convert to DAG ordering
            boundary_points = tri.convex_hull() + pStart
            for pt in boundary_points:
                dm.setLabelValue("boundary", pt, 1)
        else:
            boundary_points = np.nonzero(~bmask)[0]

            # mark edges
            boundary_edges = _points_to_edges(tri, boundary_points)

            # convert to DAG ordering
            boundary_edges  += pStart
            boundary_points += pStart

            # join is the common edge to which they are connected
            for idx, e in enumerate(boundary_edges):
                join = dm.getJoin(e)
                dm.setLabelValue("boundary", join[0], 1)

            # mark points
            for pt in boundary_points:
                dm.setLabelValue("boundary", pt, 1)


    # Distribute to other processors
    origVec = dm.createGlobalVector()

    if PETSc.COMM_WORLD.size > 1:
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    # Label coarse DM in case it is ever needed again
    pStart, pEnd = dm.getDepthStratum(0)
    dm.createLabel("coarse")
    for pt in range(pStart, pEnd):
        dm.setLabelValue("coarse", pt, 1)


    # Refinement
    for i in range(0, refinement_steps):
        dm = dm.refine()

    origSect = dm.createSection(1, [1,0,0]) # define one DoF on the nodes
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    dm.stratify()
    return dm


def create_DMPlex_from_box(minX, maxX, minY, maxY, resX, resY, refinement=None):
    """
    Create a box and fill with triangles up to a specified refinement
    - This only works if PETSc was configured with triangle
    """
    from petsc4py import PETSc

    dm = PETSc.DMPlex().create()
    dm.setDimension(1)
    boundary = dm.createSquareBoundary([minX,minY], [maxX,maxY], [resX,resY])
    dm.generate(boundary, name='triangle')
    if refinement:
        dm.setRefinementLimit(refinement) # Maximum cell volume
        dm = dm.refine()
    dm.markBoundaryFaces('BC')

    pStart, pEnd = dm.getChart()

    origSect = dm.createSection(1, [1,0,0])
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    origVec = dm.createGlobalVec()

    if PETSc.COMM_WORLD.size > 1:
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    dm.stratify()
    return dm


def create_DMDA(minX, maxX, minY, maxY, resX, resY):
    """
    Create a PETSc DMDA object from the bounding box of the regularly-
    spaced grid.
    """
    from petsc4py import PETSc

    dx = (maxX - minX)/resX
    dy = (maxY - minY)/resY

    if dx != dy:
        raise ValueError("Spacing must be uniform in x and y directions")

    dim = 2
    dm = PETSc.DMDA().create(dim, sizes=(resX, resY), stencil_width=1)
    dm.setUniformCoordinates(minX, maxX, minY, maxY)
    return dm



def lloyd_mesh_improvement(x, y, bmask, iterations):
    """
    Applies Lloyd's algorithm of iterated voronoi construction 
    to improve the mesh point locations (assumes no current triangulation)

    (e.g. see http://en.wikipedia.org/wiki/Lloyd's_algorithm )

    We do not move boundary points, but some issues can arise near
    boundaries if the initial mesh is poorly constructed with non-boundary points
    very close to the boundary such that the centroid of the cell falls outside the boundary.

    Caveat emptor !
    """

    from scipy.spatial import Voronoi  as __Voronoi


    points = np.column_stack((x,y))

    for i in range(0,iterations):
        vor = __Voronoi(points)
        new_coords = vor.points.copy()

        for centre_point, coords in enumerate(vor.points):
            region = vor.regions[vor.point_region[centre_point]]
            if not -1 in region and bmask[centre_point]:
                polygon = vor.vertices[region]      
                new_coords[centre_point] = [polygon[:,0].sum() / len(region), polygon[:,1].sum() / len(region)]
              
        points = new_coords

    x = np.array(new_coords[:,0])
    y = np.array(new_coords[:,1])  
     
    return x, y


def square_mesh(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ):
    


    lin_samples = int(np.sqrt(samples))

    tiX = np.linspace(minX + 0.75 * spacingX, maxX - 0.75 * spacingX, lin_samples) 
    tiY = np.linspace(minY + 0.75 * spacingY, maxY - 0.75 * spacingY, lin_samples) 

    x,y = np.meshgrid(tiX, tiY)

    x = np.reshape(x,len(x)*len(x[0]))
    y = np.reshape(y,len(y)*len(y[0]))

    xscale = (x.max()-x.min()) / (2.0 * np.sqrt(samples))
    yscale = (y.max()-y.min()) / (2.0 * np.sqrt(samples))

    x += xscale * (0.5 - np.random.rand(len(x)))
    y += yscale * (0.5 - np.random.rand(len(y)))


    bmask = np.ones_like(x, dtype="Bool") # It's all true !

    # add boundary points too 

    x = np.append(x, np.linspace(minX, maxX, boundary_samples) )
    y = np.append(y, np.ones(boundary_samples)*minY )
    bmask = np.append(bmask, np.zeros(boundary_samples, dtype="Bool"))

    x = np.append(x, np.linspace(minX, maxX, boundary_samples) )
    y = np.append(y, np.ones(boundary_samples)*maxY )
    bmask = np.append(bmask, np.zeros(boundary_samples, dtype="Bool"))

    x = np.append(x, np.ones(boundary_samples)[1:-1] * minX )
    y = np.append(y, np.linspace(minY, maxY, boundary_samples)[1:-1] )
    bmask = np.append(bmask, np.zeros(boundary_samples-2, dtype="Bool"))

    x = np.append(x, np.ones(boundary_samples)[1:-1] * maxX )
    y = np.append(y, np.linspace(minY, maxY, boundary_samples)[1:-1] )
    bmask = np.append(bmask, np.zeros(boundary_samples-2, dtype="Bool"))

    # mask: need to keep the boundary conditions from being changed but equally have them available

    return x, y, bmask


def elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ): 



    originX = 0.5 * (maxX + minX)
    originY = 0.5 * (maxY + minY)
    radiusX = 0.5 * (maxX - minX)
    aspect = 0.5 * (maxY - minY) / radiusX
    
    print "Origin = ", originX, originY, "Radius = ", radiusX, "Aspect = ", aspect
    
    lin_samples = int(np.sqrt(samples))

    tiX = np.linspace(minX + 0.75 * spacingX, maxX - 0.75 * spacingX, lin_samples) 
    tiY = np.linspace(minY + 0.75 * spacingY, maxY - 0.75 * spacingY, lin_samples) 

    x,y = np.meshgrid(tiX, tiY)

    x = np.reshape(x,len(x)*len(x[0]))
    y = np.reshape(y,len(y)*len(y[0]))

    xscale = (x.max()-x.min()) / (2.0 * np.sqrt(samples))
    yscale = (y.max()-y.min()) / (2.0 * np.sqrt(samples))

    x += xscale * (0.5 - np.random.rand(len(x)))
    y += yscale * (0.5 - np.random.rand(len(y)))

    mask = np.where( (x**2 + y**2 / aspect**2) < (radiusX-0.5*spacingX)**2 )

    X = x[mask]
    Y = y[mask]
    bmask = np.ones_like(X, dtype=bool)

    # Now add boundary points
    
    theta = np.array( [ 2.0 * np.pi * i / (3 * boundary_samples) for i in range(0, 3 * boundary_samples) ])
        
    X = np.append(X, 1.001 * radiusX * np.sin(theta))    
    Y = np.append(Y, 1.001 * radiusX * aspect * np.cos(theta))    
    bmask = np.append(bmask, np.zeros_like(theta, dtype=bool))

    return X, Y, bmask
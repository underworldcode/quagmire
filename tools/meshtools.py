"""
Mesh generation functions

"""


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



def create_DMPlex_from_points(x, y, bmask=None, convex_hull=False):
    """
    Triangulates x,y coordinates and creates a PETSc DMPlex object
    from the cells and vertices.

    bmask is ignored if convex_hull=True
    """
    from petsc4py import PETSc
    import numpy as np

    try:
        import triangle
        triangulation = _Triangulation
        ConvexHull = _ConvexHull
    except ImportError:
        from scipy.spatial import Delaunay as triangulation
        from scipy.spatial import ConvexHull
    except ImportError:
        raise ImportError("Install triangle or scipy to triangulate points")


    if PETSc.COMM_WORLD.rank == 0 or PETSc.COMM_WORLD.size == 1:
        coords = np.column_stack((x,y))
        tri = triangulation(coords)
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

    dm.createLabel("boundary")
    pStart, eEnd = dm.getDepthStratum(0) # points in DAG

    if convex_hull:
        hull = ConvexHull(tri.points)
        # convert to DAG ordering
        boundary_points = hull.vertices + pStart
        for pt in boundary_points:
            dm.setLabelValue("boundary", pt, 1)

    elif bmask is not None:
        # convert to DAG ordering
        boundary_points = np.nonzero(~bmask)[0] + pStart
        for pt in boundary_points:
            dm.setLabelValue("boundary", pt, 1)

    else:
        raise ValueError("Set convex_hull to True or assign bmask")


    # Distribute to other processors
    origVec = dm.createGlobalVector()

    if PETSc.COMM_WORLD.size > 1:
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    dm.stratify()
    return dm


def create_DMPlex_from_box(minX, maxX, minY, maxY, resX, resY, refinement=None):
    """
    Create a box and fill with triangles up to a specified refinement
    - This only works if PETSc was configured with triangle
    """
    from petsc4py import PETSc
    import numpy as np

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
    import numpy as np

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
    import numpy as np


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
    
    import numpy as np


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

    import numpy as np


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
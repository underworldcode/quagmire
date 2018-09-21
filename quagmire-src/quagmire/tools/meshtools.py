"""
Copyright 2016-2017 Louis Moresi, Ben Mather, Romain Beucher

This file is part of Quagmire.

Quagmire is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or any later version.

Quagmire is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

try: range = xrange
except: pass


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
    from stripy import Triangulation

    def points_to_edges(tri, boundary):
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

    tri = Triangulation(x,y, permute=True)

    if type(bmask) == type(None):
        hull = tri.convex_hull()
        boundary_vertices = np.column_stack([hull, np.hstack([hull[1:], hull[0]])])
    else:
        boundary_indices = np.nonzero(~bmask)[0]
        boundary_vertices = points_to_edges(tri, boundary_indices)

    return create_DMPlex(tri.x, tri.y, tri.simplices, boundary_vertices)



def set_DMPlex_boundary_points(dm):
    """
    Finds the points that join the edges that have been
    marked as "boundary" faces in the DAG then sets them
    as boundaries.
    """
    pStart, pEnd = dm.getDepthStratum(0) # points
    eStart, eEnd = dm.getDepthStratum(1) # edges
    edgeIS = dm.getStratumIS('boundary', 1)

    edge_mask = np.logical_and(edgeIS.indices >= eStart, edgeIS.indices < eEnd)
    boundary_edges = edgeIS.indices[edge_mask]

    # query the DAG for points that join an edge
    for edge in boundary_edges:
        vertices = dm.getCone(edge)
        # mark the boundary points
        for vertex in vertices:
            dm.setLabelValue("boundary", vertex, 1)

def set_DMPlex_boundary_points_and_edges(dm, boundary_vertices):
    """ Label boundary points and edges """

    from petsc4py import PETSc

    if np.ndim(boundary_vertices) != 2 and np.shape(boundary_vertices)[1] != 2:
        raise ValueError("boundary vertices must be of shape (n,2)")

    # points in DAG
    pStart, eEnd = dm.getDepthStratum(0)

    # convert to DAG ordering
    boundary_edges = np.array(boundary_vertices + pStart, dtype=PETSc.IntType)
    boundary_indices = np.array(np.unique(boundary_edges), dtype=PETSc.IntType)

    # mark edges
    for edge in boundary_edges:
        # join is the common edge to which they are connected
        join = dm.getJoin(edge)
        for j in join:
            dm.setLabelValue("boundary", j, 1)

    # mark points
    for ind in boundary_indices:
        dm.setLabelValue("boundary", ind, 1)

def get_boundary_points(dm):

    pStart, pEnd = dm.getDepthStratum(0) # points
    eStart, eEnd = dm.getDepthStratum(1) # edges
    edgeIS = dm.getStratumIS('boundary', 1)

    edge_mask = np.logical_and(edgeIS.indices >= eStart, edgeIS.indices < eEnd)
    boundary_edges = edgeIS.indices[edge_mask]

    boundary_vertices = np.empty((boundary_edges.size,2), dtype=PETSc.IntType)

    # query the DAG for points that join an edge
    for idx, edge in enumerate(boundary_edges):
        boundary_vertices[idx] = dm.getCone(edge)

    # convert to local point ordering
    boundary_vertices -= pStart
    return np.unique(boundary_vertices)



def create_DMPlex_from_hdf5(file):
    """
    Creates a DMPlex object from an HDF5 file.
    This is useful for rebuilding a mesh that is saved from a
    previous simulation.

    Parameters
    ----------
     file : string
        point to the location of hdf5 file

    Returns
    -------
     DM : object
        PETSc DMPlex object

    Notes
    -----
     This function requires petsc4py >= 3.8
    """
    from petsc4py import PETSc

    file = str(file)
    if not file.endswith('.h5'):
        file += '.h5'

    dm = PETSc.DMPlex().createFromFile(file)

    # define one DoF on the nodes
    origSect = dm.createSection(1, [1,0,0])
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    origVec = dm.createGlobalVector()

    if PETSc.COMM_WORLD.size > 1:
        # Distribute to other processors
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    return dm


def create_DMPlex_from_box(minX, maxX, minY, maxY, resX, resY, refinement=None):
    """
    Create a box and fill with triangles up to a specified refinement
    - This only works if PETSc was configured with triangle
    """
    from petsc4py import PETSc

    nx = int((maxX - minX)/resX)
    ny = int((maxY - minY)/resY)

    dm = PETSc.DMPlex().create()
    dm.setDimension(1)
    boundary = dm.createSquareBoundary([minX,minY], [maxX,maxY], [nx,ny])
    dm.generate(boundary, name='triangle')
    if refinement:
        dm.setRefinementLimit(refinement) # Maximum cell volume
        dm = dm.refine()
    dm.markBoundaryFaces('boundary')

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
        raise ValueError("Spacing must be uniform in x and y directions [{:.4f}, {:.4f}]".format(dx,dy))

    dim = 2
    dm = PETSc.DMDA().create(dim, sizes=(resX, resY), stencil_width=1)
    dm.setUniformCoordinates(minX, maxX, minY, maxY)
    return dm


def create_DMPlex(x, y, simplices, boundary_vertices=None):
    """
    Create a PETSc DMPlex object on root processor
    and distribute to other processors

    Parameters
    ----------
     x : array of floats, shape (n,) x coordinates
     y : array of floats, shape (n,) y coordinates
     simplices : connectivity of the mesh
     boundary_vertices : array of ints, shape(l,2)
        (optional) boundary edges

    Returns
    -------
     DM : PETSc DMPlex object
    """
    from petsc4py import PETSc

    if PETSc.COMM_WORLD.rank == 0:
        coords = np.column_stack([x,y])
        cells  = simplices.astype(PETSc.IntType)
    else:
        coords = np.zeros((0,2), dtype=np.float)
        cells  = np.zeros((0,3), dtype=PETSc.IntType)

    dim = 2
    dm = PETSc.DMPlex().createFromCellList(dim, cells, coords)

    # create labels
    dm.createLabel("boundary")
    dm.createLabel("coarse")

    ## label boundary
    if type(boundary_vertices) == type(None):
        # boundary is convex hull
        # mark edges and points
        dm.markBoundaryFaces("boundary")
        set_DMPlex_boundary_points(dm)
    else:
        # boundary is convex hull
        set_DMPlex_boundary_points_and_edges(dm, boundary_vertices)

    ## label coarse DM in case it is ever needed again
    pStart, pEnd = dm.getDepthStratum(0)
    for pt in range(pStart, pEnd):
        dm.setLabelValue("coarse", pt, 1)


    # define one DoF on the nodes
    origSect = dm.createSection(1, [1,0,0])
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    origVec = dm.createGlobalVector()

    if PETSc.COMM_WORLD.size > 1:
        # Distribute to other processors
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    return dm


def save_DM_to_hdf5(dm, file):
    """
    Saves mesh information stored in the DM to HDF5 file
    If the file already exists, it is overwritten.
    """
    from petsc4py import PETSc

    file = str(file)
    if not file.endswith('.h5'):
        file += '.h5'

    ViewHDF5 = PETSc.Viewer()
    ViewHDF5.createHDF5(file, mode='w')
    ViewHDF5.view(obj=dm)
    ViewHDF5.destroy()
    return


def refine_DM(dm, refinement_steps=1):
    """
    Refine DM a specified number of refinement steps
    For each step, the midpoint of every line segment is added
    to the DM.
    """

    for i in range(0, refinement_steps):
        dm = dm.refine()

    origSect = dm.createSection(1, [1,0,0]) # define one DoF on the nodes
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    dm.stratify()
    return dm



def lloyd_mesh_improvement(x, y, bmask, iterations):
    """
    Applies Lloyd's algorithm of iterated voronoi construction
    to improve the mesh point locations (assumes no current triangulation)

    (e.g. see http://en.wikipedia.org/wiki/Lloyd's_algorithm )

    This can be very slow for anything but a small mesh.

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

## These are not very well cooked - we need boundary points etc

def square_mesh(minX, maxX, minY, maxY, spacingX, spacingY, random_scale=0.0, refinement_levels=0):
    """
    Generate a square mesh using stripy
    """
    from stripy import cartesian_meshes

    extent_xy = [minX, maxX, minY, maxY]

    tri = cartesian_meshes.square_mesh(extent_xy, spacingX, spacingY, random_scale, refinement_levels)

    return tri.x, tri.y, tri.simplices


def elliptical_mesh(minX, maxX, minY, maxY, spacingX, spacingY, random_scale=0.0, refinement_levels=0):
    """
    Generate an elliptical mesh using stripy
    """
    from stripy import cartesian_meshes

    extent_xy = [minX, maxX, minY, maxY]

    tri = cartesian_meshes.elliptical_mesh(extent_xy, spacingX, spacingY, random_scale, refinement_levels)

    return tri.x, tri.y, tri.simplices


def generate_square_points(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ):

    lin_samples = int(np.sqrt(samples))

    tiX = np.linspace(minX + 0.75 * spacingX, maxX - 0.75 * spacingX, lin_samples)
    tiY = np.linspace(minY + 0.75 * spacingY, maxY - 0.75 * spacingY, lin_samples)

    x,y = np.meshgrid(tiX, tiY)

    x = x.ravel()
    y = y.ravel()

    xscale = (x.max()-x.min()) / (2.0 * np.sqrt(samples))
    yscale = (y.max()-y.min()) / (2.0 * np.sqrt(samples))

    x += xscale * (0.5 - np.random.rand(x.size))
    y += yscale * (0.5 - np.random.rand(y.size))


    bmask = np.ones_like(x, dtype=bool) # It's all true !

    # add boundary points too

    xc = np.linspace(minX, maxX, boundary_samples)
    yc = np.linspace(minY, maxY, boundary_samples)

    i = 1.0 - np.linspace(-0.5, 0.5, boundary_samples)**2

    x = np.append(x, xc)
    y = np.append(y, minY - spacingY*i)

    x = np.append(x, xc)
    y = np.append(y, maxY + spacingY*i)

    x = np.append(x, minX - spacingX*i[1:-1])
    y = np.append(y, yc[1:-1])

    x = np.append(x, maxX + spacingX*i[1:-1])
    y = np.append(y, yc[1:-1])

    bmask = np.append(bmask, np.zeros(2*i.size + 2*(i.size-2), dtype=bool))

    return x, y, bmask


def generate_elliptical_points(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ):

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

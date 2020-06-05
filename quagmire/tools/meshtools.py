# Copyright 2016-2020 Louis Moresi, Ben Mather, Romain Beucher
# 
# This file is part of Quagmire.
# 
# Quagmire is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
# 
# Quagmire is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.

"""
Tools for creating Quagmire meshes.

Quagmire constructs meshes as `PETSc DMPlex` and `DMDA` objects.

## Regular Meshes

Create a __regularly-spaced Cartesian__ grid with `PETSc DMDA` using:

- `create_DMDA`

## Unstructured Cartesian meshes

Create an __unstructured Cartesian__ mesh with `PETSc DMPlex` using:

- `create_DMPlex`
- `create_DMPlex_from_points`
- `create_DMPlex_from_box`

## Unstructured Spherical meshes

Create an __unstructured Spherical__ mesh with `PETSc DMPlex` using:

- `create_spherical_DMPlex`
- `create_DMPlex_from_spherical_points`

Reconstruct a mesh that has been saved to an HDF5 file:

- `create_DMPlex_from_hdf5`


### Additional tools

Refine any mesh with `refine_DM`. This adds the midpoint of all line
segments to the mesh.

Save any mesh to an HDF5 file with `save_DM_to_hdf5`.

"""

import numpy as _np

try: range = xrange
except: pass


def points_to_edges(tri, boundary):
    """
    Finds the edges connecting two boundary points
    """
    i1 = _np.sort([tri.simplices[:,0], tri.simplices[:,1]], axis=0)
    i2 = _np.sort([tri.simplices[:,0], tri.simplices[:,2]], axis=0)
    i3 = _np.sort([tri.simplices[:,1], tri.simplices[:,2]], axis=0)

    a = _np.hstack([i1, i2, i3]).T

    # find unique rows in numpy array
    edges = _np.unique(a, axis=0)

    # label where boundary nodes are
    ix = _np.in1d(edges.ravel(), boundary).reshape(edges.shape)
    boundary2 = ix.sum(axis=1)

    # both points are boundary points that share the line segment
    boundary_edges = edges[boundary2==2]
    return boundary_edges


def find_boundary_segments(simplices):
    """
    Finds all boundary segments in the triangulation.

    Boundary segments should not share a segment with another triangle.

    Parameters
    ----------
    simplices : array shape (nt,3)
        list of simplices in a triangulaton.

    Returns
    -------
    segments : array shape (n,2)
        list of segments that define the convex hull (boundary)
        of the triangulation.

    Notes
    -----
    Spherical meshes generated with stripy will not have any boundary
    segments, thus an empty list is returned.
    """
    i1 = _np.sort([simplices[:,0], simplices[:,1]], axis=0)
    i2 = _np.sort([simplices[:,0], simplices[:,2]], axis=0)
    i3 = _np.sort([simplices[:,1], simplices[:,2]], axis=0)
    
    a = _np.hstack([i1, i2, i3]).T

    # find unique rows in numpy array
    edges, counts = _np.unique(a, return_counts=True, axis=0)
    return edges[counts < 2]


def create_DMPlex_from_points(x, y, bmask=None, refinement_levels=0):
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
    refinement_levels : int
        number of iterations to refine the mesh (default: 0)

    Returns
    -------
    DM : PETSc DMPlex object

    Notes
    -----
    `x` and `y` are shuffled on input to aid triangulation efficiency

    Refinement adds the midpoints of every line segment to the DM.
    Boundary markers are automatically updated with each iteration.
    """
    from stripy import Triangulation

    tri = Triangulation(x,y, permute=True)

    if bmask is None:
        hull = tri.convex_hull()
        boundary_vertices = _np.column_stack([hull, _np.hstack([hull[1:], hull[0]])])
    else:
        boundary_indices = _np.nonzero(~bmask)[0]
        boundary_vertices = points_to_edges(tri, boundary_indices)

    return _create_DMPlex(tri.points, tri.simplices, boundary_vertices, refinement_levels)


def create_DMPlex_from_spherical_points(lons, lats, bmask=None, refinement_levels=0):
    """
    Triangulates lon,lat coordinates on rank 0 and creates a PETSc DMPlex object
    from the cells and vertices to distribute among processors.

    Parameters
    ----------
    lons : array of floats, shape (n,)
        longitudinal coordinates in radians
    lats : array of floats, shape (n,)
        latitudinal coordinates in radians
    bmask : array of bools, shape (n,)
        boundary mask where points along the boundary
        equal False, and the interior equal True
        if bmask=None (default) then the convex hull of points is used
    refinement_levels : int
        number of iterations to refine the mesh (default: 0)

    Returns
    -------
    DM : PETSc DMPlex object

    Notes
    -----
    `lons` and `lats` are shuffled on input to aid triangulation efficiency

    Refinement adds the midpoints of every line segment to the DM.
    Boundary markers are automatically updated with each iteration.

    """
    from stripy import sTriangulation

    tri = sTriangulation(lons, lats, permute=True)

    if bmask is None:
        boundary_vertices = find_boundary_segments(tri.simplices)
    else:
        boundary_indices = _np.nonzero(~bmask)[0]
        boundary_vertices = points_to_edges(tri, boundary_indices)

    return _create_DMPlex(tri.points, tri.simplices, boundary_vertices, refinement_levels)



def set_DMPlex_boundary_points(dm):
    """
    Finds the points that join the edges that have been
    marked as "boundary" faces in the DAG then sets them
    as boundaries.
    """

    pStart, pEnd = dm.getDepthStratum(0) # points
    eStart, eEnd = dm.getDepthStratum(1) # edges
    edgeIS = dm.getStratumIS('boundary', 1)

    if eEnd == eStart:
        ## CAUTION: don't do this if any of the dm calls have barriers
        return

    edge_mask = _np.logical_and(edgeIS.indices >= eStart, edgeIS.indices < eEnd)
    boundary_edges = edgeIS.indices[edge_mask]

    # query the DAG for points that join an edge
    for edge in boundary_edges:
        vertices = dm.getCone(edge)
        # mark the boundary points
        for vertex in vertices:
            dm.setLabelValue("boundary", vertex, 1)

    return

def set_DMPlex_boundary_points_and_edges(dm, boundary_vertices):
    """ Label boundary points and edges """

    from petsc4py import PETSc

    if _np.ndim(boundary_vertices) != 2 and _np.shape(boundary_vertices)[1] != 2:
        raise ValueError("boundary vertices must be of shape (n,2)")

    # points in DAG
    pStart, pEnd = dm.getDepthStratum(0)

    if pStart == pEnd:
        ## CAUTION not if there is a barrier in any of the dm calls that lie ahead.
        return

    # convert to DAG ordering
    boundary_edges = _np.array(boundary_vertices + pStart, dtype=PETSc.IntType)
    boundary_indices = _np.array(_np.unique(boundary_edges), dtype=PETSc.IntType)

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

    edge_mask = _np.logical_and(edgeIS.indices >= eStart, edgeIS.indices < eEnd)
    boundary_edges = edgeIS.indices[edge_mask]

    boundary_vertices = _np.empty((boundary_edges.size,2), dtype=PETSc.IntType)

    # query the DAG for points that join an edge
    for idx, edge in enumerate(boundary_edges):
        boundary_vertices[idx] = dm.getCone(edge)

    # convert to local point ordering
    boundary_vertices -= pStart
    return _np.unique(boundary_vertices)



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
    DM : PETSc DMPlex object

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
    dm.setNumFields(1)
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

    Parameters
    ----------
    minX : float
        minimum x-coordinate
    maxX : float
        maximum x-coordinate
    minY : float
        minimum y-coordinate
    maxY : float
        maximum y-coordinate
    resX : float
        resolution in the x direction
    resY : float
        resolution in the y direction
    refinement : float (default: None)
        (optional) set refinement limit

    Returns
    -------
    DM : PETSc DMPlex object

    Notes
    -----
    This only works if PETSc was configured with triangle
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

    dm.setNumFields(1)
    origSect = dm.createSection(1, [1,0,0])
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

    origVec = dm.createGlobalVec()

    if PETSc.COMM_WORLD.size > 1:
        sf = dm.distribute(overlap=1)
        newSect, newVec = dm.distributeField(sf, origSect, origVec)
        dm.setDefaultSection(newSect)

    return dm


def create_DMDA(minX, maxX, minY, maxY, resX, resY):
    """
    Create a PETSc DMDA mesh object from the bounding box of the
    regularly-spaced grid.

    Parameters
    ----------
    minX : float
        minimum x-coordinate
    maxX : float
        maximum x-coordinate
    minY : float
        minimum y-coordinate
    maxY : float
        maximum y-coordinate
    resX : float
        resolution in the x direction
    resY : float
        resolution in the y direction
    refinement : float (default: None)
        (optional) set refinement limit

    Returns
    -------
    DM : PETSc DMDA object
    """
    from petsc4py import PETSc

    dx = (maxX - minX)/resX
    dy = (maxY - minY)/resY

    if dx != dy:
        raise ValueError("Spacing must be uniform in x and y directions [{:.4f}, {:.4f}]".format(dx,dy))

    dim = 2
    dm = PETSc.DMDA().create(dim, sizes=(resX, resY), stencil_width=1)
    dm.setUniformCoordinates(minX, maxX, minY, maxY)
    dm.createLabel("PixMesh")
    return dm


def _create_DMPlex(points, simplices, boundary_vertices=None, refinement_levels=0):
    """
    Create a PETSc DMPlex object on root processor
    and distribute to other processors

    Parameters
    ----------
     points : array of floats, shape (n,dim) coordinates
     simplices : connectivity of the mesh
     boundary_vertices : array of ints, shape(l,2)
        (optional) boundary edges

    Returns
    -------
     DM : PETSc DMPlex object
    """
    from petsc4py import PETSc

    ndim = _np.shape(points)[1]
    mesh_type = {2: "TriMesh", 3: "sTriMesh"}

    if PETSc.COMM_WORLD.rank == 0:
        coords = _np.array(points, dtype=_np.float)
        cells  = simplices.astype(PETSc.IntType)
    else:
        coords = _np.zeros((0,ndim), dtype=_np.float)
        cells  = _np.zeros((0,3), dtype=PETSc.IntType)

    dim = 2
    dm = PETSc.DMPlex().createFromCellList(dim, cells, coords)

    # create labels
    # these can be accessed with dm.getLabelName(i) where i=0,1,2
    # and the last label added is the 0-th index.
    dm.createLabel("boundary")
    dm.createLabel("coarse")
    dm.createLabel(mesh_type[ndim])

    ## label boundary
    if boundary_vertices is None:
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
    dm.setNumFields(1)
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

    # parallel mesh refinement
    dm = refine_DM(dm, refinement_levels)

    return dm


def create_DMPlex(x, y, simplices, boundary_vertices=None, refinement_levels=0):
    """
    Create a PETSc DMPlex object on root processor
    and distribute to other processors

    Parameters
    ----------
    x : array of floats shape (n,)
        x coordinates
    y : array of floats shape (n,)
        y coordinates
    simplices : array of ints shape (nt,3)
        connectivity of the mesh
    boundary_vertices : array of ints, shape(l,2)
        (optional) boundary edges

    Returns
    -------
    DM : PETSc DMPlex object
    """
    points = _np.c_[x,y]
    return _create_DMPlex(points, simplices, boundary_vertices, refinement_levels)


def create_spherical_DMPlex(lons, lats, simplices, boundary_vertices=None, refinement_levels=0):
    """
    Create a PETSc DMPlex object on root processor
    and distribute to other processors

    Parameters
    ----------
    lons : array of floats shape (n,)
        longitudinal coordinates
    lats : array of floats shape (n,)
        latitudinal coordinates
    simplices : connectivity of the mesh
    boundary_vertices : array of ints, shape(l,2)
        (optional) boundary edges

    Returns
    -------
    DM : PETSc DMPlex object
    """
    from stripy.spherical import lonlat2xyz

    # convert to xyz to construct the DM
    x,y,z = lonlat2xyz(lons, lats)
    points = _np.c_[x,y,z]

    # PETSc's markBoundaryFaces routine cannot detect boundary segments
    # for our spherical implementation. We do it here instead.
    if boundary_vertices is None:
        boundary_vertices = find_boundary_segments(simplices)

    return _create_DMPlex(points, simplices, boundary_vertices, refinement_levels)


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


def refine_DM(dm, refinement_levels=1):
    """
    Refine DM a specified number of refinement steps
    For each step, the midpoint of every line segment is added
    to the DM.
    """

    for i in range(0, refinement_levels):
        dm = dm.refine()

    dm.setNumFields(1)
    origSect = dm.createSection(1, [1,0,0]) # define one DoF on the nodes
    origSect.setFieldName(0, "points")
    origSect.setUp()
    dm.setDefaultSection(origSect)

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


    points = _np.c_[x,y]

    for i in range(0,iterations):
        vor = __Voronoi(points)
        new_coords = vor.points.copy()

        for centre_point, coords in enumerate(vor.points):
            region = vor.regions[vor.point_region[centre_point]]
            if not -1 in region and bmask[centre_point]:
                polygon = vor.vertices[region]
                new_coords[centre_point] = [polygon[:,0].sum() / len(region), polygon[:,1].sum() / len(region)]

        points = new_coords

    x = _np.array(new_coords[:,0])
    y = _np.array(new_coords[:,1])

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



def global_CO_mesh(stripy_mesh_name, include_face_points=False, refinement_C=7, refinement_O=4, verbose=False, return_heights=False):
    """
    Returns a mesh for global problems with different resolution in ocean and continental regions with
    the transition between meshes determined using the ETOPO1 contour at -100m 
    
    Valid stripy_mesh_name values are "icosahedral_mesh", "triangulated_soccerball_mesh", and "octahedral_mesh"

    This has additonal dependencies of xarray and scipy and functools
    """
        
    from functools import partial
    import stripy
    import xarray
    import numpy as np
    
    strimesh = {"icosahedral_mesh": partial(stripy.spherical_meshes.icosahedral_mesh, include_face_points=include_face_points),
                "triangulated_soccerball_mesh": stripy.spherical_meshes.triangulated_soccerball_mesh, 
                "octahedral_mesh": partial(stripy.spherical_meshes.octahedral_mesh, include_face_points=include_face_points)
               }
    
    try:
        stC = strimesh[stripy_mesh_name](refinement_levels = refinement_C, tree=False)
        stO = strimesh[stripy_mesh_name](refinement_levels = refinement_O, tree=False)
    except:
        print("Suitable mesh types (stripy_mesh_name):")
        print("     - icosahedral_mesh")
        print("     - triangulated_soccerball_mesh")
        print("     - octahedral_mesh" )
        raise 
     

    etopo_dataset = "http://thredds.socib.es/thredds/dodsC/ancillary_data/bathymetry/ETOPO1_Bed_g_gmt4.nc"
    etopo_data = xarray.open_dataset(etopo_dataset)
    etopo_coarse = etopo_data.sel(x=slice(-180.0,180.0,30), y=slice(-90.0,90.0,30))

    lons = etopo_coarse.coords.get('x')
    lats = etopo_coarse.coords.get('y')
    vals = etopo_coarse['z']

    x,y = np.meshgrid(lons.data, lats.data)
    height = vals.data 
    height = 6.370 + 1.0e-6 * vals.data 

    meshheightsC = map_raster_to_mesh(stC, height)
    meshheightsO = map_raster_to_mesh(stO, height)

    clons = stC.lons[np.where(meshheightsC >= 6.3699)]  # 100m depth
    clats = stC.lats[np.where(meshheightsC >= 6.3699)]

    olons = stO.lons[np.where(meshheightsO < 6.3699)]  # 100m depth
    olats = stO.lats[np.where(meshheightsO < 6.3699)]

    nlons = np.hstack((clons, olons))
    nlats = np.hstack((clats, olats))
    nheights = np.hstack((meshheightsC, meshheightsO))

    stN = stripy.spherical.sTriangulation(lons=nlons, lats=nlats, refinement_levels=0, tree=False)
   
    if return_heights:
        return stN.lons, stN.lats, stN.simplices, nheights
    else:
        return stN.lons, stN.lats, stN.simplices
    


def generate_square_points(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ):

    lin_samples = int(_np.sqrt(samples))

    tiX = _np.linspace(minX + 0.75 * spacingX, maxX - 0.75 * spacingX, lin_samples)
    tiY = _np.linspace(minY + 0.75 * spacingY, maxY - 0.75 * spacingY, lin_samples)

    x,y = _np.meshgrid(tiX, tiY)

    x = x.ravel()
    y = y.ravel()

    xscale = (x.max()-x.min()) / (2.0 * _np.sqrt(samples))
    yscale = (y.max()-y.min()) / (2.0 * _np.sqrt(samples))

    x += xscale * (0.5 - _np.random.rand(x.size))
    y += yscale * (0.5 - _np.random.rand(y.size))


    bmask = _np.ones_like(x, dtype=bool) # It's all true !

    # add boundary points too

    xc = _np.linspace(minX, maxX, boundary_samples)
    yc = _np.linspace(minY, maxY, boundary_samples)

    i = 1.0 - _np.linspace(-0.5, 0.5, boundary_samples)**2

    x = _np.append(x, xc)
    y = _np.append(y, minY - spacingY*i)

    x = _np.append(x, xc)
    y = _np.append(y, maxY + spacingY*i)

    x = _np.append(x, minX - spacingX*i[1:-1])
    y = _np.append(y, yc[1:-1])

    x = _np.append(x, maxX + spacingX*i[1:-1])
    y = _np.append(y, yc[1:-1])

    bmask = _np.append(bmask, _np.zeros(2*i.size + 2*(i.size-2), dtype=bool))

    return x, y, bmask


def generate_elliptical_points(minX, maxX, minY, maxY, spacingX, spacingY, samples, boundary_samples ):

    originX = 0.5 * (maxX + minX)
    originY = 0.5 * (maxY + minY)
    radiusX = 0.5 * (maxX - minX)
    aspect = 0.5 * (maxY - minY) / radiusX

    print("Origin = ", originX, originY, "Radius = ", radiusX, "Aspect = ", aspect)

    lin_samples = int(_np.sqrt(samples))

    tiX = _np.linspace(minX + 0.75 * spacingX, maxX - 0.75 * spacingX, lin_samples)
    tiY = _np.linspace(minY + 0.75 * spacingY, maxY - 0.75 * spacingY, lin_samples)

    x,y = _np.meshgrid(tiX, tiY)

    x = _np.reshape(x,len(x)*len(x[0]))
    y = _np.reshape(y,len(y)*len(y[0]))

    xscale = (x.max()-x.min()) / (2.0 * _np.sqrt(samples))
    yscale = (y.max()-y.min()) / (2.0 * _np.sqrt(samples))

    x += xscale * (0.5 - _np.random.rand(len(x)))
    y += yscale * (0.5 - _np.random.rand(len(y)))

    mask = _np.where( (x**2 + y**2 / aspect**2) < (radiusX-0.5*spacingX)**2 )

    X = x[mask]
    Y = y[mask]
    bmask = _np.ones_like(X, dtype=bool)

    # Now add boundary points

    theta = _np.array( [ 2.0 * _np.pi * i / (3 * boundary_samples) for i in range(0, 3 * boundary_samples) ])

    X = _np.append(X, 1.001 * radiusX * _np.sin(theta))
    Y = _np.append(Y, 1.001 * radiusX * aspect * _np.cos(theta))
    bmask = _np.append(bmask, _np.zeros_like(theta, dtype=bool))

    return X, Y, bmask


def map_raster_to_mesh(mesh, latlongrid, order=3, origin="lower"):
    """
    Map a lon/lat "image" (assuming origin="lower" in matplotlib parlance) to nodes on a quagmire mesh
    """
    from scipy import ndimage

    raster = latlongrid.T

    latitudes_in_radians  = mesh.lats
    longitudes_in_radians = mesh.lons 
    latitudes_in_degrees  = np.degrees(latitudes_in_radians) 
    longitudes_in_degrees = np.degrees(longitudes_in_radians)%360.0 - 180.0

    dlons = np.mod(longitudes_in_degrees+180.0, 360.0)
    dlats = np.mod(latitudes_in_degrees+90, 180.0)

    if origin != "lower":
        dlats *= -1

    ilons = raster.shape[0] * dlons / 360.0
    ilats = raster.shape[1] * dlats / 180.0

    icoords = np.array((ilons, ilats))

    mvals = ndimage.map_coordinates(raster, icoords , order=order, mode='nearest').astype(np.float)

    return mvals
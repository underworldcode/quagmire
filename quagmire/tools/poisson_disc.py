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

def poisson_disc_sampler(minX, maxX, minY, maxY, spacing, k=30, r_grid=None,\
                         cpts=None, spts=None):
    """
    Poisson disc sampler in two dimensions.
    This is a flood-fill algorithm for generating points that are
    separated by a minimum radius.

    Arguments
    ---------
     minX, maxX, minY, maxY : float
        coordinates of the domain
     spacing : float
        constant radius to sample across the domain
        every point is guaranteed to be no less than this distance
        from each other
     k : int (default: k=30)
        number of random samples to generate around a single point
        30 generally gives good results
     r_grid : array of floats, optional, shape (height,width)
        support for variable radii
        radius is ignored if an array is given here
     cpts : array of floats, optional, shape (n,2)
        points that must be sampled; useful for irregular boundaries
     spts : array of floats, optional, shape (s,2)
        points used to seed the flood-fill algorithm,
        samples are generated outwards from these seed points

    Returns
    -------
     pts : array of floats, shape (N,2)
        x, y coordinates for each sample point
     cpts_mask : array of bools, shape (N,2)
        boolean array where new points are True and
        cpts are False

    Notes
    -----
     One should aim to sample around 10,000 points, much more than that
     and the algorithm slows rapidly.
    """
    def transform_to_points(coords, minX, maxX, minY, maxY, width, height):
        points = np.empty_like(coords)
        points[:,0] = width*(coords[:,0] - minX)/(maxX - minX)
        points[:,1] = height*(coords[:,1] - minY)/(maxY - minY)
        return points

    def transform_to_coords(points, minX, maxX, minY, maxY, width, height):
        coords = np.empty_like(points)
        coords[:,0] = (maxX-minX)*(points[:,0]/width) + minX
        coords[:,1] = (maxY-minY)*(points[:,1]/height) + minY
        return coords

    def distance_squared(point0, point1):
        """ squared distance is compared to squared radius """
        return (point0[0] - point1[0])**2 + (point0[1] - point1[1])**2

    def random_point_around(point, r, k=1):
        """ Generate random points within a given radius limit """
        rr = np.random.uniform(r, 2.0*r, k) # radius
        rt = np.random.uniform(0, 2.0*np.pi, k) # angle
        P = np.empty((k, dim))
        P[:,0] = point[0] + rr*np.sin(rt)
        P[:,1] = point[1] + rr*np.cos(rt)
        return P

    def neighbourhood(index):
        """ Find all the closest neighbours """
        row, col = index
        row0, row1 = max(row - dim, 0), min(row + dim + 1, rows)
        col0, col1 = max(col - dim, 0), min(col + dim + 1, cols)
        I = np.dstack(np.mgrid[row0:row1, col0:col1])
        I = I.reshape(-1, dim).tolist()
        I.remove([row, col]) # own neighbour
        return I

    def in_neighbourhood(point, r_sqr):
        """ Checks if point is in the neighbourhood """
        i, j = int(point[0]/cellsize), int(point[1]/cellsize)
        if M[i,j]:
            return True
        for (i,j) in N[(i, j)]:
            if M[i,j] and distance_squared(point, P[i,j]) < r_sqr:
                return True
        return False

    def in_limits(point):
        """ Returns True if point is within box """
        return 0 <= point[0] < width and 0 <= point[1] < height

    def add_point(point):
        """ Append point to the points list """
        points.append(point)
        i, j = int(point[0]/cellsize), int(point[1]/cellsize)
        P[i,j] = point
        M[i,j] = True

    # size of the domain
    if r_grid is not None:
        height, width = r_grid.shape
        r_min, r_max = r_grid.min(), r_grid.max()
        r1 = r_max/(maxX - minX)
        r2 = r_max/(maxY - minY)
        radius = np.mean([r1*width, r2*height]) # scale
        r_grid = r_grid*(radius/r_max)
    else:
        cellsize = spacing/np.sqrt(2.0)
        height = int(np.ceil((maxY - minY)/cellsize))
        width = int(np.ceil((maxX - minX)/cellsize))
        r1 = spacing/(maxX - minX)
        r2 = spacing/(maxY - minY)
        radius = np.mean([r1*width,r2*height]) # scale
        r_grid = np.ones((height, width)) * radius


    cellsize = radius/np.sqrt(2.0)

    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
    dim = 2


    # Position cells
    P = np.zeros((rows, cols, dim), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)
    C = np.ones((rows, cols), dtype=bool)


    # Add constraint points
    if cpts is not None:
        cpts = cpts.reshape(-1,2)
        cpts = transform_to_points(cpts, minX, maxX, minY, maxY, width, height)
        ci = (cpts[:,0]/cellsize).astype(int)
        cj = (cpts[:,1]/cellsize).astype(int)
        P[ci,cj] = cpts
        M[ci,cj] = True
        C[ci,cj] = False


    # Cache generation for neighbourhood
    N = {}
    for i in range(0, rows):
        for j in range(0, cols):
            N[(i,j)] = neighbourhood((i,j))


    # Add seed points
    points = []
    if spts is not None:
        spts = spts.reshape(-1,2)
        spts = transform_to_points(spts, minX, maxX, minY, maxY, width, height)
        for pt in spts:
            add_point(tuple(pt))
    else:
        # add a random initial point
        add_point((np.random.uniform(0, width),\
                   np.random.uniform(0, height)))

    length = len(points)
    while length:
        i = np.random.randint(0,length)
        pt = points.pop(i)
        ### This works because height, width are smaller than rows, cols
        r = r_grid[int(pt[1]),int(pt[0])]
        # r = r_min + radius[int(pt[1]),int(pt[0])]*(r_max - r_min)
        r_sqr = r**2
        qt = random_point_around(pt, r, k)
        for q in qt:
            if in_limits(q) and not in_neighbourhood(q, r_sqr):
                add_point(q)

        # re-evaluate length
        length = len(points)

    P[M] = transform_to_coords(P[M], minX, maxX, minY, maxY, width, height)
    return P[M][:,0], P[M][:,1], C[M]


def poisson_square_mesh(minX, maxX, minY, maxY, spacing, boundary_samples, r_grid=None):
    """
    Create a square mesh using the poisson disc sampler.

    Boundary points are generated around the edge of the domain and used
    as constraint points. The optimal number of boundary points should be
    the circumference of the square divided by spacing.
    """

    originX = 0.5 * (maxX + minX)
    originY = 0.5 * (maxY + minY)
    centroid = np.array([originX, originY])
    ratio = (maxY - minY)/(maxX - minX)

    minX1 = minX + 0.75 * spacing
    minY1 = minY + 0.75 * spacing
    maxX1 = maxX - 0.75 * spacing
    maxY1 = maxY - 0.75 * spacing

    x, y, bmask = poisson_disc_sampler(minX1, maxX1, minY1, maxY1, spacing,\
                                     r_grid=r_grid, spts=centroid)

    bmask = np.ones(x.size, dtype=bool)

    # now add boundary points

    boundary_samples_x = int(boundary_samples)
    boundary_samples_y = int(boundary_samples*ratio)
    bspace = 1.*spacing

    xb = np.linspace(minX, maxX, boundary_samples_x)
    yb = np.linspace(minY, maxY, boundary_samples_y)

    xi = 0.25 - np.linspace(-0.5, 0.5, boundary_samples_x)**2
    yi = 0.25 - np.linspace(-0.5, 0.5, boundary_samples_y)**2

    x = np.append(x, xb)
    y = np.append(y, minY - spacing*xi) # minY

    x = np.append(x, xb)
    y = np.append(y, maxY + spacing*xi) # maxY

    x = np.append(x, minX - spacing*yi[1:-1]) # minX
    y = np.append(y, yb[1:-1] )

    x = np.append(x, maxX + spacing*yi[1:-1]) # maxX
    y = np.append(y, yb[1:-1] )

    bmask = np.append(bmask, np.zeros(2*xb.size+2*(yb.size-2), dtype=bool))

    return x, y, bmask


def poisson_elliptical_mesh(minX, maxX, minY, maxY, spacing, boundary_samples, r_grid=None):
    """
    Create an elliptical mesh using the poisson disc sampler.

    Boundary points are generated around the edge of the domain and used
    as constraint points. The optimal number of boundary points is the
    length of a boundary divided by the spacing, e.g. (maxX-minX)/spacing.
    """
    originX = 0.5 * (maxX + minX)
    originY = 0.5 * (maxY + minY)
    radiusX = 0.5 * (maxX - minX)
    aspect = 0.5 * (maxY - minY) / radiusX

    centroid = np.array([originX, originY])

    i = np.arange(0, 3*boundary_samples, 2)
    theta = 2.*np.pi*i/(3*boundary_samples)

    X = originX + radiusX * np.sin(theta)
    Y = originY + radiusX * aspect * np.cos(theta)

    i = np.arange(1, 3*boundary_samples+1, 2)
    theta = 2.*np.pi*i/(3*boundary_samples)

    X = np.append(X, originX + (radiusX - spacing) * np.sin(theta))
    Y = np.append(Y, originY + (radiusX - spacing) * aspect * np.cos(theta))

    # these will form the bmask
    cpts = np.column_stack([X,Y])

    x, y, bmask = poisson_disc_sampler(minX, maxX, minY, maxY, spacing,\
                                       r_grid=r_grid, cpts=cpts, spts=centroid)
    return x, y, bmask


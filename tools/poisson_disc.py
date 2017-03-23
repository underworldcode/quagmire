import numpy as np
try: range = xrange
except: pass

def poisson_disc_sampler(width, height, radius, k=30, r_grid=None):
    """
    Poisson disc sampler in two dimensions
    
    """

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

    def in_neighbourhood(point, r2):
        """ Checks if point is in the neighbourhood """
        i, j = int(point[0]/cellsize), int(point[1]/cellsize)
        if M[i,j]:
            return True
        for (i,j) in N[(i, j)]:
            if M[i,j] and distance_squared(point, P[i,j]) < r2:
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


    if r_grid is not None:
        assert r_grid.shape == (height, width)
        r_min, r_max = r_grid.min(), r_grid.max()
        radius = r_max
    else:
        r_grid = np.ones((height, width)) * radius

    cellsize = radius/np.sqrt(2.0)

    rows = int(np.ceil(width/cellsize))
    cols = int(np.ceil(height/cellsize))
    dim = 2


    # Position cells
    P = np.zeros((rows, cols, dim), dtype=np.float32)
    M = np.zeros((rows, cols), dtype=bool)


    # Cache generation for neighbourhood
    N = {}
    for i in range(0, rows):
        for j in range(0, cols):
            N[(i,j)] = neighbourhood((i,j))


    points = []
    # add a random initial point
    add_point((rows//2, \
               cols//2))

    length = len(points)
    while length:
        i = np.random.randint(0,length)
        pt = points.pop(i)
        ### This works because height, width are smaller than rows, cols
        r = r_grid[int(pt[1]),int(pt[0])]
        # r = r_min + radius[int(pt[1]),int(pt[0])]*(r_max - r_min)
        r2 = r**2
        qt = random_point_around(pt, r, k)
        for q in qt:
            if in_limits(q) and not in_neighbourhood(q, r2):
                add_point(q)

        # re-evaluate length
        length = len(points)

    return P[M]



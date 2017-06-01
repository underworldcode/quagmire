subroutine ntriw ( n, x, y, nt, ltri, area, weight )
!*****************************************************************************
!! NTRIW computes the pointwise area to calculate local areas on a mesh
!
! Parameters:
! 
!   Input, integer ( kind = 4 ), n
!   number of points in the triangulation
!
!   Input, real ( kind = 8 ), x(n), y(n)
!   x and y coordinates that make up the triangulation
!
!   Input, integer ( kind = 4 ), n
!   number of points in the triangulation
!
!   Input, integer ( kind = 4 ), nt
!   number of triangles in the triangulation
!
!   Input, integer ( kind = 4 ), ltri(nt)
!   list of triangles in the triangulation
!
!   Ouput, real ( kind = 8 ), area(n), weight(n)
!   areas and weights for each point

  implicit none
  
  integer ( kind = 4 ) n,nt,i
  real ( kind = 8 ) x(n),y(n),area(n)
  integer ( kind = 4 ) weight(n)
  integer ( kind = 4 ) ltri(3,nt)
  real ( kind = 8 ) v1x,v1y,v2x,v2y
  integer ( kind = 4 ) tri(3)

!
! Get 2 sides of triangle
!
  do i = 1, nt
    tri = ltri(:,i)
    v1x = x(tri(2)) - x(tri(1))
    v1y = y(tri(2)) - y(tri(1))
    v2x = x(tri(1)) - x(tri(3))
    v2y = y(tri(1)) - y(tri(3))

    area(tri) = area(tri) + abs(v1x*v2y - v1y*v2x)
    weight(tri) = weight(tri) + 1
  end do
!
! Now we divide each element by 6
!
  area = area/6
  return
end

function euclidean ( x1, y1, x2, y2 )
  implicit none

  real ( kind = 8 ) x1, y1, x2, y2
  real ( kind = 8 ) euclidean

  euclidean = sqrt((x2 - x1)**2 + (y2 - y1)**2)
  return
end function

function argsort ( n, a )
! Returns the indices that would sort an array
  implicit none

  integer ( kind = 4 ) n, i, imin, temp1
  real ( kind = 8 ) a(n)
  integer ( kind = 4 ) argsort(n)
  real ( kind = 8 ) temp2
  real ( kind = 8 ) a2(n)

  a2 = a
  do i = 1, n
    argsort(i) = i
  end do
  do i = 1, n-1
    ! find ith smallest in 'a'
    imin = minloc(a2(i:), 1) + i - 1
    ! swap to position i in 'a' and 'argsort', if not already there
    if (imin /= i) then
      temp2 = a2(i)
      a2(i) = a2(imin)
      a2(imin) = temp2

      temp1 = argsort(i)
      argsort(i) = argsort(imin)
      argsort(imin) = temp1
    end if
  end do
  return
end function

subroutine node_neighbours ( nptr, nind, nmax, indptr, indices, node, nn, neighbours )
  implicit none

  integer ( kind = 4 ) nptr, nind, nmax, node, nn
  integer ( kind = 4 ) indptr(nptr)
  integer ( kind = 4 ) indices(nind)
! output variables
  integer ( kind = 4 ) neighbours(nmax)

  nn = size( indices(indptr(node):indptr(node + 1)) )
  neighbours(1:nn) = indices(indptr(node):indptr(node + 1))

  return
end

subroutine downhill_neighbour ( nptr, nind, nd, nmax, n, indptr, indices, height, dneighbour )
  implicit none

  integer ( kind = 4 ) nptr, nind, nd, nmax, n
  integer ( kind = 4 ) indptr(nptr) 
  integer ( kind = 4 ) indices(nind)
  real ( kind = 8 ) height(n)
! work variables
  integer ( kind = 4 ) cnt, i, j, ej, nn, enn, node, enode
  ! real ( kind = 8 ) dist(nmax)
  integer ( kind = 4 ) neighbours(nmax), eneighbours(nmax)
  real ( kind = 8 ) h
! output variables
  integer ( kind = 4 ) dneighbour(nd,n)

  do i = 1, n
    cnt = 0
    dneighbour(:,i) = i
    h = height(i)

    call node_neighbours(nptr, nind, nmax, indptr, indices, i, nn, neighbours)

    do j = 1, nn
      node = neighbours(j)
      if (height(node) < h) then
        dneighbour(i,cnt) = node
        cnt = cnt + 1
      end if
      if (cnt == nd) then
        exit
      end if
    end do
    
!***********************************    
!   look through extended neighbours
!***********************************
    if (cnt < nd) then
      do j = 1, nn
        node = neighbours(j)

        call node_neighbours(nptr, nind, nmax, indptr, indices, node, enn, eneighbours)

        do ej = 1, enn
          enode = eneighbours(ej)
          if (height(enode) < h) then
            dneighbour(i,cnt) = enode
            cnt = cnt + 1
          end if
          if (cnt == nd) then
            exit
          end if
        end do
      end do
    end if
  end do
  return
end
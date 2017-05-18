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

    weight(tri) = weight(tri) + 1
    area(tri) = area(tri) + (v1x*v2y - v1y*v2x)
  end do
!
! Now we divide each element by 6
!
  do i = 1, n
    area(i) = area(i) / 6
  end do

  return
end
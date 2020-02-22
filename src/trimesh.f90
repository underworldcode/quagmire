! Copyright 2016-2017 Louis Moresi, Ben Mather, Romain Beucher
!
! This file is part of Quagmire.
!
! Quagmire is free software: you can redistribute it and/or modify
! it under the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or any later version.
!
! Quagmire is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with Quagmire.  If not, see <http://www.gnu.org/licenses/>.
!

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

subroutine add_pt ( pt, array, n )
!*****************************************************************************
! ADD_PT adds a point to an integer array if it does not already exist
! in this way it mimics a set in Python

  implicit none

  integer ( kind = 4 ) pt, n, i
  integer ( kind = 4 ) array(n)

  do i = 1, n
    if (array(i) .eq. 0) then
      exit
    else if (array(i) .eq. pt) then
      return
    end if
  end do

  array(i) = pt
  return
end subroutine

subroutine ncloud ( nt, ltri, n, nnz, ecloud )
!*****************************************************************************
! NCLOUD finds all neighbours and extended neighbours for every point
! in a triangulation

  implicit none

  integer ( kind = 4 ) nt, nnz, n
  integer ( kind = 4 ) ltri(3,nt)
  integer ( kind = 4 ) ecloud(n,nnz*nnz/2)
  integer ( kind = 4 ) cloud(n,nnz)
  integer ( kind = 4 ) tri(3)
  integer ( kind = 4 ) neighbours(nnz), eneighbours(nnz)
  integer ( kind = 4 ) i, t, ncol, np, enp, pt, ept

  ncol = nnz*nnz/2

  ecloud(:,:) = 0
  do t = 1, nt
    tri = ltri(:,t)

    call add_pt(tri(2), ecloud(tri(1),:), ncol)
    call add_pt(tri(3), ecloud(tri(1),:), ncol)
    call add_pt(tri(1), ecloud(tri(2),:), ncol)
    call add_pt(tri(3), ecloud(tri(2),:), ncol)
    call add_pt(tri(1), ecloud(tri(3),:), ncol)
    call add_pt(tri(2), ecloud(tri(3),:), ncol)

  end do

  cloud = ecloud(:,1:nnz)

  do i = 1, n
    neighbours = cloud(i,:)
    do np = 1, nnz
      ! Get neighbours
      pt = neighbours(np)
      if (pt .eq. 0) then
        exit
      endif
      eneighbours = cloud(pt,:)
      do enp = 1, nnz
        ! Get extended neighbours
        ept = eneighbours(enp)
        if (ept .eq. 0) then
          exit
        endif
        call add_pt(ept, ecloud(i,:), ncol)
      end do
    end do
  end do

  return
end subroutine

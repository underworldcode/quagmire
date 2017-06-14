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

subroutine remove_dups ( n, array_in, array_out, k )
!*****************************************************************************
! Remove duplicate entries from a list
! k on input is the size of the array
! k on output is the number of unique elements
  implicit none

  integer ( kind = 4 ) n
  integer ( kind = 4 ) array_in(n)         ! The input
  integer ( kind = 4 ) array_out(n)        ! The output
  integer ( kind = 4 ) k                   ! The number of unique elements  
  integer ( kind = 4 ) i, j
 
  k = 1
  array_out(:) = 0
  array_out(1) = array_in(1)
  outer: do i = 2, n
     do j = 1, k
        if (array_out(j) == array_in(i)) then
           ! Found a match so start looking again
           cycle outer
        end if
     end do
     ! No match found so add it to the output
     k = k + 1
     array_out(k) = array_in(i)
  end do outer
  return
end subroutine

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
!   Input, integer ( kind = 4 ), ltri(3,nt)
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

subroutine tricloud ( nt, ltri, n, ncol, cloud, kmax )
! Finds all neighbours and extended neighbours for every point
! in a triangulation
!
!   Input, integer ( kind = 4 ), nt
!   number of points in the triangulation
!
!   Input, integer ( kind = 4 ), ltri(3,nt)
!   list of triangles in the triangulation
!
!   Input, integer ( kind = 4 ), n
!   number of points in the triangulation
!
!   Input, integer ( kind = 4 ), ncol
!   estimated number of neighbours and extended neighbours
!
!   Output, integer ( kind = 4 ), cloud(n,ncol)
!   number of points in the triangulation
!
!   Output, integer ( kind = 4 ), kmax
!   max number of neighbours and extended neighbours
  implicit none

  integer ( kind = 4 ) nt, n, kmax, ncol
  integer ( kind = 4 ) ltri(3,nt)
  integer ( kind = 4 ), target :: cloud(n,ncol)
  integer ( kind = 4 ) rcloud(ncol*ncol), dedup(ncol*ncol)
  integer ( kind = 4 ) t1, t2, t3, s1, s2, s3
  integer ( kind = 4 ) rsize, ksize, i, j, k, p
  integer ( kind = 4 ) nnz(n)
  integer, pointer :: neighbours(:)
  integer, pointer :: eneighbours(:)

  nnz(:) = 1
  cloud(:,:) = 0

  do i = 1, nt
    t1 = ltri(1,i); s1 = nnz(t1)
    t2 = ltri(2,i); s2 = nnz(t2)
    t3 = ltri(3,i); s3 = nnz(t3)

    cloud(t1,s1) = t2; cloud(t1,s1+1) = t3
    cloud(t2,s2) = t1; cloud(t2,s2+1) = t3
    cloud(t3,s3) = t2; cloud(t3,s3+1) = t1

    nnz(t1) = nnz(t1) + 2
    nnz(t2) = nnz(t2) + 2
    nnz(t3) = nnz(t3) + 2
  end do

  kmax = 0

  do i = 1, n
    rcloud(:) = 0
    neighbours => cloud(i,:)
    rsize = nnz(i)
    rcloud(1:rsize) = neighbours(1:rsize)
    do j = 1, nnz(i)
      p = neighbours(j)
      eneighbours => cloud(p,:)
      ksize = nnz(p)
      rcloud(rsize:rsize+ksize) = eneighbours(1:ksize)
      rsize = rsize + ksize
    end do

    ! dedup
    dedup(:) = 0
    call remove_dups(rsize, rcloud(1:rsize), dedup(1:rsize), k)
    cloud(i,:) = pack(dedup(1:rsize), dedup(1:rsize) /= 0)
    cloud(i,k+1:ncol) = 0
    kmax = max(kmax, k)
  end do
  return
end subroutine

subroutine pixcloud ( nx, ny, cloud, kmax )
!*****************************************************************************
! PIXCLOUD finds all neighbours and extended neighbours for every point
  implicit none

  integer ( kind = 4 ) nx, ny, i, j, k
  integer ( kind = 4 ), target :: cloud(25,nx*ny)
  integer ( kind = 4 ) nodes(nx*ny)
  integer ( kind = 4 ), target :: indices(nx+2,ny+2)
  integer ( kind = 4 ) closure(2,9)
  integer ( kind = 4 ) n, rs, re, cs, ce, ind, kmax
  logical :: mask(nx*ny)
  integer ( kind = 4 ) dedup(25)
  integer, pointer :: diag(:)
  integer, pointer :: neighbours(:)
  integer, pointer :: icloud(:,:)

  n = nx*ny
  do i = 1, n
    nodes(i) = i
  end do
  cloud(:,:) = 0
  indices(:,:) = 0
  indices(2:nx-1,2:ny-1) = reshape(nodes, (/nx, ny/))

  ! hard-coded closure
  closure(1,1) = 1; closure(2,1) = -2
  closure(1,2) = 1; closure(2,2) = -2
  closure(1,3) = 3; closure(2,3) = 0
  closure(1,4) = 3; closure(2,4) = 0
  closure(1,5) = 1; closure(2,5) = -2
  closure(1,6) = 2; closure(2,6) = -1
  closure(1,7) = 3; closure(2,7) = 0
  closure(1,8) = 2; closure(2,8) = -1
  closure(1,9) = 1; closure(2,9) = -2

  ind = 1
  do i = 1, 8
    rs = closure(1, i+1)
    re = closure(2, i+1)
    cs = closure(1, i)
    ce = closure(2, i)

    icloud => indices(cs:nx+ce+2,rs:ny+re+2)
    cloud(ind,:) = pack(icloud, .true.)

    ind = ind + 1
  end do

  ! Centre node
  cloud(ind,:) = nodes
  ind = ind + 1

  do j = 1, 4
    diag => cloud(j,:)
    do i = 5, 8, 1
      rs = closure(1, i+1)
      re = closure(2, i+1)
      cs = closure(1, i)
      ce = closure(2, i)

      icloud => indices(cs:nx+ce+2,rs:ny+re+2)
      nodes = pack(icloud, .true.)
      mask = nodes == 0

      ! reassign  zero elements that will not slice an array
      where (mask)
        nodes = 1
      end where
      cloud(ind,:) = diag(nodes)
      where (mask)
        cloud(ind,:) = 0
      end where

      ind = ind + 1
    end do
  end do

  kmax = 0
  do i = 1, n
    neighbours => cloud(:,i)
    call remove_dups(25, neighbours, dedup, k)
    cloud(:,i) = pack(dedup, dedup /= 0)
    cloud(k+1:25,i) = 0
    kmax = max(kmax, k)
  end do

  return
end subroutine
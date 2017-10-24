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
end subroutine remove_dups

! module qsort_mod
! implicit none
 
! type group
!   integer :: order    ! original order of unsorted data
!   real :: value       ! values to be sorted by
! end type group
 
! contains
 
! recursive subroutine QSort(a,na)
 
! ! DUMMY ARGUMENTS
! integer, intent(in) :: nA
! type (group), dimension(nA), intent(in out) :: A
 
! ! LOCAL VARIABLES
! integer :: left, right
! real :: random
! real :: pivot
! type (group) :: temp
! integer :: marker
 
!     if (nA > 1) then
 
!         call random_number(random)
!         pivot = A(int(random*real(nA-1))+1)%value   ! random pivor (not best performance, but avoids worst-case)
!         left = 0
!         right = nA + 1
 
!         do while (left < right)
!             right = right - 1
!             do while (A(right)%value > pivot)
!                 right = right - 1
!             end do
!             left = left + 1
!             do while (A(left)%value < pivot)
!                 left = left + 1
!             end do
!             if (left < right) then
!                 temp = A(left)
!                 A(left) = A(right)
!                 A(right) = temp
!             end if
!         end do
 
!         if (left == right) then
!             marker = left + 1
!         else
!             marker = left
!         end if
 
!         call QSort(A(:marker-1),marker-1)
!         call QSort(A(marker:),nA-marker+1)
 
!     end if
 
! end subroutine QSort
 
! end module qsort_mod

recursive subroutine sort_quick (n, a, idx, first, last)
!*****************************************************************************
! Sorting using the quick sort algorithm.
! a is sorted in-place
! idx is the integers that would sort the array
! first and last specify where in the array is to be sorted
! 
! idx should be a range of integers from 1 to n
! 
! sort_quick.f -*-f90-*-
! Author: t-nissie
! License: GPLv3
! Gist: https://gist.githuidx.com/t-nissie/479f0f16966925fa29ea
  implicit none
  integer ( kind = 4 ) n
  integer ( kind = 4 ) a(n)
  integer ( kind = 4 ) idx(n)
  real ( kind = 8 ) x, t
  integer ( kind = 4 ) first, last
  integer ( kind = 4 ) i, j, ti

  x = a( (first+last) / 2 )
  i = first
  j = last
  do
    do while (a(i) < x)
      i=i+1
    end do
    do while (x < a(j))
      j=j-1
    end do
    if (i >= j) exit
    t = a(i);  a(i) = a(j);  a(j) = int(t, 4)
    ti= idx(i);  idx(i) = idx(j);  idx(j) = ti
    i=i+1
    j=j-1
  end do
  if (first < i-1) call sort_quick(n, a, idx, first, i-1)
  if (j+1 < last)  call sort_quick(n, a, idx, j+1, last)
end subroutine sort_quick

subroutine cumsum ( n, array, array_sum )
!*****************************************************************************
! cumsum evaluates the cumulative sum on an array
  implicit none
  integer ( kind = 4 ) n
  integer ( kind = 4 ) array(n)
  integer ( kind = 4 ) array_sum(n)
  integer ( kind = 4 ) i

  array_sum(:) = 0
  array_sum(1) = array(1)
  
  do i = 2, n
    array_sum(i) = array_sum(i-1) + array(i)
  end do
  return
end subroutine cumsum

! function cumsum ( n, arr )

!   implicit none
!   integer ( kind = 4 ) n
!   integer ( kind = 4 ) arr(n)
!   integer ( kind = 4 ) j, i
!   integer ( kind = 4 ), intent(out) :: cumsum(n)

!   do j = 2, n
!     cumsum(j) = cumsum(j-1) + arr(j)
!   end do
!   return
! end function cumsum

subroutine edgesort (n, edges)
!*****************************************************************************
! Sorts edges by rows with lowest integer first
! 
! if the lowest integer is in the second row, then the order is switched
! array is returned in-place
  implicit none
  integer ( kind = 4 ) n
  integer ( kind = 4 ) edges(2,n)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) a(n)
  logical ( kind = 4 ) mask(n)

  do i = 1, n
    a(i) = i
  end do

  mask = minloc(edges, dim=1).eq.2

  edges(:,pack(a, mask)) = edges(2:1:-1,pack(a, mask))
  return
end subroutine edgesort

subroutine bincount ( n, array, nnz )
!*****************************************************************************
! bincount evaluates the number of times an integer appears in an array
  implicit none
  integer ( kind = 4 ) n
  integer ( kind = 4 ) array(n)
  integer ( kind = 4 ) nnz(maxval(array))
  integer ( kind = 4 ) i

  nnz = 0
  do i = 1, n
    nnz(array(i)) = nnz(array(i)) + 1
  end do
  return
end subroutine bincount

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
  integer ( kind = 4 ), target :: ltri(3,nt)
  real ( kind = 8 ) v1x,v1y,v2x,v2y
  integer ( kind = 4 ), pointer :: tri(:)

!
! Get 2 sides of triangle
!
  do i = 1, nt
    tri => ltri(:,i)
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
end subroutine ntriw

subroutine remove_dups2D (nr, nc, arraya, arrayb)
!*****************************************************************************
! remove duplicate entries in a 2D array
! nc is modified in place to give number of unique columns
! https://stackoverflow.com/questions/14137610/...
! remove-repeated-elements-in-a-2d-array-in-fortran
  implicit none

  integer ( kind = 4 ), intent(in) :: nr  ! number of rows
  integer ( kind = 4 ), intent(inout) :: nc  ! number of columns
  integer ( kind = 4 ), intent(in) :: arraya(nr,nc)
  integer ( kind = 4 ), intent(out) :: arrayb(nr,nc)
  integer ( kind = 4 ) index_vector(nc)
  logical ( kind = 4 ) mask(nc)
  integer ( kind = 4 ) ix, k

  ! First, find the duplicate elements
  mask(:) = .true.
  arrayb(:,:) = 0

  do ix = nc, 2, -1
    mask(ix) = .not.(any(arraya(1,:ix-1)==arraya(1,ix).and.&
                         arraya(2,:ix-1)==arraya(2,ix)))
  end do

  k = count(mask)

  index_vector = pack([(ix, ix=1,nc)], mask)
  arrayb(:,1:k) = arraya(:,index_vector(1:k))

  nc = k
  return
end subroutine remove_dups2D

! module something
!   integer ( kind = 4 ), allocatable :: cloud(:)
! contains
subroutine tricloud2 ( nt, ltri, n, callable )
!*****************************************************************************
! Finds all neighbours and extended neighbours for every point
  implicit none

  integer ( kind = 4 ) nt, n
  integer ( kind = 4 ) ltri(3,nt)
  integer ( kind = 4 ) edges(2,3*nt)
  integer ( kind = 4 ) edgey(2,3*nt)
  integer ( kind = 4 ), allocatable :: row(:), col(:), sort(:)
  integer ( kind = 4 ), allocatable :: bnnz(:), bnnzsum(:), colsum(:)
  integer ( kind = 4 ) nr, nc, i, k
  integer ( kind = 4 ) nnz(n), nnzsum(n)
  ! integer ( kind = 4 ), intent(out), allocatable :: cloud(n,k)
  logical ( kind = 4 ), allocatable :: nmask(:)
  external callable
  integer ( kind = 4 ), allocatable :: cloud(:)
  integer ( kind = 4 ) nk
  
  ! nnz(maxval(ltri))

  nr = 2; nc = 3*nt
  write(*,*) "removed dups", maxval(ltri)

  ! slice edges from triangles
  edges(1,1:nt) = ltri(1,:)
  edges(2,1:nt) = ltri(2,:)
  edges(1,nt+1:2*nt) = ltri(1,:)
  edges(2,nt+1:2*nt) = ltri(3,:)
  edges(1,2*nt+1:3*nt) = ltri(2,:)
  edges(2,2*nt+1:3*nt) = ltri(3,:)

  write(*,*) "shapes", shape(edges(1,1:nt)), shape(edges(1,nt:2*nt))
  write(*,*) "removed dups", maxval(edges), maxval(edgey)

  ! sort edges
  call edgesort(nc, edges)

  ! remove duplicate entries
  call remove_dups2D(nr, nc, edges, edgey)

  write(*,*) "removed dups", maxval(edges), maxval(edgey)

  ! allocate arrays now we know the number of columns
  allocate(row(2*nc))
  allocate(col(2*nc))
  row(1:nc) = edgey(1,:nc); row(nc+1:2*nc) = edgey(2,:nc)
  col(1:nc) = edgey(2,:nc); col(nc+1:2*nc) = edgey(1,:nc)

  ! allocate(row, source=reshape(edgey(:,:nc), [ 2*nc ]))
  ! allocate(col, source=reshape(edgey(:,:nc), [ 2*nc ]))

  write(*,*) "This is the length of row and col", 2*nc, size(row), size(col)
  nc = nc*2 ! nc is now the length of row and col vectors

  write(*,*) "Shape of nc", nc

  ! allocate these for insertion to cloud
  allocate(sort(nc))
  allocate(bnnz(nc))
  allocate(bnnzsum(nc))
  allocate(colsum(nc))

  do i = 1, nc
    sort(i) = i
    colsum(i) = i
  end do

  k = nc

  ! sort these arrays
  call sort_quick(nc, row, sort, 1, k)
  col = col(sort)

  write(*,*) "sorted arrays", nc, size(nnz), maxval(row), maxval(col), maxval(edgey)

  ! get the number of nonzeros per row
  call bincount(nc, row, nnz)

  write(*,*) "bincount nnz", size(nnz), minval(nnz), maxval(nnz)

  ! get the cumulative sum of nnz
  call cumsum(nc, nnz, nnzsum)

  write(*,*) "cumsum nnz", size(nnzsum), minval(nnzsum), maxval(nnzsum)

  k = maxval(nnz) ! max number of columns in cloud

  write(*,*) "bnnz", size(bnnz), size(nnz), nnz(n-1), nnz(n)

  ! place this in a vector where the number of nonzeros are
  ! subtracted from the number of columns
  bnnz = 0
  bnnz(nnzsum(:nc-1)) = k - nnz(:n-1)
  call cumsum(nc, bnnz, bnnzsum)

  colsum = colsum + bnnzsum

  write(*,*) "colsum", maxval(colsum)



  ! allocate(nmask(k*n))
  ! nmask(:) = .true.
  ! allocate(cloud, source=reshape(pack(col, .true., colsum), (/n,k/)))
  allocate(cloud(k*n))
  cloud = 0
  write(*,*) "allocated cloud", k, n
  cloud(colsum) = col
  write(*,*) "shoving it in is done", n, k, maxval(cloud)
  nk = k*n
  call callable(nk,cloud)
  ! cloud = pack(col, .true., (/n,k/))
  write(*,*) "cloud", cloud(0)

  ! allocate(cloud(n,k))
  ! cloud(colsum) = col
  ! cloud = reshape(cloud, (/ n, k /))
  ! allocate(nmask(k*n))
  ! nmask(colsum) = .true.
  ! nmask = reshape(nmask, (/ n, k /))

  ! return
end subroutine tricloud2
! end module something

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
end subroutine tricloud

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
end subroutine pixcloud
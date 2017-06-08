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
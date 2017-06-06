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

subroutine remove_dups ( k, array_in, array_out )
!*****************************************************************************
! Remove duplicate entries from a list
! k on input is the size of the array
! k on output is the number of unique elements
  implicit none

  integer ( kind = 4 ) k                   ! The number of unique elements  
  integer ( kind = 4 ) array_in(k)         ! The input
  integer ( kind = 4 ) array_out(k)        ! The output
  integer ( kind = 4 ) i, j
 
  k = 1
  array_out(1) = array_in(1)
  outer: do i=2,size(array_in)
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
  write(*,advance='no',fmt='(a,i0,a)') 'Unique list has ',k,' elements: '
  write(*,*) array_out(1:k)
end subroutine

subroutine pixcloud ( nx, ny, cloud )
!*****************************************************************************
! PIXCLOUD finds all neighbours and extended neighbours for every point
  implicit none

  integer ( kind = 4 ) nx, ny, i
  integer ( kind = 4 ) cloud(25,nx*ny)
  integer ( kind = 4 ) nodes(nx*ny)
  integer ( kind = 4 ) indices(nx+2,ny+2)
  integer ( kind = 4 ) closure(2,9)
  integer ( kind = 4 ) n, rs, re, cs, ce, ind, maxC
  integer ( kind = 4 ) icloud(nx,ny)
  logical ( kind = 4 ) imask(nx,ny)
  logical ( kind = 4 ) m_block(4,nx*ny)
  integer ( kind = 4 ) n_block(4,nx*ny)

  n = nx*ny
  do i = 1, n
    nodes(i) = i
  end do
  cloud(:,:) = 0
  indices(:,:) = 0
  indices(2:nx-1,2:ny-1) = reshape(nodes, (/nx, ny/))

  ! hard-coded closure
  closure(1,1) = 1
  closure(2,1) = -2
  closure(1,2) = 1
  closure(2,2) = -2
  closure(1,3) = 3
  closure(2,3) = 0
  closure(1,4) = 3
  closure(2,4) = 0
  closure(1,5) = 1
  closure(2,5) = -2
  closure(1,6) = 2
  closure(2,6) = -1
  closure(1,7) = 3
  closure(2,7) = 0
  closure(1,8) = 2
  closure(2,8) = -1
  closure(1,9) = 1
  closure(2,9) = -2

  ind = 1
  do i = 1, 8
    rs = closure(1, i+1)
    re = closure(2, i+1)
    cs = closure(1, i)
    ce = closure(2, i)

    icloud = indices(cs:nx+ce+2,rs:ny+re+2)
    imask =  icloud > 0

    cloud(ind,1:count(imask)) = pack(icloud, mask=imask)

    ind = ind + 1
  end do

  ! Centre node
  cloud(ind,:) = nodes
  ind = ind + 1

  ! ! Block mask
  ! m_block = cloud(5:8,:) == 0

  ! do i = 1, 4
  !   nodes = cloud(i,:)
  !   n_block = cloud(5:8, nodes)
  !   write(*,*) n_block
  !   ! n_block = pack(n_block, mask=m_block)
  !   ! n_block(where m_block) = 0
  !   ! cloud(ind:ind+4,:) = n_block
  !   ind = ind + 4
  ! end do

  return
end subroutine
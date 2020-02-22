function uniform ( low, high, k )
  implicit none

  real ( kind = 8 ) low
  real ( kind = 8 ) high
  integer ( kind = 4 ) k
  real ( kind = 8 ) uniform(k)


function random_pt  ( pt, r, k )
  implicit none

  real ( kind = 8 ) pt(2)
  real ( kind = 8 ) r
  integer ( kind = 4 ) k
  real ( kind = 8 ) random_pt (2,k)
  real ( kind = 8 ) pi
  real ( kind = 8 ) rr(k),rt(k)
  integer ( kin = 4 ) i
  real ( kind = 8 ) rnum(k)

  pi = 4.D0*DATAN(1.D0)

  call random_number(rnum)
  rr = uniform( r, 2*r, k )
  rt = uniform( 0, 2*pi)

  random_pt()
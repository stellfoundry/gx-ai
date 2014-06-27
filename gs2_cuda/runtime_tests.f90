






!>  This module is intended to be used for runtime tests
!!  which interrogate what is functional/what compile time
!!  options were enabled/disabled. 
!!
module runtime_tests

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Tests for compilers
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  function compiler_pgi()
    logical :: compiler_pgi
    compiler_pgi = .false.
  end function compiler_pgi

  function get_compiler_name()
    character(len=9) :: get_compiler_name
    get_compiler_name='unknown'
    get_compiler_name='intel'
  end function get_compiler_name

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!  Tests for svn info
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !>This function returns the output of svnversion 
  !It would be nice if we could strip of trailing text
  !so that we're just left with the integer revision.
  function get_svn_rev()
    character(len=8) :: get_svn_rev
    get_svn_rev="exported"
  end function get_svn_rev

  !>This function returns true if the source code has
  !been modified relative to repo
  function get_svn_modified()
    logical :: get_svn_modified
    integer :: indx
    indx=index("exported","M")
    if(indx.eq.0)then
       get_svn_modified=.false.
    else
       get_svn_modified=.true.
    endif
  end function get_svn_modified
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
end module runtime_tests

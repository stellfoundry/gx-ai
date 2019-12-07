subroutine namelistRead(nmlfn, ncfn) bind(c, name='namelistRead')
    
  use netcdf
  use iso_c_binding, only: c_char, c_null_char
    
    character (kind=c_char, len=1), dimension(255), intent(in) :: nmlfn, ncfn
    character (len=255) :: nml_filename, nc_filename
    character (len=32)  :: note, note_name
    
    integer :: nwrite, nsave, navg, j, ncid
    integer :: retval, id_nwrite, id_nsave, id_navg, id_write_ascii

    logical :: print_flux_line
    logical :: write_ascii = .false.
    real, dimension(2) :: asdf = 2.
    
    namelist /gs2_diagnostics_knobs/ print_flux_line, write_ascii, &
         nwrite, nsave, navg
    
! Convert C-string to Fortran character type
    nml_filename = " "
    loop_string: do i=1,256
       if (nmlfn(i) == c_null_char) then
          exit loop_string
       else
          nml_filename(i:i) = nmlfn(i)
       end if
    end do loop_string
       
! Read the namelist     
    open(unit=100, file=nml_filename)
    read(unit=100, nml=gs2_diagnostics_knobs)
    close(unit=100)
    
!    write(*,*)'Fortran procedure has nwrite, nsave, navg:', nwrite, nsave, navg
    
! Convert C-string to Fortran character type
    nc_filename = " "
    loop_string2: do i=1,256
       if (ncfn(i) == c_null_char) then
          exit loop_string2
       else
          nc_filename(i:i) = ncfn(i)
       end if
    end do loop_string2
    
!    write(*,*)
!    write(*,*) 'NetCDF filename will be ',nc_filename
    
    retval = nf90_create (nc_filename, NF90_CLOBBER, ncid)
!    retval = nf90_open (nc_filename, NF90_WRITE, ncid)
    retval = nf90_def_var (ncid, "nwrite", NF90_INT, id_nwrite)
    note_name = "note"
    note = "eik.out"
    retval = nf90_put_att (ncid, id_nwrite, trim(note_name), trim(note))
    retval = nf90_def_var (ncid, "nsave",  NF90_INT, id_nsave)
    retval = nf90_def_var (ncid, "navg",   NF90_INT, id_navg)
    retval = nf90_def_var (ncid, "write_ascii", NF90_INT, id_write_ascii)
    retval = nf90_enddef (ncid)

!    retval = nf90_inq_varid(ncid, "write_ascii", id_write_ascii)
    j = write_ascii
    retval = nf90_put_var (ncid, id_write_ascii, j)

!    retval = nf90_inq_varid(ncid, "nwrite", id_nwrite)
    retval = nf90_put_var (ncid, id_nwrite, nwrite)

!    retval = nf90_inq_varid(ncid, "nsave", id_nsave) 
    retval = nf90_put_var (ncid, id_nsave, nsave)

!    retval = nf90_inq_varid(ncid, "navg", id_navg)
    retval = nf90_put_var (ncid, id_navg, navg)
    retval = nf90_close(ncid) 

!    write(*,*) 'retval = ',retval,' ',nf90_strerror(retval)
    
  end subroutine namelistRead


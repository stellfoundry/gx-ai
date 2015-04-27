
#define EXTERN_SWITCH extern

#include "cufft.h"
#include "simpledataio_cuda.h"
#include "everything_struct.h"
#include "allocations.h"


input_parameters_struct * inps_str(everything_struct * ev_d){
	input_parameters_struct * pars;
	cudaMemcpyToSymbol(&pars, &ev_d->pars, cudaMemcpyDeviceToHost);
  return pars;
}

void alloc_dealloc(void * pointer_to_array_pointer, int allocate_or_deallocate, int when_to_allocate, int memory_location, int type, int size){

	void ** void_ptrptr = (void **)pointer_to_array_pointer;
	if (allocate_or_deallocate == ALLOCATE){
		if ( memory_location == ON_HOST){
			if (when_to_allocate == ON_HOST_AND_DEVICE || when_to_allocate == ON_HOST){
				switch(type){
					case TYPE_INT:
						*void_ptrptr = (void *)malloc(sizeof(int)*size);
						break;
					case TYPE_FLOAT:
						*void_ptrptr = (void *)malloc(sizeof(float)*size);
						break;
					case TYPE_CUCOMPLEX:
						*void_ptrptr = (void *)malloc(sizeof(cuComplex)*size);
						break;
					case TYPE_CUCOMPLEX_PTR:
						*void_ptrptr = (void *)malloc(sizeof(cuComplex *)*size);
						break;
					default:
						printf("Unknown type %d in alloc_dealloc\n", type);
						abort();
				}
			}
			else
				//printf("Not allocating!\n");
				*void_ptrptr = NULL;
		}
		else if ( memory_location == ON_DEVICE){
			if (when_to_allocate == ON_HOST_AND_DEVICE || when_to_allocate == ON_DEVICE){
				switch(type){
					case TYPE_INT:
  					cudaMalloc(void_ptrptr, sizeof(int)*size);
						break;
					case TYPE_FLOAT:
  					cudaMalloc(void_ptrptr, sizeof(float)*size);
						break;
					case TYPE_CUCOMPLEX:
  					cudaMalloc(void_ptrptr, sizeof(cuComplex)*size);
						break;
					case TYPE_CUCOMPLEX_PTR:
  					cudaMalloc(void_ptrptr, sizeof(cuComplex *)*size);
						break;
					default:
						printf("Unknown type %d in alloc_dealloc\n", type);
						abort();
				}
			}
		}
		else {
			printf("Don't know whether the memory should be allocated on the host or device.\n");
			abort();
		}
	}
	else if (allocate_or_deallocate == DEALLOCATE){
		if ( memory_location == ON_HOST){
			if (when_to_allocate == ON_HOST_AND_DEVICE || when_to_allocate == ON_HOST){
				if (*void_ptrptr != NULL){
					free(*void_ptrptr);
				}
				else {
					printf("Trying to free an unallocated array: help!\n");
					abort();
				}
			}
		}
		else if ( memory_location == ON_DEVICE){
			if (when_to_allocate == ON_HOST_AND_DEVICE || when_to_allocate == ON_DEVICE){
				if (*void_ptrptr != NULL){
					cudaFree(*void_ptrptr);
				}
				else {
					printf("Trying to free an unallocated array: help!\n");
					abort();
				}
			}
		}
		else {
			printf("Don't know whether the memory should be deallocated on the host or device.\n");
			abort();
		}
	}
	else{
		printf("Don't know whether to allocate or deallocate in alloc_dealloc\n");
		abort();
	}

}


void allocate_field_species_array(cuComplex *** ptr, int aod, int ml, grids_struct * grids){
	int i;
	if (aod == ALLOCATE) alloc_dealloc((void *)ptr, aod, ON_HOST, ON_HOST, TYPE_CUCOMPLEX_PTR, grids->Nspecies); 
	/* Remember:
		  ptr is a pointer to a single cuComplex **
			*ptr is a pointer to an array of Nspecies (cuComplex *)s
		  (*ptr)[i] is a cuComplex *;
		  &(*ptr)[i] is the address of that cuComplex *;
		*/	
	for (i=0; i<grids->Nspecies; i++){
		alloc_dealloc(&(*ptr)[i], aod, ON_DEVICE, ml, TYPE_CUCOMPLEX, grids->NxNycNz); 
	}
	if (aod == DEALLOCATE) alloc_dealloc((void *)ptr, aod, ON_HOST, ON_HOST, TYPE_CUCOMPLEX_PTR, grids->Nspecies); 
	//printf("allocate_field_species_array\n");
}

void allocate_fields(int aod, int ml, grids_struct * grids, fields_struct * fields){
	/* Host and device */
	alloc_dealloc(&fields->phi, aod, ON_HOST_AND_DEVICE, ml,  TYPE_CUCOMPLEX, grids->NxNycNz);
	/* A temporary array */
	alloc_dealloc(&fields->field, aod, ON_HOST_AND_DEVICE, ml,  TYPE_CUCOMPLEX, grids->NxNycNz);
	/* Device only*/
	alloc_dealloc(&fields->phi1, aod, ON_DEVICE, ml,  TYPE_CUCOMPLEX, grids->NxNycNz);


//	allocate_field_species_array(&fields->dens, aod, ml, grids);
//	allocate_field_species_array(&fields->dens1, aod, ml, grids);
//	allocate_field_species_array(&fields->upar, aod, ml, grids);
//	allocate_field_species_array(&fields->upar1, aod, ml, grids);
//	allocate_field_species_array(&fields->tpar, aod, ml, grids);
//	allocate_field_species_array(&fields->tpar1, aod, ml, grids);
//	allocate_field_species_array(&fields->tprp, aod, ml, grids);
//	allocate_field_species_array(&fields->tprp1, aod, ml, grids);
//	allocate_field_species_array(&fields->qpar, aod, ml, grids);
//	allocate_field_species_array(&fields->qpar1, aod, ml, grids);
//	allocate_field_species_array(&fields->qprp, aod, ml, grids);
//	allocate_field_species_array(&fields->qprp1, aod, ml, grids);


	/*Globals...to be deleted eventually*/
	if (ml == ON_HOST){
		phi_h = fields->phi;
	}
	else if (ml == ON_DEVICE){
	}
}

void allocate_outputs(int aod, int ml, grids_struct * grids, outputs_struct * outs){

	alloc_dealloc(&outs->phi2_by_ky, aod, ON_HOST, ml, TYPE_FLOAT, grids->Ny_complex);
	alloc_dealloc(&outs->phi2_by_kx, aod, ON_HOST, ml, TYPE_FLOAT, grids->Ny);

	alloc_dealloc(&outs->omega, aod, ON_HOST_AND_DEVICE, ml,  TYPE_CUCOMPLEX, grids->Nx*grids->Ny_complex);

	alloc_dealloc(&outs->phi2_by_mode_movav, aod, ON_DEVICE, ml,  TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&outs->phi2_zonal_by_kx_movav, aod, ON_DEVICE, ml,  TYPE_FLOAT, grids->Nx);
	
	alloc_dealloc(&outs->hflux_by_mode_movav, aod, ON_DEVICE, ml,  TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&outs->hflux_by_species, aod, ON_HOST, ml,  TYPE_FLOAT, grids->Nspecies);
	alloc_dealloc(&outs->hflux_by_species_movav, aod, ON_HOST, ml,  TYPE_FLOAT, grids->Nspecies);

	alloc_dealloc(&outs->par_corr_by_ky_by_deltaz_movav, aod, ON_DEVICE, ml,  TYPE_FLOAT, grids->Nz*grids->Ny_complex);

	/*Globals...to be deleted eventually*/
	if (ml == ON_HOST){
  	omega_h = outs->omega;
	}
	else if (ml == ON_DEVICE){
	}
}

void allocate_temporary_arrays(int aod, int ml, grids_struct * grids, temporary_arrays_struct * tmp){

	alloc_dealloc(&tmp->CXYZ, aod, ON_DEVICE, ml, TYPE_CUCOMPLEX, grids->Nx*grids->Ny_complex*grids->Nz);
	alloc_dealloc(&tmp->X, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Nx);
	alloc_dealloc(&tmp->X2, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nx);
	alloc_dealloc(&tmp->Y, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Ny_complex);
	alloc_dealloc(&tmp->Y2, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Ny_complex);
	alloc_dealloc(&tmp->Z, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nz);
	alloc_dealloc(&tmp->XY, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&tmp->XY2, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&tmp->XY3, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&tmp->XY4, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Ny_complex);
	alloc_dealloc(&tmp->XY_R, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Ny);
	alloc_dealloc(&tmp->XZ, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nx*grids->Nz);
	alloc_dealloc(&tmp->YZ, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Ny_complex*grids->Nz);
}

void allocate_grids(int aod, int ml, grids_struct * grids){

	alloc_dealloc(&grids->kx, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Nx);
	alloc_dealloc(&grids->ky, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, grids->Ny_complex);
	alloc_dealloc(&grids->kz, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Nz);
	alloc_dealloc(&grids->kx_shift, aod, ON_DEVICE, ml, TYPE_FLOAT, grids->Ny_complex);
	alloc_dealloc(&grids->jump, aod, ON_DEVICE, ml, TYPE_INT, grids->Ny_complex);
	/*Globals...to be deleted eventually*/
	if (ml == ON_HOST){
		kx_h = grids->kx;
		ky_h =  grids->ky;
	}
	else if (ml == ON_DEVICE){
		kx = grids->kx;
		ky = grids->ky;
		kz = grids->kz;
	}

}

/* This one gets called separately for allocating, but not for deallocating */
void allocate_geo(int aod, int ml, geometry_coefficents_struct * geo, float ** z_array, int *Nz){
	alloc_dealloc(&geo->gradpar_arr, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->gbdrift, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->grho, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&*z_array, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	//*z = z_h  = (float*) malloc(sizeof(float)**Nz);
	alloc_dealloc(&geo->cvdrift, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->gds2, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->bmag, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->bgrad, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->gds21, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->gds22, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->cvdrift0, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->gbdrift0, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->jacobian, aod, ON_HOST_AND_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->Rplot, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->Zplot, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->aplot, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->Xplot, aod, ON_HOST, ml, TYPE_CUCOMPLEX, *Nz);
	alloc_dealloc(&geo->Yplot, aod, ON_HOST, ml, TYPE_CUCOMPLEX, *Nz);
	alloc_dealloc(&geo->Rprime, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->Zprime, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->aprime, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->deltaFL, aod, ON_HOST, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->bmagInv, aod, ON_DEVICE, ml, TYPE_FLOAT, *Nz);
	alloc_dealloc(&geo->bmag_complex, aod, ON_DEVICE, ml, TYPE_CUCOMPLEX, *Nz);

	/* Geometry globals... to be deleted eventually*/
	if (ml == ON_HOST){
    gradpar_arr_h = geo->gradpar_arr;
		gbdrift_h = geo->gbdrift;
		grho_h = geo->grho;
		z_h = *z_array;
		cvdrift_h = geo->cvdrift;
		gds2_h = geo->gds2;
		bmag_h = geo->bmag;
		bgrad_h = geo->bgrad;
		gds21_h = geo->gds21;
		gds22_h = geo->gds22;
		cvdrift0_h = geo->cvdrift0;
		gbdrift0_h = geo->gbdrift0;
		jacobian_h = geo->jacobian;
		Rplot_h = geo->Rplot;
		Zplot_h = geo->Zplot;
		aplot_h = geo->aplot;
		Xplot_h = geo->Xplot;
		Yplot_h = geo->Yplot;
		Rprime_h = geo->Rprime;
		Zprime_h = geo->Zprime;
		aprime_h = geo->aprime;
    deltaFL_h = geo->deltaFL;
	}
	else if (ml == ON_DEVICE){
		gbdrift = geo->gbdrift;
		grho = geo->grho;
		z = *z_array;
		cvdrift = geo->cvdrift;
		gds2 = geo->gds2;
		bmag = geo->bmag;
		bgrad = geo->bgrad;
		gds21 = geo->gds21;
		gds22 = geo->gds22;
		cvdrift0 = geo->cvdrift0;
		gbdrift0 = geo->gbdrift0;
	}
}

void allocate_info(int aod, int ml, info_struct * info, int run_name_size, int restart_name_size){

	alloc_dealloc(&info->run_name, aod, ON_HOST, ml, TYPE_FLOAT, run_name_size);
	alloc_dealloc(&info->restart_file_name, aod, ON_HOST, ml, TYPE_FLOAT, restart_name_size);
	/*Globals...to be deleted eventually*/
	if (ml == ON_HOST){
	}
	else if (ml == ON_DEVICE){
	}

}

void allocate_or_deallocate_everything(int allocate_or_deallocate, everything_struct * ev){
	allocate_grids(allocate_or_deallocate, ev->memory_location, &ev->grids);
	allocate_outputs(allocate_or_deallocate, ev->memory_location, &ev->grids, &ev->outs);
	allocate_fields(allocate_or_deallocate, ev->memory_location, &ev->grids, &ev->fields);
	allocate_temporary_arrays(allocate_or_deallocate, ev->memory_location, &ev->grids, &ev->tmp);
	if (allocate_or_deallocate == DEALLOCATE){
		allocate_geo(allocate_or_deallocate, ev->memory_location, &ev->geo, &ev->grids.z, &ev->grids.Nz);
    //Info needs to be deallocated on not proc0
    allocate_info(allocate_or_deallocate, ev->memory_location, &ev->info, -1, -1);
    //Need to deallocate strings in input pars.
	}
}


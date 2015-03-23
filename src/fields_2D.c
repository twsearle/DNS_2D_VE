/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_2D_c.c							      *
 *                                                                            *
 *  functions for 2D fields in C					      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Mon 23 Mar 12:52:32 2015

#include"fields_2D.h"

// Functions

void save_state(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /*
     * This function takes a pointer to the first element of a 2D array and
     * uses values from it to fill a a file.  Read these files using the scipy
     * command: genfromtxt(filename, dtype='complex128')
     */

    int N = cnsts.N;
    int M = cnsts.M;
    int rows = (2*N+1);
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fprintf(fp, " (%e18+%e18j) ", creal(arr[ind(i, j)]),
					  cimag(arr[ind(i, j)]));
	}
	fprintf(fp, "\n");

    }
    
}

void save_hdf5_arr(char *filename, fftw_complex *arr, int size)
{
    int i;

    hid_t  file_id, dataset_id, dataspace_id, datatype_id, filetype_id;
    herr_t status;
    complex_hdf *wdata;

    hsize_t arr_size[1];
    arr_size[0] = size;

    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // create the filetype for the scipy complex numbers
    filetype_id = H5Tcreate(H5T_COMPOUND, 8 + 8);
    status = H5Tinsert(filetype_id, "r", 0, H5T_NATIVE_DOUBLE);
    status = H5Tinsert(filetype_id, "i", 8, H5T_NATIVE_DOUBLE);

    // create the file
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the dataset
    dataset_id = H5Dcreate(file_id, "/psi", filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    wdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    for (i=0; i<size; i++)
    {
	wdata[i].r = creal(arr[i]);
	wdata[i].i = cimag(arr[i]);
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(datatype_id);
    status = H5Tclose(filetype_id);
    free(wdata);

    // close the file
    status = H5Fclose(file_id);

}
void save_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;

    hid_t  file_id, dataset_id, dataspace_id, datatype_id, filetype_id;
    herr_t status;
    complex_hdf *wdata;

    hsize_t arr_size[1];
    arr_size[0] = (2*N+1)*M;

    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // create the filetype for the scipy complex numbers
    filetype_id = H5Tcreate(H5T_COMPOUND, 8 + 8);
    status = H5Tinsert(filetype_id, "r", 0, H5T_NATIVE_DOUBLE);
    status = H5Tinsert(filetype_id, "i", 8, H5T_NATIVE_DOUBLE);

    // create the file
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the dataset
    dataset_id = H5Dcreate(file_id, "/psi", filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    wdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    for (j=0; j<M; j++)
    {
	wdata[j].r = creal(arr[j]);
	wdata[j].i = cimag(arr[j]);
    }

    for (i=1; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(arr[M*i + j]);
	    wdata[M*i + j].i = cimag(arr[M*i + j]);

	    wdata[M*(2*N+1-i) + j].r = creal(arr[M*i + j]);
	    wdata[M*(2*N+1-i) + j].i = -cimag(arr[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(datatype_id);
    status = H5Tclose(filetype_id);
    free(wdata);

    // close the file
    status = H5Fclose(file_id);

}

void save_hdf5_snapshot(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	fftw_complex *arr, double time, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;
    char dataset_name[30];

    hid_t  dataset_id, dataspace_id;
    herr_t status;
    complex_hdf *wdata;

    hsize_t arr_size[1];
    arr_size[0] = (N+1)*M;

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the dataset
    // TODO: look into how to use nested datasets.
    sprintf(dataset_name, "/t%f", time);

    dataset_id = H5Dcreate(*file_id, dataset_name, *filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    wdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(arr[M*i + j]);
	    wdata[M*i + j].i = cimag(arr[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, *datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    free(wdata);

}

void save_hdf5_snapshot_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex *psi, complex *cxx, complex *cyy, complex *cxy, double time, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;
    char group_name[30];

    hid_t  dataset_id, dataspace_id, group_id;
    herr_t status;
    complex_hdf *wdata;

    hsize_t arr_size[1];
    arr_size[0] = (N+1)*M;

    wdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the group 
    sprintf(group_name, "/t%f", time);

    group_id = H5Gcreate(*file_id, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // create the dataset
    dataset_id = H5Dcreate(group_id, "psi", *filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(psi[M*i + j]);
	    wdata[M*i + j].i = cimag(psi[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, *datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 
    status = H5Dclose(dataset_id);

    // create the dataset
    dataset_id = H5Dcreate(group_id, "cxx", *filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(cxx[M*i + j]);
	    wdata[M*i + j].i = cimag(cxx[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, *datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 
    status = H5Dclose(dataset_id);

    // create the dataset
    dataset_id = H5Dcreate(group_id, "cyy", *filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(cyy[M*i + j]);
	    wdata[M*i + j].i = cimag(cyy[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, *datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 
    status = H5Dclose(dataset_id);

    // create the dataset
    dataset_id = H5Dcreate(group_id, "cxy", *filetype_id, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // set up the data for writing
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++)
	{
	    wdata[M*i + j].r = creal(cxy[M*i + j]);
	    wdata[M*i + j].i = cimag(cxy[M*i + j]);
	}
    }
    
    // write the file 
    status = H5Dwrite(dataset_id, *datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, wdata); 
    status = H5Dclose(dataset_id);

    // clean up 
    status = H5Sclose(dataspace_id);
    status = H5Gclose(group_id);
    free(wdata);

}

void load_hdf5_state(char *filename, fftw_complex *arr, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j, ndims;

    hid_t  file_id, dataset_id, dataspace_id, datatype_id;
    herr_t status;
    complex_hdf *rdata;

    hsize_t arr_size[1];
    arr_size[0] = (2*N+1)*M;

    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/psi", H5P_DEFAULT);

    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);

    rdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));
    
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 

    // copy the result into the complex array
    for (i=0; i<N+1; i++)
    {
	for (j=0; j<M; j++) 
	{
	    arr[ind(i,j)] = rdata[ind(i,j)].r + I * rdata[ind(i,j)].i;
	}
    }

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(datatype_id);
    free(rdata);

    // close the file
    status = H5Fclose(file_id);

}

void load_hdf5_state_visco(char *filename, complex *psi, complex *cxx, complex *cyy, complex *cxy, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j, ndims;

    hid_t  file_id, dataset_id, dataspace_id, datatype_id;
    herr_t status;
    complex_hdf *rdata;

    hsize_t arr_size[1];
    arr_size[0] = (2*N+1)*M;
    rdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/psi", H5P_DEFAULT);
    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 
    // copy the result into the complex array
    for (i=0; i<(N+1); i++)
    {
	for (j=0; j<M; j++) 
	{
	    psi[ind(i,j)] = rdata[ind(i,j)].r + I * rdata[ind(i,j)].i;
	}
    }
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/cxx", H5P_DEFAULT);
    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 
    // copy the result into the complex array
    for (i=0; i<(N+1); i++)
    {
	for (j=0; j<M; j++) 
	{
	    cxx[ind(i,j)] = rdata[ind(i,j)].r + I * rdata[ind(i,j)].i;
	}
    }
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/cyy", H5P_DEFAULT);
    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 
    // copy the result into the complex array
    for (i=0; i<(N+1); i++)
    {
	for (j=0; j<M; j++) 
	{
	    cyy[ind(i,j)] = rdata[ind(i,j)].r + I * rdata[ind(i,j)].i;
	}
    }
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/cxy", H5P_DEFAULT);
    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 
    // copy the result into the complex array
    for (i=0; i<(N+1); i++)
    {
	for (j=0; j<M; j++) 
	{
	    cxy[ind(i,j)] = rdata[ind(i,j)].r + I * rdata[ind(i,j)].i;
	}
    }
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);


    status = H5Tclose(datatype_id);
    free(rdata);

    // close the file
    status = H5Fclose(file_id);

}

void load_state(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /* 
     * This function loads a 2D array of complex doubles in scipy format to a
     * 2D array.
     *
     */

    float tmpRe, tmpIm;

    int N = cnsts.N;
    int M = cnsts.M;
    int rows = (2*N+1);
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fscanf(fp, " (%e+%ej) ", &tmpRe, &tmpIm);
	    arr[ind(i, j)] = (double)tmpRe + (double)tmpIm * I;
	    //if(j==1) { printf(" (%e+%ej) ", creal(arr[i][j]), cimag(arr[i][j])); }
	}
	fscanf(fp, "\n");

    }
    
}

void load_operator(FILE *fp, fftw_complex *arr,  flow_params cnsts)
{
    /* 
     * This function loads a 2D array of complex doubles in scipy format to a
     * 2D array of size M by M.
     *
     */

    float tmpRe, tmpIm;

    int M = cnsts.M;
    int rows = M;
    int cols = M;
    int i, j;

    for(i=0; i<rows; i++)
    {

	for(j=0; j<cols; j++)
	{
	    fscanf(fp, " (%e+%ej) ", &tmpRe, &tmpIm);
	    arr[M*i + j] = (double)tmpRe + (double)tmpIm * I;
	    //if(j==1) { printf(" (%e+%ej) ", creal(arr[i][j]), cimag(arr[i][j])); }
	}
	fscanf(fp, "\n");

    }
    
}

void load_hdf5_operator(char *filename, fftw_complex *arr, flow_params cnsts)
{
    int M = cnsts.M;
    int i, j, ndims;

    hid_t  file_id, dataset_id, dataspace_id, datatype_id;
    herr_t status;
    complex_hdf *rdata;

    hsize_t arr_size[1];
    arr_size[0] = M*M;

    // create the datatype for scipy complex numbers
    datatype_id = H5Tcreate(H5T_COMPOUND, sizeof (complex_hdf));
    status = H5Tinsert(datatype_id, "r",
                HOFFSET(complex_hdf, r), H5T_NATIVE_DOUBLE);
    status = H5Tinsert(datatype_id, "i",
                HOFFSET(complex_hdf, i), H5T_NATIVE_DOUBLE);

    // open the file
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);

    // open the dataset
    dataset_id = H5Dopen2(file_id, "/op", H5P_DEFAULT);

    // get dataspace and allocate memory to read buffer
    dataspace_id = H5Dget_space(dataset_id);
    ndims = H5Sget_simple_extent_dims(dataspace_id, arr_size, NULL);

    rdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));
    
    // read the file 
    status = H5Dread(dataset_id, datatype_id, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, rdata); 

    // copy the result into the complex array
    for (i=0; i<M; i++)
    {
	for (j=0; j<M; j++) 
	{
	    arr[i*M + j] = rdata[i*M + j].r + I * rdata[i*M + j].i;
	}
    }

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);
    status = H5Tclose(datatype_id);
    free(rdata);

    // close the file
    status = H5Fclose(file_id);


}

void dx(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    double kx = cnsts.kx;
    int i, j;


    // For positive fourier modes
    for(i=0; i<N+1; i++)
    {
	//printf("%d\n",i);
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = i*kx*I*arrin[ind(i, j)];
	}

    }
}

void d2x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    double kx = cnsts.kx;
    int i, j;


    // For positive fourier modes
    for(i=0; i<N+1; i++)
    {
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = -pow(i*kx, 2)*arrin[ind(i, j)];
	}

    }
}

void d4x(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    double kx = cnsts.kx;
    int i, j;


    // For positive fourier modes
    for(i=0; i<N+1; i++)
    {
	//For all Chebyshev modes
	for(j=0; j<M; j++)
	{
	    arrout[ind(i, j)] = pow(i*kx, 4)*arrin[ind(i, j)];
	}

    }
}

void dy(fftw_complex *arrin, fftw_complex *arrout,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;

    // For all Fourier modes
    for(i=0; i<N+1; i++)
    {
	// Last 2 modes
	arrout[ind(i, M-1)] = 0;
	arrout[ind(i, M-2)] = 2.0*(M-1.0)*arrin[ind(i, M-1)];
	
	// For rest of the Chebyshev modes
	for(j=M-3; j>0; j=j-1)
	{
	    arrout[ind(i, j)] = 2.0*(j+1.0)*arrin[ind(i, j+1)] + arrout[ind(i, j+2)];
	}

	// Zeroth mode
	arrout[ind(i, 0)] = arrin[ind(i, 1)] + 0.5*arrout[ind(i, 2)];
    }
    
}

void matdy(fftw_complex *matarr, flow_params cnsts)
{
    int i, j;
    int M = cnsts.M;

    for (i=0; i<M*M; i++)
    {
	matarr[i] = 0;
    }


    for (j=1; j<M; j=j+2)
    {
	matarr[j] = j;
    }

    for (i=1; i<M; i++)
    {
	for (j=i+1; j<M; j=j+2)
	{
	    matarr[M*i + j] = 2*j;
	}
    }

}

void to_physical_1(fftw_complex *arrin, fftw_complex *arrout,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *phys_fou_plan, fftw_plan *phys_cheb_plan,
		 flow_params cnsts)
{
    int i,j;
    int M = cnsts.M;
    int Mf = cnsts.Mf;
    int N = cnsts.N;
    int Nf = cnsts.Nf;

    // Do the Fourier Transfrom

    // for each Chebyshev mode, fourier transform
    for (j=0; j<M; j++)
    {

        // copy the fourier mode into scratch
        if (cnsts.dealiasing)
        {
            scratchFin[0] = arrin[ind(0,j)];
            for (i=1; i<N+1; i++)
            {
        	scratchFin[i] = arrin[ind(i,j)];
        	scratchFin[2*Nf+1-i] = arrin[ind(2*N+1-i,j)];
            }
            // zero off the rest of the modes
            for(i=N+1; i<2*Nf+1-N; i++)
            {
        	scratchFin[i] = 0;
            }

        } else
        {
            for (i=0; i<2*N+1; i++)
            {
        	scratchFin[i] = arrin[ind(i,j)];
            }
        }

        fftw_execute(*phys_fou_plan);

        // copy the transformed array into the output for now
        for (i=0; i<2*Nf+1; i++)
        {
            arrout[indfft(i,j)] = scratchFout[i];
        }
    
    }

    // For each x column, do the Chebyshev transform

    for(i=0; i<2*Nf+1; i++)
    {

        for (j=0; j<M; j++)
        {
            scratchCin[j] = 0.5*arrout[indfft(i,j)];
        }

        if (cnsts.dealiasing)
        {
            // zero off the rest of the Chebyshev modes
            for(j=M; j<Mf; j++)
            {
        	scratchCin[j] = 0;
            }
        }


        //out2D[M:, :] = out2D[M-2:0:-1, :]

        for (j=2; j<Mf; j++)
        {
            scratchCin[Mf-2+j] = scratchCin[Mf-j];
        }

        //out2D[0, :] = 2*out2D[0, :]
        scratchCin[0] = 2*scratchCin[0];

        //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
        scratchCin[Mf-1] = 2*scratchCin[Mf-1];


        //perform the 2D fft.
        //out2D = fftpack.fft2(out2D)

        fftw_execute(*phys_cheb_plan);

        for (j=0; j<Mf; j++)
        {
            arrout[indfft(i,j)] = scratchCout[j];
        }
    }
}

void to_physical_r(complex *arrin, double *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *phys_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    // This uses an inverse FT. sign is +1.
    // factor of 2*N+1 is just here because my other method I am comparing with
    // has transforms in the inverse direction which pick up a normalisation
    // factor. 

    if (cnsts.dealiasing)
    {
        for (j=0; j<M; j++)
        {
            scratchin[indfft(0,j)] = arrin[ind(0,j)];
        }

        for (i=1; i<N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scratchin[indfft(i,j)] = arrin[ind(i,j)];
        	scratchin[indfft(2*Nf+1-i, j)] =  conj(arrin[ind(i,j)]);
           }
        }

        // zero off the rest of the fourier modes
        for(i=N+1; i<2*Nf+1-N; i++)
        {
            for(j=0; j<Mf; j++)
            {
        	scratchin[indfft(i,j)] = 0;
            }
        }

        // zero off the rest of the Chebyshev modes
        for(i=0; i<2*Nf+1; i++)
        {
            for(j=M; j<Mf; j++)
            {
        	scratchin[indfft(i,j)] = 0;
            }
        }
    }
    else
    {
        for (j=0; j<M; j++)
        {
            scratchin[indfft(0,j)] = arrin[ind(0,j)];
        }

        for (i=1; i<N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scratchin[indfft(i,j)] = arrin[ind(i,j)];
        	scratchin[indfft(2*Nf+1-i, j)] =  conj(arrin[ind(i,j)]);
           }
        }
    }


    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=2; j<Mf; j++)
        {
            scratchin[indfft(i, Mf-2+j)] = scratchin[indfft(i, Mf-j)];
        }

        //out2D[0, :] = 2*out2D[0, :]
        scratchin[indfft(i, 0)] = 2*scratchin[indfft(i, 0)];

        //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
        scratchin[indfft(i, Mf-1)] = 2*scratchin[indfft(i, Mf-1)];
    }

    ////perform the 2D ifft?.
    ////out2D = 0.5*fftpack.ifft2(out2D)

    fftw_execute(*phys_plan);

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=0; j<Mf; j++)
        {
            arrout[indfft(i,j)] = 0.5*creal(scratchout[indfft(i,j)]);
        }
    }
}

void to_physical(fftw_complex *arrin, fftw_complex *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *phys_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    // This uses an inverse FT. sign is +1.
    // factor of 2*N+1 is just here because my other method I am comparing with
    // has transforms in the inverse direction which pick up a normalisation
    // factor. 

    if (cnsts.dealiasing)
    {
        for (j=0; j<M; j++)
        {
            scratchin[indfft(0,j)] = arrin[ind(0,j)];
        }

        for (i=1; i<N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scratchin[indfft(i,j)] = arrin[ind(i,j)];
        	scratchin[indfft(2*Nf+1-i, j)] = arrin[ind(2*N+1-i,j)];
            }
        }

        // zero off the rest of the fourier modes
        for(i=N+1; i<2*Nf+1-N; i++)
        {
            for(j=0; j<Mf; j++)
            {
        	scratchin[indfft(i,j)] = 0;
            }
        }

        // zero off the rest of the Chebyshev modes
        for(i=0; i<2*Nf+1; i++)
        {
            for(j=M; j<Mf; j++)
            {
        	scratchin[indfft(i,j)] = 0;
            }
        }
    }
    else
    {
        for (i=0; i<2*N+1; i++)
        {
            for (j=0; j<M; j++)
            {
        	scratchin[indfft(i,j)] = arrin[ind(i,j)];
            }
        }
    }


    //out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=2; j<Mf; j++)
        {
            scratchin[indfft(i, Mf-2+j)] = scratchin[indfft(i, Mf-j)];
        }

        //out2D[0, :] = 2*out2D[0, :]
        scratchin[indfft(i, 0)] = 2*scratchin[indfft(i, 0)];

        //out2D[Mf-1, :] = 2*out2D[Mf-1, :]
        scratchin[indfft(i, Mf-1)] = 2*scratchin[indfft(i, Mf-1)];
    }

    ////perform the 2D ifft?.
    ////out2D = 0.5*fftpack.ifft2(out2D)

    fftw_execute(*phys_plan);

    for (i=0; i<2*Nf+1; i++)
    {
        for (j=0; j<Mf; j++)
        {
            arrout[indfft(i,j)] = 0.5*creal(scratchout[indfft(i,j)]);
        }
    }
}

void to_spectral_1(fftw_complex *arrin, fftw_complex *arrout, fftw_complex *scratch,
		 fftw_complex *scratchFin, fftw_complex *scratchFout,
		 fftw_complex *scratchCin, fftw_complex *scratchCout,
		 fftw_plan *spec_fou_plan, fftw_plan *spec_cheb_plan,
		 flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    double normalise = 2*Nf+1;
    int i,j;

    // zero out the interim array

    for (i=0; j<2*Nf+1; i++)
    {
	for (j=0; j<2*Mf-2; j++)
	{
	    scratch[indfft(i,j)] = 0;
	}
    }

    // Perform the Chebyshev transforms
    
    for (i=0; i<2*Nf+1; i++)
    {

	for (j=0; j<Mf; j++)
	{
	    scratchCin[j] = arrin[indfft(i,j)] / normalise;
	    //scratch[indfft(i,j)] = arrin[indfft(i,j)] ;// normalise;
	}

	// The second half contains the vector on the Gauss-Labatto points excluding
	// the first and last elements and in reverse order
	// out2D[M:, :] = out2D[M-2:0:-1, :]

	for (j=2; j<Mf; j++)
	{
	    scratchCin[Mf-2+j] = scratchCin[Mf-j];
	    //scratch[indfft(i,Mf-2+j)] = scratch[indfft(i, Mf-j)];
	}

	// Perform the transformation on this temporary vector
	// out2D = fftpack.fft2(out2D)
	fftw_execute(*spec_cheb_plan);

	if (cnsts.dealiasing)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    scratch[indfft(i,0)] = (0.5/(Mf-1.0)) * scratchCout[0];

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		scratch[indfft(i,j)] = (1.0/(Mf-1.0))*scratchCout[j];
	    }

	}

	else
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    scratch[indfft(i,0)] = (0.5/(M-1.0)) * scratchCout[0]; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		scratch[indfft(i,j)] = (1.0/(M-1.0))*scratchCout[j];
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    scratch[indfft(i,M-1)] = (0.5/(M-1.0)) * scratchCout[M-1]; 
	}
    }

    // Perform the Fourier transform

    for (j=0; j<M; j++)
    {
        // copy the Chebyshev mode from the interim array
        for (i=0; i<2*Nf+1;  i++)
        {
            scratchFin[i] = scratch[indfft(i,j)];
        }

        fftw_execute(*spec_fou_plan);

        // copy the Chebyshev mode into the output

        if (cnsts.dealiasing)
        {
            arrout[ind(0,j)] = scratchFout[0];
            for (i=1; i<N+1; i++)
            {
        	arrout[ind(i,j)] = scratchFout[i];
        	arrout[ind(2*N+1-i,j)] = scratchFout[2*Nf+1-i];
            }

        } else
        {
            for (i=0; i<2*N+1; i++)
            {
        	arrout[ind(i,j)] = scratchFout[i];
            }
        }
    }

    // // skip the Fourier transform, just output the chebyshev transform
    // for (j=0; j<M; j++)
    // {
    //         for (i=0; i<2*N+1; i++)
    //         {
    //     	arrout[ind(i,j)] = scratch[indfft(i,j)];
    //         }
    // }
}

void to_spectral_r(double *arrin, complex *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *spec_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    fftw_complex normalise = 2*Nf+1;

    // Perform the FFT across the x direction   

    // The first half contains the vector on the Gauss-Labatto points
    // out2D[:M, :] = real(in2D)
    // include normalisation here so that spectral space has same normalisation as it	   
    // started with.

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=0; j<Mf; j++)
	{
	    scratchin[indfft(i,j)] = arrin[indfft(i,j)]/normalise;
	}
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=2; j<Mf; j++)
	{
	    scratchin[indfft(i, Mf-2+j)] = scratchin[indfft(i, Mf-j)];
	}
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*spec_plan);
    if (cnsts.dealiasing)
    {
	// copy zeroth and positive modes into output

	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[ind(0,0)] = (0.5/(Mf-1.0)) * scratchout[indfft(0,0)]; 

	// when dealiasing this will be still x 1.0 not 0.5, because it isn't
	// the last element in the transformed array
	for (j=1; j<M; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[ind(0,j)] = (1.0/(Mf-1.0))*scratchout[indfft(0,j)];
	}


	for (i=1; i<N+1; i++)
	{
	    arrout[ind(i,0)] = (0.5/(Mf-1.0)) * scratchout[indfft(i,0)]; 

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		arrout[ind(i,j)] = (1.0/(Mf-1.0))*scratchout[indfft(i,j)];
	    }
	}
    }

    else
    {
	for (i=0; i<N+1; i++)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(i,0)] = (0.5/(M-1.0)) * scratchout[indfft(i,0)]; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*scratchout[indfft(i,j)];
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    arrout[ind(i,M-1)] = (0.5/(M-1.0)) * scratchout[indfft(i,M-1)]; 
	}
    }

}

void to_spectral(fftw_complex *arrin, fftw_complex *arrout,
	fftw_complex *scratchin, fftw_complex *scratchout,
	fftw_plan *spec_plan,  flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;
    int i,j;

    fftw_complex normalise = 2*Nf+1;

    // Perform the FFT across the x direction   

    // The first half contains the vector on the Gauss-Labatto points
    // out2D[:M, :] = real(in2D)
    // include normalisation here so that spectral space has same normalisation as it	   
    // started with.

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=0; j<Mf; j++)
	{
	    scratchin[indfft(i,j)] = arrin[indfft(i,j)];
	}
    }

    // The second half contains the vector on the Gauss-Labatto points excluding
    // the first and last elements and in reverse order
    // out2D[M:, :] = out2D[M-2:0:-1, :]

    for (i=0; i<2*Nf+1; i++)
    {
	for (j=2; j<Mf; j++)
	{
	    scratchin[indfft(i, Mf-2+j)] = scratchin[indfft(i, Mf-j)];
	}
    }

    // Perform the transformation on this temporary vector
    // out2D = fftpack.fft2(out2D)
    fftw_execute(*spec_plan);

    // Renormalise and divide by c_k to convert to Chebyshev polynomials
    // if dealiasing make sure to zero off the inaccurate modes
    // int k;
    // for (i=1; i<N+1; i++)
    // {

    //     k = 2*N+1 - i;
    //     printf("i = %d, k = %d", i, k);
    //     for (j=0; j<M; j++)
    //     {
    //         printf(" %e ", (creal(arrout[ind(i,j)]) - creal(arrout[ind(k,j)]))/
    //     	    creal(arrout[ind(i,j)]));
    //         printf(" %ej ", (cimag(arrout[ind(i,j)]) + cimag(arrout[ind(k,j)]))/
    //     	    cimag(arrout[ind(i,j)]));
    //         printf("\t %d ", creal(arrout[ind(k,j)])==creal(arrout[ind(i,j)]));
    //         printf(" %d ", cimag(arrout[ind(k,j)])==-cimag(arrout[ind(i,j)]));
    //         printf("\n");
    //     }

    // }

    if (cnsts.dealiasing)
    {
	// copy zeroth and positive modes into output

	// out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	arrout[ind(0,0)] = (0.5/(Mf-1.0)) * scratchout[indfft(0,0)]/normalise; 

	// when dealiasing this will be still x 1.0 not 0.5, because it isn't
	// the last element in the transformed array
	for (j=1; j<M; j++)
	{
	    // out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
	    arrout[ind(0,j)] = (1.0/(Mf-1.0))*scratchout[indfft(0,j)]/normalise;
	}


	for (i=1; i<N+1; i++)
	{
	    arrout[ind(i,0)] = (0.5/(Mf-1.0)) * scratchout[indfft(i,0)]/normalise; 

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		arrout[ind(i,j)] = (1.0/(Mf-1.0))*scratchout[indfft(i,j)]/normalise;
	    }
	}

	// copy negative modes into output
	for (i=1; i<N+1; i++)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(2*N+1-i,0)] = (0.5/(Mf-1.0)) * scratchout[indfft(2*Nf+1-i,0)]/normalise; 

	    // when dealiasing this will be still x 1.0 not 0.5, because it isn't
	    // the last element in the transformed array
	    for (j=1; j<M; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(2*N+1-i,j)] = (1.0/(Mf-1.0))*scratchout[indfft(2*Nf+1-i,j)]/normalise;
	    }
	}
    }

    else
    {
	for (i=0; i<2*N+1; i++)
	{
	    // out2D[0, :] = (0.5/(M-1.0))*out2D[0, :]
	    arrout[ind(i,0)] = (0.5/(M-1.0)) * scratchout[indfft(i,0)]/normalise; 

	    for (j=1; j<M-1; j++)
	    {
		// out2D[1:M-1, :] = (1.0/(M-1.0))*out2D[1:M-1, :]
		arrout[ind(i,j)] = (1.0/(M-1.0))*scratchout[indfft(i,j)]/normalise;
	    }

	    //out2D[M-1, :] = (0.5/(M-1.0))*out2D[M-1, :]
	    arrout[ind(i,M-1)] = (0.5/(M-1.0)) * scratchout[indfft(i,M-1)]/normalise; 
	}
    }

}

void fft_convolve(fftw_complex *arr1, fftw_complex *arr2, fftw_complex *arrout,
	fftw_complex *scratchp1, fftw_complex *scratchp2, fftw_complex
	*scratchin, fftw_complex *scratchout, fftw_plan *phys_plan, fftw_plan
	*spec_plan, flow_params cnsts)
{
    // all scratch arrays must be different
    // out array may be the same as one in array

    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;

    int i, j;

    to_physical(arr1, scratchp1, scratchin, scratchout, phys_plan, cnsts);
    to_physical(arr2, scratchp2, scratchin, scratchout, phys_plan, cnsts);

    for (i=0; i<(2*Nf+1); i++)
    {
	for(j=0; j<Mf; j++)
	{
	    scratchp1[indfft(i,j)] = scratchp2[indfft(i,j)]*scratchp1[indfft(i,j)];
	}
    }

    to_spectral(scratchp1, arrout, scratchin, scratchout, spec_plan, cnsts);
}

void fft_convolve_r(complex *arr1, complex *arr2, complex *arrout,
	double *scratchp1, double *scratchp2, fftw_complex
	*scratchin, fftw_complex *scratchout, fftw_plan *phys_plan, fftw_plan
	*spec_plan, flow_params cnsts)
{
    // all scratch arrays must be different
    // out array may be the same as one in array

    int Nf = cnsts.Nf;
    int Mf = cnsts.Mf;

    int i, j;

    to_physical_r(arr1, scratchp1, scratchin, scratchout, phys_plan, cnsts);
    to_physical_r(arr2, scratchp2, scratchin, scratchout, phys_plan, cnsts);

    for (i=0; i<(2*Nf+1); i++)
    {
	for(j=0; j<Mf; j++)
	{
	    scratchp1[indfft(i,j)] = scratchp2[indfft(i,j)]*scratchp1[indfft(i,j)];
	}
    }

    to_spectral_r(scratchp1, arrout, scratchin, scratchout, spec_plan, cnsts);
}

double calc_KE_mode(fftw_complex *u, fftw_complex *v, int n, flow_params cnsts)
{
    double KE = 0;
    int i=0;
    int m=0;
    int p=0;
    complex usq=0;
    complex vsq=0;
    complex tmpu=0;
    complex tmpv=0;
    int M=cnsts.M;

	for (i=0; i<M; i+=2)
	{
	    usq = 0;
	    vsq = 0;

	    for (m=i-M+1; m<M; m++)
	    {
		p = abs(i-m);

		tmpu = u[ind(n,p)];
		tmpv = v[ind(n,p)];

		if (p==0)
		{
		    tmpu *= 2.0;
		    tmpv *= 2.0;
		}

		tmpu *= conj(u[ind(n,abs(m))]);
		tmpv *= conj(v[ind(n,abs(m))]);

		if (abs(m)==0)
		{
		    tmpu *= 2.0;
		    tmpv *= 2.0;
		}

		if (i==0)
		{
		    usq += 0.25*tmpu;
		    vsq += 0.25*tmpv;
		} else
		{
		    usq += 0.5*tmpu;
		    vsq += 0.5*tmpv;
		}

	    }

	    KE += (2. / (1.-i*i)) * usq;
	    KE += (2. / (1.-i*i)) * vsq;

	}

	if (n == 0)
	{
	    return 0.5*creal(KE);
	} else {
	    return creal(KE);
	}
}

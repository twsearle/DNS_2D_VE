/* -------------------------------------------------------------------------- *
 *									      *
 *  fields_IO.c								      *
 *                                                                            *
 *  functions for input and output of flow fields in C			      *
 *                                                                            *
 *                                                                            *
 * -------------------------------------------------------------------------- */

// Last modified: Wed 30 Sep 15:49:23 2015

#include"fields_2D.h"
#include"fields_IO.h"

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

void save_hdf5_real_arr(char *filename, double *arr, int size)
{
    hid_t  file_id, dataset_id, dataspace_id;
    herr_t status;

    hsize_t arr_size[1];
    arr_size[0] = size;

    // create the filetype for the scipy real numbers
    //filetype_id = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, arr_size, NULL);

    // create the file
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the dataset
    dataset_id = H5Dcreate(file_id, "/psi", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT,
			  H5P_DEFAULT, H5P_DEFAULT);

    // write the file 
    status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
		      H5P_DEFAULT, arr); 

    // clean up 
    status = H5Dclose(dataset_id);
    status = H5Sclose(dataspace_id);

    // close the file
    status = H5Fclose(file_id);

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

void save_hdf5_state_visco(hid_t *file_id, hid_t *filetype_id, hid_t *datatype_id,
	complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, flow_params cnsts)
{
    int N = cnsts.N;
    int M = cnsts.M;
    int i, j;

    hid_t  dataset_id, dataspace_id;
    herr_t status;
    complex_hdf *wdata;

    hsize_t arr_size[1];
    arr_size[0] = (N+1)*M;

    wdata = (complex_hdf *) malloc (arr_size[0] * sizeof (complex_hdf));

    // create the dataspace
    dataspace_id = H5Screate_simple(1, arr_size, NULL);

    // create the dataset
    dataset_id = H5Dcreate(*file_id, "psi", *filetype_id, dataspace_id, H5P_DEFAULT,
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
    dataset_id = H5Dcreate(*file_id, "cxx", *filetype_id, dataspace_id, H5P_DEFAULT,
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
    dataset_id = H5Dcreate(*file_id, "cyy", *filetype_id, dataspace_id, H5P_DEFAULT,
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
    dataset_id = H5Dcreate(*file_id, "cxy", *filetype_id, dataspace_id, H5P_DEFAULT,
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
    free(wdata);

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
	complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, double time, flow_params cnsts)
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

void load_hdf5_state_visco(char *filename, complex_d *psi, complex_d *cxx, complex_d *cyy, complex_d *cxy, flow_params cnsts)
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


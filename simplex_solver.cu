#include"core_class.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

/*__global__ void test_simplex_table1(simplex_table_cuda st)
{
    st.slack_var[threadIdx.x*st.slack_var_size_row+threadIdx.y]+=255;
}

__global__ void test_simplex_table2(simplex_table_cuda st)
{
    st.slack_var[threadIdx.x*st.slack_var_size_row+threadIdx.y]-=255;
}*/

void copy_table_to_vram(simplex_table_cuda *st_d,simplex_table_cuda *st)
{
    st_d->basic_var_size_col=st->basic_var_size_col;
    st_d->basic_var_size_row=st->basic_var_size_row;
    cudaMalloc(&st_d->basic_var,sizeof(float)*st_d->basic_var_size_col*st_d->basic_var_size_row);
    cudaMemcpy(st_d->basic_var,st->basic_var,sizeof(float)*st_d->basic_var_size_col*st_d->basic_var_size_row,cudaMemcpyHostToDevice);

    st_d->c_id_size=st->c_id_size;
    cudaMalloc(&st_d->c_id,sizeof(id)*st_d->c_id_size);
    cudaMemcpy(st_d->c_id,st->c_id,sizeof(id)*st_d->c_id_size,cudaMemcpyHostToDevice);
    
    st_d->r_id_size=st->r_id_size;
    cudaMalloc(&st_d->r_id,sizeof(id)*st_d->r_id_size);
    cudaMemcpy(st_d->r_id,st->r_id,sizeof(id)*st_d->r_id_size,cudaMemcpyHostToDevice);
    
    st_d->slack_var_size_col=st->slack_var_size_col;
    st_d->slack_var_size_row=st->slack_var_size_row;
    cudaMalloc(&st_d->slack_var,sizeof(float)*st_d->slack_var_size_col*st_d->slack_var_size_row);
    cudaMemcpy(st_d->slack_var,st->slack_var,sizeof(float)*st_d->slack_var_size_col*st_d->slack_var_size_row,cudaMemcpyHostToDevice);
}

void copy_table_to_ram(simplex_table_cuda *st,simplex_table_cuda *st_d)
{
    cudaMemcpy(st->basic_var,st_d->basic_var,sizeof(float)*st_d->basic_var_size_col*st_d->basic_var_size_row,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->c_id,st_d->c_id,sizeof(id)*st_d->c_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->r_id,st_d->r_id,sizeof(id)*st_d->r_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->slack_var,st_d->slack_var,sizeof(float)*st_d->slack_var_size_col*st_d->slack_var_size_row,cudaMemcpyDeviceToHost);
}

void simplex_solver(simplex_table_cuda *st)
{
    //transfer simplex table to vram
    simplex_table_cuda st_d;
    copy_table_to_vram(&st_d,st);
    
    
    /*
    dim3 thread_vec(st_d.slack_var_size_row,st_d.slack_var_size_col,1);
    test_simplex_table1<<<1,thread_vec>>>(st_d);
    cudaDeviceSynchronize();
    test_simplex_table2<<<1,thread_vec>>>(st_d);
    cudaDeviceSynchronize();
    copy_table_to_ram(st,&st_d);
    */
}
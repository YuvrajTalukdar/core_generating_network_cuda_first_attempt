#include"core_class.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

__global__ void view_cdp_vec_cuda(converted_data_pack_cuda* cdp_vec)
{
    printf("\nthreadIdx/cdp: %d fi: %d fl: %f corrupt: %d",threadIdx.x,cdp_vec[threadIdx.x].firing_neuron_index,cdp_vec[threadIdx.x].firing_label,cdp_vec[threadIdx.x].corupt_pack);
}

__device__ float get_element(float* data,int x,int y,int size)
{
    return data[y*size+x];
}

__global__ void view_f_nf_data(float* firing_data,int width,int height)
{
    if(threadIdx.y==height-1)
    {
        printf("%f,",firing_data[threadIdx.y*width+threadIdx.x]);
    }
}

__global__ void view_data(converted_data_pack_f_nf_cuda* f_nf_vec)
{
    if(threadIdx.y==58)
    printf("%f,",f_nf_vec[0].firing_data_arr[threadIdx.y*f_nf_vec[0].horizontal_size+threadIdx.x]);
}

__global__ void start_lp_solver()
{
    
}

void simplex_solver(vector<converted_data_pack> &cdps,datapack_structure_defination &ds,ann &network1)
{
    vector<converted_data_pack_cuda> cdp_d_vec;
    vector<converted_data_pack_f_nf_cuda> f_nf_vec;
    //float *fdp_arr[cdps.size()],*nfdp_arr[cdps.size()];
    int height,width,data_size;
    for(int a=0;a<cdps.size();a++)
    {
        converted_data_pack_cuda cdp_cuda;
        cdp_cuda.corupt_pack=cdps[a].corupt_pack;
        cdp_cuda.firing_label=cdps[a].firing_label;
        cdp_cuda.firing_neuron_index=cdps[a].firing_neuron_index;

        converted_data_pack_f_nf_cuda f_nf_data;
        f_nf_data.horizontal_size=cdps[a].firing_data[0].size();
        width=f_nf_data.horizontal_size;
        //preparing firing data and copying it to vram
        height=cdps[a].firing_data.size();
        data_size=sizeof(float)*height*width;
        float* firing_data=(float*)malloc(data_size);
        for(int b=0;b<cdps[a].firing_data.size();b++)
        {
            for(int c=0;c<cdps[a].firing_data[b].size();c++)
            {   *(firing_data+(b*width+c))=cdps[a].firing_data[b][c];}
        }
        float* firing_data_d;
        cudaMalloc(&firing_data_d,data_size);
        cudaMemcpy(firing_data_d,firing_data,data_size,cudaMemcpyHostToDevice);
        f_nf_data.firing_data_height=height;
        f_nf_data.firing_data_arr=(firing_data_d);
        
        //preparing not firing data and copying it to vram
        height=cdps[a].not_firing_data.size();
        data_size=sizeof(float)*height*width;
        float* not_firing_data=(float*)malloc(data_size);
        for(int b=0;b<cdps[a].not_firing_data.size();b++)
        {
            for(int c=0;c<cdps[a].not_firing_data[b].size();c++)
            {   *(not_firing_data+b*width+c)=cdps[a].not_firing_data[b][c];}
        }
        float* not_firing_data_d;
        cudaMalloc(&not_firing_data_d,data_size);
        cudaMemcpy(not_firing_data_d,not_firing_data,data_size,cudaMemcpyHostToDevice);
        f_nf_data.not_firing_data_height=height;
        f_nf_data.not_firing_data_arr=(not_firing_data_d);

        cdp_d_vec.push_back(cdp_cuda);
        f_nf_vec.push_back(f_nf_data);
    }
    //copying rest of the cdp data to vram
    thrust::device_vector<converted_data_pack_cuda> cdps_cuda_thrust=cdp_d_vec;
    converted_data_pack_cuda* cdp_vec_cuda=thrust::raw_pointer_cast(cdps_cuda_thrust.data());
    //copying the firing and not firing data pointers to vram
    thrust::device_vector<converted_data_pack_f_nf_cuda> f_nf_vec_thrust=f_nf_vec;
    converted_data_pack_f_nf_cuda* f_nf_data=thrust::raw_pointer_cast(f_nf_vec_thrust.data());

    
    
}

/*
cout<<"\n\ncopying done f_data_size="<<f_nf_vec[0].firing_data_height<<endl;
    sleep(1);
*/

/*
dim3 thread_vector(f_nf_vec[0].horizontal_size,f_nf_vec[0].firing_data_height,1);
    view_data<<<1,thread_vector>>>(f_nf_data);
    cudaDeviceSynchronize();
    int gh;cin>>gh;
/*
dim3 thread_vector(width,height,1);
cout<<"\nwidth="<<width<<" height="<<height<<endl;
sleep(1);
view_f_nf_data<<<1,thread_vector>>>(fdp_arr_d,width,height);
cudaDeviceSynchronize();
int gh;cin>>gh;*/
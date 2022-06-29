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

void free_simplex_table_from_vram(simplex_table_cuda st_d)
{
    cudaFree(st_d.basic_var);
    cudaFree(st_d.c_id);
    cudaFree(st_d.r_id);
    cudaFree(st_d.rhs);
    cudaFree(st_d.slack_var);
    cudaFree(st_d.theta);
}

__global__ void termination_condition_checker_kernel(simplex_table_cuda st,bool *status)//need to be checked
{
    int index=threadIdx.x*blockIdx.x+threadIdx.x;
    if(index<st.slack_var_size_row)
    {
        if(index<st.slack_var_size_row && st.slack_var[index*st.slack_var_size_row+st.r_id[index].id-st.basic_var_size_col]<0 && st.rhs[index]>=0)
        {   *status=false;}
    }
}

bool termination_condition_checker(simplex_table_cuda st_d)//need to be checked
{
    bool status=true;
    int no_of_threads,no_of_blocks=1;
    if(st_d.slack_var_size_row>1024)
    {   
        no_of_threads=512;
        no_of_blocks=st_d.slack_var_size_row/512;
        if(st_d.slack_var_size_row%512>0)
        {   no_of_blocks++;}
    }
    else
    {   no_of_threads=st_d.slack_var_size_row;}
    
    bool *status_d;
    cudaMalloc(&status_d,sizeof(bool));
    cudaMemcpy(status_d,&status,sizeof(bool),cudaMemcpyHostToDevice);
    termination_condition_checker_kernel<<<no_of_blocks,no_of_threads>>>(st_d,status_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&status,status_d,sizeof(bool),cudaMemcpyDeviceToHost);
    cudaFree(status_d);

    return status;
}

__global__ void find_row_with_negative_slack_kernel(simplex_table_cuda st,int *row_with_negative_slack)//initial test passed
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.slack_var_size_row)
    {
        //printf("\nindex: %d basic_size_row: %d basic_col_size: %d slack_index: %d id: %d  slack_row: %d slack_col: %d rhs_size: %d",index,st.basic_var_size_row,st.basic_var_size_col,index*st.slack_var_size_col+(st.r_id[index].id-st.basic_var_size_col),st.r_id[index].id,st.slack_var_size_row,st.slack_var_size_col,st.rhs_size);
        if(st.slack_var[index*st.slack_var_size_col+(st.r_id[index].id-st.basic_var_size_col)]<0 && st.rhs[index]>=0)//originally it was just rhs>0, but now i feel it shouls be >=. Need further testing
        {
            if(*row_with_negative_slack==-1 || *row_with_negative_slack>index)
            {   *row_with_negative_slack=index;}
        }
    }
}

int find_row_with_negative_slack(simplex_table_cuda st_d)//initial test passed
{
    int row_with_negative_slack=-1;
    int no_of_threads,no_of_blocks=1;
    if(st_d.slack_var_size_row>1024)
    {   
        no_of_threads=512;
        no_of_blocks=st_d.slack_var_size_row/512;
        if(st_d.slack_var_size_row%512>0)
        {   no_of_blocks++;}
    }
    else
    {   no_of_threads=st_d.slack_var_size_row;}
    cout<<"\nno_of_threads: "<<no_of_threads;
    int *row_with_negative_slack_d;
    cudaMalloc(&row_with_negative_slack_d,sizeof(int));
    cudaMemcpy(row_with_negative_slack_d,&row_with_negative_slack,sizeof(int),cudaMemcpyHostToDevice);
    find_row_with_negative_slack_kernel<<<no_of_blocks,no_of_threads>>>(st_d,row_with_negative_slack_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&row_with_negative_slack,row_with_negative_slack_d,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(row_with_negative_slack_d);

    return row_with_negative_slack;
}

__global__ void pivote_col_finder_kernel(simplex_table_cuda st,int *pivote_col,int row_with_negative_slack,bool basic_var)//initial test passed
{
    if(basic_var)
    {
        if(st.basic_var[row_with_negative_slack*st.basic_var_size_col+threadIdx.x]>0)
        {
            if(*pivote_col==-1 || *pivote_col>threadIdx.x)
            {   *pivote_col=threadIdx.x;}
        }
    }
    else
    {
        int index=blockIdx.x*512+threadIdx.x;
        if(index<st.slack_var_size_row && st.slack_var[row_with_negative_slack*st.slack_var_size_col+index]>0)
        {
            if(*pivote_col==-1 || *pivote_col>index)
            {   *pivote_col=index+st.basic_var_size_col;}
        }
    }
}

int pivote_col_finder(simplex_table_cuda st_d,int row_with_negative_slack)//initial test passed
{
    //basic variable col size cannot be more than 60 as max possible horizontal data size id 30 fixed by genetic algorithm.
    int pivote_col=-1;
    int no_of_threads=st_d.basic_var_size_col,no_of_blocks=1;

    int *pivote_col_d;
    cudaMalloc(&pivote_col_d,sizeof(int));
    cudaMemcpy(pivote_col_d,&pivote_col,sizeof(int),cudaMemcpyHostToDevice);
    pivote_col_finder_kernel<<<no_of_blocks,no_of_threads>>>(st_d,pivote_col_d,row_with_negative_slack,true);
    cudaDeviceSynchronize();
    cudaMemcpy(&pivote_col,pivote_col_d,sizeof(int),cudaMemcpyDeviceToHost);
    if(pivote_col==-1)//check in slack variable
    {   
        if(st_d.slack_var_size_col>1024)
        {   
            no_of_threads=512;
            no_of_blocks=st_d.slack_var_size_col/512;
            if(st_d.slack_var_size_col%512>0)
            {   no_of_blocks++;}
        }
        else
        {   no_of_threads=st_d.slack_var_size_col;}
        pivote_col_finder_kernel<<<no_of_blocks,no_of_threads>>>(st_d,pivote_col_d,row_with_negative_slack,false);
        cudaDeviceSynchronize();
        cudaMemcpy(&pivote_col,pivote_col_d,sizeof(int),cudaMemcpyDeviceToHost);
    }
    cudaFree(pivote_col_d);

    return pivote_col;
}

vector<int> conflict_data_finder(simplex_table_cuda st_d)//need to be checked
{
    vector<int> conflict_id;
    double *rhs;
    rhs=(double*)malloc(sizeof(double)*st_d.rhs_size);
    cudaMemcpy(rhs,st_d.rhs,sizeof(double)*st_d.rhs_size,cudaMemcpyDeviceToHost);
    float *slack_var;
    slack_var=(float*)malloc(sizeof(float)*st_d.slack_var_size_col*st_d.slack_var_size_row);
    cudaMemcpy(slack_var,st_d.slack_var,sizeof(float)*st_d.slack_var_size_col*st_d.slack_var_size_row,cudaMemcpyDeviceToHost);
    id *r_id;
    r_id=(id*)malloc(sizeof(id)*st_d.r_id_size);
    cudaMemcpy(r_id,st_d.r_id,sizeof(id)*st_d.r_id_size,cudaMemcpyDeviceToHost);
    for(int a=0;a<st_d.r_id_size;a++)
    {
        if(slack_var[a*st_d.slack_var_size_col+r_id[a].id-st_d.basic_var_size_col] && rhs[a]>0)
        {   conflict_id.push_back(a);}
    }

    free(rhs);
    free(slack_var);
    free(r_id);

    return conflict_id;
}

__global__ void pivote_row_finder_kernel(simplex_table_cuda st,int pivote_col_index)//initial test passed
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.basic_var_size_row)
    {
        if(pivote_col_index<st.basic_var_size_col)
        {
            //printf("\nfirst set basic_var: %f rhs: %f",(double)st.basic_var[index*st.basic_var_size_col+pivote_col_index],st.rhs[index]);
            if(st.basic_var[index*st.basic_var_size_col+pivote_col_index]==0)
            {   st.theta[index]=0;}
            else
            {   st.theta[index]=st.rhs[index]/(double)st.basic_var[index*st.basic_var_size_col+pivote_col_index];}
        }
        else
        {
            int temp_col_index=pivote_col_index-st.basic_var_size_col;
            //printf("\nsecond set basic_var: %f theta: %f",st.slack_var[index*st.slack_var_size_col+temp_col_index],st.rhs[index]);
            if(st.slack_var[index*st.slack_var_size_col+temp_col_index]==0)
            {   st.theta[index]=0;}
            else
            {   st.theta[index]=st.rhs[index]/(double)st.slack_var[index*st.slack_var_size_col+temp_col_index];}
        }
    }
}

int pivote_row_finder(simplex_table_cuda st_d,int pivote_col)//initial test passed
{
    int pivote_row_index=-1;
    st_d.theta_size=st_d.r_id_size;
    cudaMalloc(&st_d.theta,sizeof(double)*st_d.theta_size);
    
    //launch kernel
    int no_of_threads,no_of_blocks=1;
    if(st_d.r_id_size>1024)
    {   
        no_of_threads=512;
        no_of_blocks=st_d.r_id_size/512;
        if(st_d.r_id_size%512>0)
        {   no_of_blocks++;}
    }
    else
    {   no_of_threads=st_d.slack_var_size_row;}

    pivote_row_finder_kernel<<<no_of_blocks,no_of_threads>>>(st_d,pivote_col);
    
    double *theta;
    theta=(double*)malloc(sizeof(double)*st_d.theta_size);
    cudaMemcpy(theta,st_d.theta,sizeof(double)*st_d.theta_size,cudaMemcpyDeviceToHost);
    double smallest_positive_theta=-1;
    //cout<<"\ntheta_size: "<<st_d.theta_size;
    for(int a=0;a<st_d.theta_size;a++)
    {   
        //cout<<"\ntheta="<<theta[a];
        if(theta[a]>0)
        {
            if(smallest_positive_theta==-1 || smallest_positive_theta>theta[a])
            {
                //cout<<"\n check="<<theta[a];
                pivote_row_index=a;
                smallest_positive_theta=theta[a];
            }
        }
    }

    return pivote_row_index;
}

vector<int> pivote_element_finder(simplex_table_cuda st_d)
{
    vector<int> conflict_id;
    int row_with_negative_slack;
    int pivote_col_index,pivote_row_index;
    do
    {
        row_with_negative_slack=find_row_with_negative_slack(st_d);//if not found it will return -1.
        cout<<"\n\nrow_with_negative_slack= "<<row_with_negative_slack;
        if(row_with_negative_slack>=0)
        {
            pivote_col_index=pivote_col_finder(st_d,row_with_negative_slack);
            cout<<"\npivote_col_index= "<<pivote_col_index;
            if(pivote_col_index<0)//it should have been ==-1 but to handle potential pricision problem
            {
                conflict_id=conflict_data_finder(st_d);
                break;
            }
            if(st_d.theta_size!=0)
            {
                cudaFree(st_d.theta);
                st_d.theta_size=0;
            }
            pivote_row_index=pivote_row_finder(st_d,pivote_col_index);
            cout<<"\npivote_row_index: "<<pivote_row_index;
            int gh;cin>>gh;
            if(pivote_row_index<=0)//bad_p_row_index_status
            {   break;}
            //simplex_table_modifier
        }
        else
        {   break;}
    } 
    while(!termination_condition_checker(st_d));

    return conflict_id;
}

void copy_table_to_vram(simplex_table_cuda *st_d,simplex_table_cuda *st)//ok tested
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

    st_d->rhs_size=st->rhs_size;
    cudaMalloc(&st_d->rhs,sizeof(double)*st_d->rhs_size);
    cudaMemcpy(st_d->rhs,st->rhs,sizeof(double)*st_d->rhs_size,cudaMemcpyHostToDevice);
}

void copy_table_to_ram(simplex_table_cuda *st,simplex_table_cuda *st_d)
{
    cudaMemcpy(st->basic_var,st_d->basic_var,sizeof(float)*st_d->basic_var_size_col*st_d->basic_var_size_row,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->c_id,st_d->c_id,sizeof(id)*st_d->c_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->r_id,st_d->r_id,sizeof(id)*st_d->r_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->slack_var,st_d->slack_var,sizeof(float)*st_d->slack_var_size_col*st_d->slack_var_size_row,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->rhs,st_d->rhs,sizeof(double)*st_d->rhs_size,cudaMemcpyDeviceToHost);
}

void simplex_solver(simplex_table_cuda* st)
{
    //transfer simplex table to vram
    simplex_table_cuda st_d;
    copy_table_to_vram(&st_d,st);
    vector<int> conflict_id=pivote_element_finder(st_d);
    
    /*
    dim3 thread_vec(st_d.slack_var_size_row,st_d.slack_var_size_col,1);
    test_simplex_table1<<<1,thread_vec>>>(st_d);
    cudaDeviceSynchronize();
    test_simplex_table2<<<1,thread_vec>>>(st_d);
    cudaDeviceSynchronize();
    copy_table_to_ram(st,&st_d);
    */
}
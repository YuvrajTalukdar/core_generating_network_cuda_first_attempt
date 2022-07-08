#include"core_class.h"
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

void check(cudaError x) {
    fprintf(stderr, "%s\n", cudaGetErrorString(x));
}

__global__ void change_row_id(simplex_table_cuda st,int p_row_index,int p_col_index)//ok check
{
    switch(threadIdx.x)
    {
        case 0:
        st.r_id[p_row_index].basic=st.c_id[p_col_index].basic;
        break;
        case 1:
        st.r_id[p_row_index].id=st.c_id[p_col_index].id;
        break;
        case 2:
        st.r_id[p_row_index].rhs=st.c_id[p_col_index].rhs;;
        break;
        case 3:
        st.r_id[p_row_index].slack=st.c_id[p_col_index].slack;
        break;
        case 4:
        st.r_id[p_row_index].theta=st.c_id[p_col_index].theta;
        break;
        default:
    }
}

__global__ void pivot_row_modifier(simplex_table_cuda st,float *pe,int p_row_index,int p_col_index)//ok check
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.basic_var_size_col)
    {   st.basic_var[p_row_index*st.basic_var_size_col+index]/=*pe;/*printf("\npe: %f basic_var: %f,",*pe,st.basic_var[p_row_index*st.basic_var_size_col+index]);*/}
    else if(index>=st.basic_var_size_col && index<(st.basic_var_size_col+st.slack_var_size_col))
    {
        int slack_col_index=index-st.basic_var_size_col;
        st.slack_var[p_row_index*st.slack_var_size_col+slack_col_index]/=*pe;
        //printf("\npe: %f slack_var: %d",*pe,st.slack_var[p_row_index*st.slack_var_size_col+slack_col_index]);
    }
    else if(index==(st.basic_var_size_col+st.slack_var_size_col))
    {   st.rhs[p_row_index]/=*pe;/*printf("\npe: %f rhs: %f",*pe,st.rhs[p_row_index]);*/}
}

__global__ void rest_of_row_modifier(simplex_table_cuda st,float *multiplying_element_arr,int p_row_index,int p_col_index)//ok check
{
    int index_col=blockIdx.x*512+threadIdx.x;
    //row is blockIdx.y
    if(index_col<st.basic_var_size_col+st.slack_var_size_col)
    {
        if(blockIdx.y!=p_row_index)//all row accept pivot row
        {
            if(index_col<st.basic_var_size_col)//basic_point
            {   
                st.basic_var[blockIdx.y*st.basic_var_size_col+index_col]-=(multiplying_element_arr[blockIdx.y]*st.basic_var[p_row_index*st.basic_var_size_col+index_col]);
            }
            else if(index_col<=st.basic_var_size_col && index_col<(st.basic_var_size_col+st.slack_var_size_col))
            {
                int slack_col_index=index_col-st.basic_var_size_col;
                st.slack_var[blockIdx.y*st.slack_var_size_col+slack_col_index]-=(multiplying_element_arr[blockIdx.y]*st.slack_var[p_row_index*st.slack_var_size_col+slack_col_index]);
            }
        }
    }
    else if(index_col==st.basic_var_size_col+st.slack_var_size_col)
    {
        if(blockIdx.y!=p_row_index)
        {   st.rhs[blockIdx.y]-=multiplying_element_arr[blockIdx.y]*st.rhs[p_row_index];}
    }
    //if(threadIdx.x==0 && blockIdx.x==0)
    //printf("\nblockIdx.y: %d me: %f",blockIdx.y,multiplying_element_arr[blockIdx.y]);
}

__global__ void get_multiplying_elements(simplex_table_cuda st,int p_col_index,float *multiplying_element_arr)//ok check
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.basic_var_size_row)
    {
        if(p_col_index<st.basic_var_size_col)
        {
            multiplying_element_arr[index]=st.basic_var[index*st.basic_var_size_col+p_col_index];
        }
        else
        {
            int p_col_temp=p_col_index-st.basic_var_size_col;
            multiplying_element_arr[index]=st.slack_var[index*st.slack_var_size_col+p_col_temp];
        }
    }    
}

void copy_table_to_ram(simplex_table_cuda *st,simplex_table_cuda *st_d);

void simplex_table_modifier(simplex_table_cuda st_d,/*simplex_table_cuda *st,*/float *pe_d,float *multiplying_element_arr_d,int p_row_index,int p_col_index)//ok chech
{
    int total_no_of_threads_required;
    int no_of_thread,no_of_blocks;
    change_row_id<<<1,5>>>(st_d,p_row_index,p_col_index);
    //pivot row modifiew
    total_no_of_threads_required=st_d.basic_var_size_col+st_d.slack_var_size_col+1;//extra 1 for rhs
    no_of_blocks=total_no_of_threads_required/512;
    if(no_of_blocks==0)
    {   no_of_thread=total_no_of_threads_required;no_of_blocks=1;}
    else
    {   no_of_thread=512;no_of_blocks++;}
    pivot_row_modifier<<<no_of_blocks,no_of_thread>>>(st_d,pe_d,p_row_index,p_col_index);
    cudaDeviceSynchronize();
    no_of_blocks=st_d.basic_var_size_row/512;
    if(no_of_blocks==0)
    {   no_of_thread=st_d.basic_var_size_row;no_of_blocks++;}
    else
    {   
        no_of_thread=512;
        if(st_d.basic_var_size_row%512!=0)
        {   no_of_blocks++;}
    }
    get_multiplying_elements<<<no_of_blocks,no_of_thread>>>(st_d,p_col_index,multiplying_element_arr_d);
    //copy_table_to_ram(st,&st_d);
    //display_st(st);
    //cout<<"\np row modified";
    //int gh;cin>>gh;
    //rest of the row modifiew
    total_no_of_threads_required=st_d.basic_var_size_row*(total_no_of_threads_required);
    int block_x,block_y;
    block_y=st_d.basic_var_size_row;//rows
    block_x=(st_d.basic_var_size_col+st_d.slack_var_size_col)/512;//cols part 1
    if(block_x==0)//cols part 2
    {   no_of_thread=(st_d.basic_var_size_col+st_d.slack_var_size_col+1);block_x=1;}
    else
    {   no_of_thread=512;block_x++;}
    dim3 block_vec(block_x,block_y,1);
    rest_of_row_modifier<<<block_vec,no_of_thread>>>(st_d,multiplying_element_arr_d,p_row_index,p_col_index);
    cudaDeviceSynchronize();
    //copy_table_to_ram(st,&st_d);
    //display_st(st);
    //cout<<"\nrest row modified";
    //cin>>gh;
}

__global__ void termination_condition_checker_kernel(simplex_table_cuda st,bool *status)//ok check
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.slack_var_size_row)
    {
        if(st.r_id[index].slack)
        {
            if(st.slack_var[index*st.slack_var_size_row+st.r_id[index].id-st.basic_var_size_col]<0 && st.rhs[index]>=0)
            {   *status=false;}
            //printf("\ntc: %d slack: %f rhs: %f index: %d slack_size: %d",*status,st.slack_var[index*st.slack_var_size_row+st.r_id[index].id-st.basic_var_size_col],st.rhs[index],index,st.slack_var_size_row);
        }
    }
}

bool termination_condition_checker(simplex_table_cuda st_d)//ok check
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
    //cout<<"\ntermination: "<<status;

    return status;
}

__global__ void find_row_with_negative_slack_kernel(simplex_table_cuda st,int *row_with_negative_slack)//ok check
{
    int index=blockIdx.x*512+threadIdx.x;
    if(index<st.slack_var_size_row)
    {
        //printf("\nindex: %d basic_size_row: %d basic_col_size: %d slack_index: %d id: %d  slack_row: %d slack_col: %d rhs_size: %d",index,st.basic_var_size_row,st.basic_var_size_col,index*st.slack_var_size_col+(st.r_id[index].id-st.basic_var_size_col),st.r_id[index].id,st.slack_var_size_row,st.slack_var_size_col,st.rhs_size);
        if(st.r_id[index].slack==true)
        {
            //int r_id_stuff=st.r_id[index].id-st.basic_var_size_col;
            //int slack_index=index*st.slack_var_size_col+(r_id_stuff);
            if(st.slack_var[index*st.slack_var_size_col+(st.r_id[index].id-st.basic_var_size_col)]<0 && st.rhs[index]>=0)//originally it was just rhs>0, but now i feel it shouls be >=. Need further testing
            {
                if(*row_with_negative_slack==-1 || *row_with_negative_slack>index)
                {   *row_with_negative_slack=index;}
            }
        }
    }
}

int find_row_with_negative_slack(simplex_table_cuda st_d)//ok check
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
    int *row_with_negative_slack_d;
    cudaMalloc(&row_with_negative_slack_d,sizeof(int));
    cudaMemcpy(row_with_negative_slack_d,&row_with_negative_slack,sizeof(int),cudaMemcpyHostToDevice);
    find_row_with_negative_slack_kernel<<<no_of_blocks,no_of_threads>>>(st_d,row_with_negative_slack_d);
    cudaDeviceSynchronize();
    cudaMemcpy(&row_with_negative_slack,row_with_negative_slack_d,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(row_with_negative_slack_d);

    return row_with_negative_slack;
}

int pivote_col_finder(simplex_table_cuda st_d,simplex_table_cuda *st,int row_with_negative_slack)//ok check
{
    int pivote_col=-1;
    cudaMemcpy(st->basic_var,st_d.basic_var,sizeof(float)*st_d.basic_var_size_col*st_d.basic_var_size_row,cudaMemcpyDeviceToHost);
    for(int a=0;a<st_d.basic_var_size_col;a++)
    {
        if(st->basic_var[row_with_negative_slack*st->basic_var_size_col+a]>0)
        {   pivote_col=a;break;}
    }
    if(pivote_col==-1)
    {
        cudaMemcpy(st->slack_var,st_d.slack_var,sizeof(float)*st_d.slack_var_size_col*st_d.slack_var_size_row,cudaMemcpyDeviceToHost);
        for(int a=0;a<st_d.slack_var_size_col;a++)
        {   
            if(st->slack_var[row_with_negative_slack*st->slack_var_size_col+a]>0)
            {   pivote_col=a+st->basic_var_size_col;break;}
        }
    }

    return pivote_col;
}

vector<int> conflicting_data_finder(simplex_table_cuda st_d)//need to be checked
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
        if(r_id[a].slack && slack_var[a*st_d.slack_var_size_col+r_id[a].id-st_d.basic_var_size_col]<0 && rhs[a]>0)
        {   conflict_id.push_back(a);}
    }

    free(rhs);
    free(slack_var);
    free(r_id);

    return conflict_id;
}

__global__ void pivote_row_finder_kernel(simplex_table_cuda st,int pivote_col_index)//ok check
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

int pivote_row_finder(simplex_table_cuda st_d,int pivote_col)//ok check
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
                pivote_row_index=a;
                smallest_positive_theta=theta[a];
            }
        }
    }

    return pivote_row_index;
}

__global__ void get_pivot_element(simplex_table_cuda st,int p_row_index,int p_col_index,float *pe)//ok check
{
    if(p_col_index<st.basic_var_size_col)
    {
        *pe=st.basic_var[p_row_index*st.basic_var_size_col+p_col_index];
    }
    else
    {
        int slack_p_col=p_col_index-st.basic_var_size_col;
        *pe=st.slack_var[p_row_index*st.slack_var_size_col+slack_p_col];
    }
}

bool check_for_cyclic_bug(int p_col_index,int p_row_index,buffer &buffer_obj)//ok check. This function is for cyclic bug checking
{
    if(buffer_obj.p_col_index.size()<4 && buffer_obj.p_row_index.size()<4)
    {
        buffer_obj.p_col_index.push_back(p_col_index);
        buffer_obj.p_row_index.push_back(p_row_index);
        return false;
    }
    else
    {
        bool status=false;
        for(int a=0;a<buffer_obj.p_row_index.size();a++)
        {
            if(buffer_obj.p_row_index[a]==p_row_index && buffer_obj.p_col_index[a]==p_col_index)
            {   status=true;}
        }
        if(status==true)
        {   return status;}
        else
        {
            buffer_obj.p_col_index.push_back(p_col_index);
            buffer_obj.p_row_index.push_back(p_row_index);
            buffer_obj.p_col_index.erase(buffer_obj.p_col_index.begin());
            buffer_obj.p_row_index.erase(buffer_obj.p_row_index.begin());
            return false;
        }
    }
}

void free_simplex_table_from_vram(simplex_table_cuda st_d)//need to be checked
{
    cudaFree(st_d.basic_var);
    cudaFree(st_d.c_id);
    cudaFree(st_d.r_id);
    cudaFree(st_d.rhs);
    cudaFree(st_d.slack_var);
    cudaFree(st_d.theta);
}

vector<int> pivot_element_finder(simplex_table_cuda st_d,simplex_table_cuda* st)
{
    vector<int> conflict_id;
    int row_with_negative_slack;
    int p_col_index,p_row_index;
    buffer buffer_obj;
    buffer_obj.p_col_index.clear();
    buffer_obj.p_row_index.clear();
    //int iteration=0;
    float *multiplying_element_arr_d;
    cudaMalloc(&multiplying_element_arr_d,sizeof(float)*st->slack_var_size_row);
    do
    {
        //display_st(st);
        //cout<<"\niteration: "<<iteration<<" ";
        //iteration++;
        //int gh;cin>>gh;
        row_with_negative_slack=find_row_with_negative_slack(st_d);//if not found it will return -1.
        //cout<<"\n\nrow_with_negative_slack= "<<row_with_negative_slack;
        if(row_with_negative_slack>=0)
        {
            p_col_index=pivote_col_finder(st_d,st,row_with_negative_slack);
            cudaDeviceSynchronize();
            //cout<<"\npivote_col_index= "<<p_col_index;
            if(p_col_index<0)//it should have been ==-1 but to handle potential precision problem,//this function is to check if data is conflicting type
            {
                //cout<<"\nconflict found!";
                conflict_id=conflicting_data_finder(st_d);
                break;
            }
            if(st_d.theta_size!=0)
            {
                cudaFree(st_d.theta);
                st_d.theta_size=0;
            }
            p_row_index=pivote_row_finder(st_d,p_col_index);
            cudaDeviceSynchronize();
            //cout<<"\npivote_row_index: "<<p_row_index;
            if(p_row_index<0)//bad_p_row_index_status
            {   break;}
            float *pe_d;
            cudaMalloc(&pe_d,sizeof(float));
            get_pivot_element<<<1,1>>>(st_d,p_row_index,p_col_index,pe_d);
            cudaDeviceSynchronize();
            //simplex_table_modifier
            if(!check_for_cyclic_bug(p_col_index,p_row_index,buffer_obj))//this is to check for cyclic bug
            {   simplex_table_modifier(st_d,/*st,*/pe_d,multiplying_element_arr_d,p_row_index,p_col_index);}
            else
            {   
                //cout<<"\ncyclic bug";
                conflict_id=conflicting_data_finder(st_d);
                break;
            }//cyclic bug present
            cudaFree(pe_d);
        }
        else
        {   break;}
    } 
    while(!termination_condition_checker(st_d));
    cudaFree(multiplying_element_arr_d);
    
    return conflict_id;
}

void copy_table_to_vram(simplex_table_cuda *st_d,simplex_table_cuda *st)//ok check
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

void copy_table_to_ram(simplex_table_cuda *st,simplex_table_cuda *st_d)//ok check
{
    cudaMemcpy(st->basic_var,st_d->basic_var,sizeof(float)*st_d->basic_var_size_col*st_d->basic_var_size_row,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->c_id,st_d->c_id,sizeof(id)*st_d->c_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->r_id,st_d->r_id,sizeof(id)*st_d->r_id_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->slack_var,st_d->slack_var,sizeof(float)*st_d->slack_var_size_col*st_d->slack_var_size_row,cudaMemcpyDeviceToHost);
    cudaMemcpy(st->rhs,st_d->rhs,sizeof(double)*st_d->rhs_size,cudaMemcpyDeviceToHost);
}

vector<int> simplex_solver(simplex_table_cuda* st)
{
    //transfer simplex table to vram
    simplex_table_cuda st_d;
    copy_table_to_vram(&st_d,st);
    vector<int> conflict_id=pivot_element_finder(st_d,st);
    copy_table_to_ram(st,&st_d);
    free_simplex_table_from_vram(st_d);

    return conflict_id;
}
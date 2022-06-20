#include"core_class.h"

using namespace std;


//modified_simplex_solver::make_solution_feasible
bool modified_simplex_solver::make_solution_feasible::termination_condition_checker(simplex_table* st)
{
    bool status=true;
    for(int a=0;a<st->r_id.size();a++)
    {
        if(st->r_id[a].slack==true)
        {
            if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>=0)
            {
                status=false;
                break;
            }
        }
        if(status==false)
        {   break;}
    }
    if(status==true)
    {
        //cout<<"\n\ntraining ended check table";
    }
    return status;
}

void modified_simplex_solver::make_solution_feasible::display_st(simplex_table st)
{
    fstream file1("simplex_table.csv",ios::out);
    file1<<",";
    for(int a=0;a<st.c_id.size();a++)
    {
        if(st.c_id[a].basic==true)
        {
            file1<<"c"<<st.c_id[a].id<<",";
        }
        else if(st.c_id[a].slack==true)
        {
            file1<<"s"<<st.c_id[a].id<<",";
        }
        else if(st.c_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st.c_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }
        else if(st.c_id[a].theta==true)
        {
            file1<<"theta,";
        }
    }
    file1<<"\n";
    for(int a=0;a<st.r_id.size();a++)
    {
        if(st.r_id[a].basic==true)
        {
            file1<<"c"<<st.r_id[a].id<<",";
        }
        else if(st.r_id[a].slack==true)
        {
            file1<<"s"<<st.r_id[a].id<<",";
        }
        else if(st.r_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st.r_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }


        for(int c=0;c<st.basic_var[a].size();c++)
        {
            file1<<st.basic_var[a][c]<<",";
        }
        for(int c=0;c<st.slack_var[a].size();c++)
        {
            file1<<st.slack_var[a][c]<<",";
        }
        file1/*<<st.z_col[a]<<","*/<<st.rhs[a]<<","<<st.theta[a]<<",";

        file1<<"\n";
    }
    /*file1<<"z,";
    for(int a=0;a<st.z_row.size();a++)
    {
        file1<<st.z_row[a]<<",";
    }*/

    file1.close();
}

long double modified_simplex_solver::make_solution_feasible::round_to_zero(long double input)
{
    if(input<0.001 && input>-0.001)
    {   return input;}//0;}
    else
    {   return input;}
}

void modified_simplex_solver::print_message()
{
    pthread_mutex_lock(&lock_1);
    cout<<message;
    pthread_mutex_unlock(&lock_1);
}

void modified_simplex_solver::make_solution_feasible::print_message()
{
    pthread_mutex_lock(&lock_1);
    cout<<message;
    pthread_mutex_unlock(&lock_1);
}

void modified_simplex_solver::make_solution_feasible::simplex_table_modifier(int p_row_index,int p_col_index,simplex_table* st)
{
    /*if(fg==1)
    {
        display_st(*st);
        int gh;cin>>gh;
    }*/
    //row id changer
    st->r_id[p_row_index].basic=st->c_id[p_col_index].basic;
    st->r_id[p_row_index].id=st->c_id[p_col_index].id;
    st->r_id[p_row_index].rhs=st->c_id[p_col_index].rhs;
    st->r_id[p_row_index].slack=st->c_id[p_col_index].slack;
    st->r_id[p_row_index].theta=st->c_id[p_col_index].theta;
    //st->r_id[p_row_index].z=st->r_id[p_col_index].z; //logically this line is fucked up dont know how it was working earlier

    float pe;
    //pivot element extractor
    if(p_col_index<st->basic_var[0].size())
    {
        pe=st->basic_var[p_row_index][p_col_index];
        if(display_iterations==true)
        {  
            message.clear();
            message="\n\ncheck table";
            print_message();
        }
    }
    else
    {
        int slack_p_col=p_col_index-st->basic_var[0].size();
        pe=st->slack_var[p_row_index][slack_p_col];
        if(display_iterations==true)
        {
            message.clear();
            message="pe= "+to_string(pe)+"\n\n\ncheck table";
            print_message();
        }
    }

    //pivot row modifier
    for(int a=0;a<st->basic_var[p_row_index].size();a++)
    {   st->basic_var[p_row_index][a]=st->basic_var[p_row_index][a]/pe;}

    for(int a=0;a<st->slack_var[p_row_index].size();a++)
    {   st->slack_var[p_row_index][a]=st->slack_var[p_row_index][a]/pe;}

    //st->z_col[p_row_index]=st->z_col[p_row_index]/pe;
    st->rhs[p_row_index]=st->rhs[p_row_index]/((double)pe);

    //rest of the row - z_row modifier
    bool basic_point=true;
    int p_col_index_temp;
    if(p_col_index>=st->basic_var[0].size())
    {
        p_col_index_temp=p_col_index-st->basic_var[0].size();
        basic_point=false;
    }

    long double multiplying_element;
    for(int a=0;a<st->basic_var.size();a++)
    {
        //multiplying element finder
        bool p_row=false;
        if(basic_point==true)
        {
            if(a==p_row_index)
            {   continue;}
            multiplying_element=st->basic_var[a][p_col_index];
        }
        else
        {
            if(a==p_row_index)
            {   continue;}
            multiplying_element=st->slack_var[a][p_col_index_temp];
        }
        //basic_var row modification instruction
        for(int b=0;b<st->basic_var[a].size();b++)
        {
            st->basic_var[a][b]=(st->basic_var[a][b]-multiplying_element*st->basic_var[p_row_index][b]);
        }
        //slack_var modification instruction
        for(int b=0;b<st->slack_var[a].size();b++)
        {
            st->slack_var[a][b]=(st->slack_var[a][b]-multiplying_element*st->slack_var[p_row_index][b]);
        }

        //z col modification
        //st->z_col[a]=(st->z_col[a]-multiplying_element*st->z_col[p_row_index]);
        //rhs col modification instruction
        st->rhs[a]=st->rhs[a]-multiplying_element*st->rhs[p_row_index];
    }

    //z row modifier
    if(display_iterations==true)
    {  
        message.clear();
        message="\np_col_index= "+to_string(p_col_index)+" multiplying element= "+to_string(multiplying_element);
        print_message();
    }
    //multiplying_element=st->z_row[p_col_index];
    //vector<float> basic_plus_slack_plus_z_plus_rhs_temp;
    //basic_plus_slack_plus_z_plus_rhs_temp.clear();
    buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.clear();
    buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.resize(st->basic_var[p_row_index].size());//bug found memory allocation problem
    for(int b=0;b<st->basic_var[p_row_index].size();b++)
    {   
        float var1=st->basic_var[p_row_index][b];
        int s1=buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.size();
        buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp[b]=var1;
        //basic_plus_slack_plus_z_plus_rhs_temp.push_back(var1);
    }//canny data crash point
    for(int b=0;b<st->slack_var[p_row_index].size();b++)
    {   buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.push_back(st->slack_var[p_row_index][b]);}// bug found long double to float conversion
    //buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.push_back(st->z_col[p_row_index]);
    buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp.push_back(st->rhs[p_row_index]);

    /*for(int b=0;b<st->z_row.size();b++)
    {
        //cout<<"\nz_row= "<<st->z_row[b]<<" basic_s_z_r_t= "<<basic_plus_slack_plus_z_plus_rhs_temp[b];
        st->z_row[b]=(st->z_row[b]-multiplying_element*buffer_obj.basic_plus_slack_plus_z_plus_rhs_temp[b]);
    }*/
}

void modified_simplex_solver::make_solution_feasible::conflicting_data_finder(simplex_table* st)
{
    conflict_id.conflict_id_present=true;
    conflict_id.id.clear();
    for(int a=0;a<st->r_id.size();a++)
    {
        if(st->r_id[a].slack==true)
        {
            if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>0) //a bug may be present here.
            {
                conflict_id.id.push_back(a);
            }
        }
    }
    //module working tested problem present in the conflictig data detection
}

void modified_simplex_solver::make_solution_feasible::pivot_element_finder(simplex_table* st)
{
    bool conflicting_data_found=false,bad_p_row_index_status=false;
    cyclic_bug_present_var=false;
    buffer_obj.p_col_index.clear();
    buffer_obj.p_row_index.clear();
    do
    {
        iteration++;
        if(display_iterations==true)
        {
            message.clear();
            message="\niteration= "+to_string(iteration)+"r_id size= "+to_string(st->r_id.size());
            print_message();
        }
        //pivot column finder
        int p_col_index,p_row_index;
        int row_with_negative_slack;
            //row havaing negative slack finder
        bool p_col_found=false;
        int row_skip=1;
        bool row_with_negative_slack_found=false;
        for(int a=0;a<st->r_id.size();a++)
        {
            if(st->r_id[a].slack==true)
            {
                if(st->slack_var[a][st->r_id[a].id-st->basic_var[a].size()]<0 && st->rhs[a]>0) //a bug may be present here.
                {
                    row_with_negative_slack=a;
                    row_with_negative_slack_found=true;
                    break;
                }
            }
        }

        if(row_with_negative_slack_found==true)
        {
            if(display_iterations==true)
            {   
                message.clear();
                message="\nrow_with_negative_slack= "+to_string(row_with_negative_slack);
                print_message();
            }
                //pivot column finding process
                    //search in basic var vector

            for(int a=0;a<st->basic_var[row_with_negative_slack].size();a++)
            {
                if(st->basic_var[row_with_negative_slack][a]>0)
                {
                    p_col_index=a;
                    p_col_found=true;
                    break;
                }
            }
                    //search in slack var vector
            if(p_col_found==false)
            {
                for(int a=0;a<st->slack_var[row_with_negative_slack].size();a++)
                {
                    if(st->slack_var[row_with_negative_slack][a]>0)
                    {
                        p_col_index=a+st->basic_var[row_with_negative_slack].size();
                        p_col_found=true;
                        break;
                    }
                }
            }

            if(p_col_found==false)
            {
                conflicting_data_finder(st);
                conflicting_data_found=true;
            }
            else
            {
                conflict_id.id.clear();
                conflict_id.conflict_id_present=false;
                st->theta.clear();
                for(int a=0;a<st->r_id.size();a++)
                {
                    if(p_col_index<st->basic_var[a].size())
                    {
                        if(st->basic_var[a][p_col_index]==0)
                        {
                            long double var1=0;
                            st->theta.push_back(var1);
                        }
                        else
                        {
                            long double rhs=st->rhs[a],basic_var=st->basic_var[a][p_col_index];//if long double is not used than precision problem may occur which may result in looping which even bland's rule fails to solve
                            long double calc=rhs/basic_var;
                            st->theta.push_back(calc);
                        }
                    }
                    else
                    {
                        int temp_col_index=p_col_index-st->basic_var[a].size();
                        if(st->slack_var[a][temp_col_index]==0)
                        {
                            st->theta.push_back(0);
                        }
                        else
                        {
                            long double rhs=st->rhs[a],slack_var=st->slack_var[a][temp_col_index];
                            long double calc=rhs/slack_var;
                            st->theta.push_back(calc);
                        }
                    }
                }
                //pivot row finder
                //vector<long double> sorted_theta;
                buffer_obj.sorted_theta.clear();
                buffer_obj.sorted_theta.insert(buffer_obj.sorted_theta.begin(),st->theta.begin(),st->theta.end());
                //sorted_theta=st->theta;
                sort(buffer_obj.sorted_theta.begin(),buffer_obj.sorted_theta.end());
                long double smallest_positive_theta;
                for(int a=0;a<buffer_obj.sorted_theta.size();a++)
                {
                    if(buffer_obj.sorted_theta[a]>0)
                    {   smallest_positive_theta=buffer_obj.sorted_theta[a];
                        break;
                    }
                }
                bool bad_p_row_index=true;
                for(int a=0;a<st->theta.size();a++)
                {
                    if(st->theta[a]==smallest_positive_theta)
                    {
                        p_row_index=a;bad_p_row_index=false;
                        break;
                    }
                }
                if(display_iterations==true)
                { 
                    message.clear();
                    message="\nsmallest positive theta= "+to_string(smallest_positive_theta)+"\npivot row= "+to_string(p_row_index)+" pivot column= "+to_string(p_col_index);
                    print_message();
                }
                if(bad_p_row_index==false)
                {   
                    if(add_index_data(p_col_index,p_row_index)==false)
                    {   simplex_table_modifier(p_row_index,p_col_index,st);}
                    else
                    {   
                        cyclic_bug_present_var=true;
                        break;
                    }
                }
                else if(bad_p_row_index==true)
                {
                    //bad_p_row_index handler
                    bad_p_row_index_status=true;
                }        
            }
        }
        else
        {   break;}
    }
    while(termination_condition_checker(st)==false && conflicting_data_found==false && bad_p_row_index_status==false);
}

bool modified_simplex_solver::make_solution_feasible::add_index_data(int p_col,int p_row)
{
    if(buffer_obj.p_col_index.size()<4 && buffer_obj.p_row_index.size()<4)//10
    {
        buffer_obj.p_col_index.push_back(p_col);
        buffer_obj.p_row_index.push_back(p_row);
        return false;
    }
    else
    {
        bool status=false;
        for(int a=0;a<buffer_obj.p_row_index.size();a++)
        {
            if(buffer_obj.p_row_index[a]==p_row && buffer_obj.p_col_index[a]==p_col)
            {   status=true;}
        }
        if(status==true)
        {   return status;}
        else
        {
            buffer_obj.p_col_index.push_back(p_col);
            buffer_obj.p_row_index.push_back(p_row);
            buffer_obj.p_col_index.erase(buffer_obj.p_col_index.begin());
            buffer_obj.p_row_index.erase(buffer_obj.p_row_index.begin());
            return false;
        }   
    }
}

bool modified_simplex_solver::make_solution_feasible::cyclic_bug_present()
{   return cyclic_bug_present_var;}

conflicting_data_id modified_simplex_solver::make_solution_feasible::return_conflict_id_pack()
{   return conflict_id;}

void modified_simplex_solver::make_solution_feasible::start(simplex_table* st)
{   pivot_element_finder(st);}

//modified_simplex_solver
void modified_simplex_solver::display_st(simplex_table st)
{
    fstream file1("simplex_table.csv",ios::out);
    file1<<",";
    for(int a=0;a<st.c_id.size();a++)
    {
        if(st.c_id[a].basic==true)
        {
            file1<<"c"<<st.c_id[a].id<<",";
        }
        else if(st.c_id[a].slack==true)
        {
            file1<<"s"<<st.c_id[a].id<<",";
        }
        else if(st.c_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st.c_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }
        else if(st.c_id[a].theta==true)
        {
            file1<<"theta,";
        }
    }
    file1<<"\n";
    for(int a=0;a<st.r_id.size();a++)
    {
        if(st.r_id[a].basic==true)
        {
            file1<<"c"<<st.r_id[a].id<<",";
        }
        else if(st.r_id[a].slack==true)
        {
            file1<<"s"<<st.r_id[a].id<<",";
        }
        else if(st.r_id[a].z==true)
        {
            file1<<"z"<<",";
        }
        else if(st.r_id[a].rhs==true)
        {
            file1<<"rhs"<<",";
        }


        for(int c=0;c<st.basic_var[a].size();c++)
        {
            file1<<st.basic_var[a][c]<<",";
        }
        for(int c=0;c<st.slack_var[a].size();c++)
        {
            file1<<st.slack_var[a][c]<<",";
        }
        file1/*<<st.z_col[a]<<","*/<<st.rhs[a]<<","<<st.theta[a]<<",";

        file1<<"\n";
    }
    /*file1<<"z,";
    for(int a=0;a<st.z_row.size();a++)
    {
        file1<<st.z_row[a]<<",";
    }*/

    file1.close();
}

bool modified_simplex_solver::check_for_corrupt_cdp(converted_data_pack* cdp)
{
    try{
        int firing_crash_size=cdp->firing_data.size();
        int not_firing_data=cdp->not_firing_data.size();
        int firing_crash=cdp->firing_data.at(0).size();
        int not_firing_crash=cdp->not_firing_data.at(0).size();
        return false;
    }
    catch(exception &e)
    {   return true;}
}

bool modified_simplex_solver::start_solver(converted_data_pack* cdp)
{
    if(display_iterations==true)
    {   
        message.clear();
        message="\n\ncdp->firing_data.size()= "+to_string(cdp->firing_data.size());
        print_message();
    }
    id temp_id;
    simplex_table* st=new simplex_table();
    st->c_id.clear();
    bool corrupt_cdp=check_for_corrupt_cdp(cdp);
    if(corrupt_cdp==true)
    {
        cout<<"\ngot it!!!!";
        cdp->corupt_pack=true;
        delete st;
        cout<<"\nlower_rhs= "<<lower_firing_constrain_rhs;
        cout<<"\nupper rhs= "<<upper_not_firing_constrain_rhs;
        cout<<"\nfiring_data_size= "<<cdp->firing_data.size();
        cout<<"\nnot_firing_size= "<<cdp->not_firing_data.size();
        return false;
    }
    else if(corrupt_cdp==false)
    {   
        for(int a=0;a<cdp->firing_data[0].size()*2;a++)
        {
            temp_id.basic=true;
            temp_id.slack=false;
            temp_id.z=false;
            temp_id.rhs=false;
            temp_id.theta=false;
            temp_id.id=a;
            st->c_id.push_back(temp_id);
        }
        int slack_id=cdp->firing_data[0].size()*2;
        for(int a=cdp->firing_data[0].size();a<(cdp->firing_data[0].size()+cdp->firing_data.size()+cdp->not_firing_data.size());a++)
        {
            temp_id.slack=true;
            temp_id.basic=false;
            temp_id.z=false;
            temp_id.theta=false;
            temp_id.rhs=false;
            temp_id.id=slack_id;
            st->c_id.push_back(temp_id);
            slack_id++;
        }

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=false;
        temp_id.z=true;
        temp_id.theta=false;
        temp_id.id=slack_id;
        st->c_id.push_back(temp_id);
        slack_id++;

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=true;
        temp_id.z=false;
        temp_id.theta=false;
        temp_id.id=slack_id;
        st->c_id.push_back(temp_id);

        temp_id.slack=false;
        temp_id.basic=false;
        temp_id.rhs=false;
        temp_id.z=false;
        temp_id.theta=true;
        temp_id.id=slack_id;
        st->c_id.push_back(temp_id);

        st->r_id.clear();
        for(int a=0;a<st->c_id.size();a++)
        {
            if(st->c_id[a].slack==true)
            {
                st->r_id.push_back(st->c_id[a]);
            }
        }

        st->slack_var.clear();
        st->basic_var.clear();
        vector<float> temp;
        int x;
        for(int a=0;a<cdp->firing_data.size();a++)
        {
            temp.clear();
            //entering basic variable data
            for(int b=0;b<cdp->firing_data[a].size();b++)
            {   temp.push_back(cdp->firing_data[a][b]);
                temp.push_back(cdp->firing_data[a][b]-2*cdp->firing_data[a][b]);
            }
            st->basic_var.push_back(temp);
            temp.clear();
            //entering slack var data
            for(int b=0;b<st->r_id.size();b++)
            {
                if(b==a)
                {   temp.push_back(-1);x=b;}
                else
                {   temp.push_back(0);}
            }
            //temp.push_back(0);//for z
            //temp.push_back(cdp->lower_firing_constrain_rhs);
            st->slack_var.push_back(temp);

            //st->z_col.push_back(0);
            st->rhs.push_back(lower_firing_constrain_rhs); //modification needs to be done here
        }
        for(int a=0;a<cdp->not_firing_data.size();a++)
        {
            temp.clear();
            //entering basic variable data
            for(int b=0;b<cdp->not_firing_data[a].size();b++)
            {   temp.push_back(cdp->not_firing_data[a][b]);
                temp.push_back(cdp->not_firing_data[a][b]-2*cdp->not_firing_data[a][b]);
            }
            st->basic_var.push_back(temp);
            temp.clear();
            //entering slack var data
            for(int b=0;b<st->r_id.size();b++)
            {
                if((b)==(a+x+1))
                {   temp.push_back(1);}
                else
                {   temp.push_back(0);}
            }
            //temp.push_back(0);//for z
            //temp.push_back(cdp->upper_not_firing_constrain_rhs);
            st->slack_var.push_back(temp);

            //st->z_col.push_back(0);
            st->rhs.push_back(upper_not_firing_constrain_rhs); //modification needs to be done here
        }
        /*
        st->z_row.clear();
        //entering z row data
        for(int a=0;a<cdp->objective_function_coefficients.size();a++)
        {
            st->z_row.push_back(cdp->objective_function_coefficients[a]);
            st->z_row.push_back(cdp->objective_function_coefficients[a]-2*cdp->objective_function_coefficients[a]);
        }
        for(int a=0;a<st->r_id.size();a++)
        {   st->z_row.push_back(0);}
        st->z_row.push_back(1);// co_orodianted (z,z)
        st->z_row.push_back(0);//(z,rhs)
        /*/
        //needs modification
        feasible_solution_calculator.start(st);
        //cout<<"data size2= "<<cdp->firing_data.size()<<endl;
        conflicting_data_id conflict_id;
        conflict_id=feasible_solution_calculator.return_conflict_id_pack();

        if(conflict_id.conflict_id_present==true)
        {
            conflicting_data.firing_data.clear();
            conflicting_data.not_firing_data.clear();
            //conflicting_data.objective_function_coefficients.clear();
            conflicting_data.weight_matrix.clear();
            if(display_iterations==true)
            {
                message.clear();
                message="\nconflicting databefore erasing:\n firing data size= "+to_string(cdp->firing_data.size())+"\n";
                print_message();
            }
            if(conflict_id.id.size()==cdp->firing_data.size() && cdp->firing_data.size()==1)
            {
                //cout<<"\nbingo!!";
                cdp->corupt_pack=true;
            }
            else if(conflict_id.id.size()==cdp->firing_data.size())//for handling 0:0 bug
            {   
                int firing_data_limit=cdp->firing_data.size()/2;
                int not_firing_data_limit=cdp->not_firing_data.size()/2;
                converted_data_pack cdp_temp;
                for(int a=cdp->firing_data.size()-1;a>=firing_data_limit;a--)//saving the 2nd half firing_data in the conflicting_data
                {
                    conflicting_data.firing_data.push_back(cdp->firing_data[a]);
                    cdp->firing_data.erase(cdp->firing_data.begin()+a);
                }
                for(int a=cdp->not_firing_data.size()-1;a>=not_firing_data_limit;a--)
                {
                    conflicting_data.not_firing_data.push_back(cdp->not_firing_data[a]);
                    cdp->not_firing_data.erase(cdp->not_firing_data.begin()+a);
                }
                conflicting_data.firing_label=cdp->firing_label;
                conflicting_data.firing_neuron_index=cdp->firing_neuron_index;
                if(check_for_corrupt_cdp(cdp)==false)
                start_solver(cdp);
                else
                cout<<"\ncheck2";
            }
            else
            {
                int count1=0;
                for(int a=conflict_id.id.size()-1;a>=0;a--)
                {
                    //save the conflicting data in a obj of converted data pack.
                    conflicting_data.firing_data.push_back(cdp->firing_data[conflict_id.id[a]]); //copying the conflicting data
                    if(display_iterations==true)
                    {   
                        message.clear();
                        message="erasing id= "+to_string(conflict_id.id[a])+",";
                        print_message();
                    }
                    cdp->firing_data.erase(cdp->firing_data.begin()+conflict_id.id[a]); //erasing the conflicting data
                    if(display_iterations==true)
                    {   
                        message.clear();
                        message="firing data size= "+to_string(cdp->firing_data.size())+"\n";
                        print_message();
                    }
                    count1++;
                }
                //ratio maintainer
                int current_not_firing_data_size=cdp->not_firing_data.size();
                int current_firing_data_size=cdp->firing_data.size();
                float no_of_notfiring_data_to_be_removed_float=(current_not_firing_data_size-current_firing_data_size);
                int no_of_notfiring_data_to_be_removed=no_of_notfiring_data_to_be_removed_float;
                    //not firing data are veing removed from cdp and added to conflicting_data
                if(display_iterations==true)//left_out
                {
                    cout<<"\nno of not_firing_data to be removed= "<<no_of_notfiring_data_to_be_removed<<endl;
                    cout<<"current not_firing_data size= "<<cdp->not_firing_data.size()<<endl;
                    cout<<"firing_data_size= "<<cdp->firing_data.size()<<endl;
                    cout<<"no of firing data removed= "<<count1<<endl;
                    cout<<"cdp->firing_neuron_index= "<<cdp->firing_neuron_index<<endl;
                }
                for(int a=current_not_firing_data_size-1;a>=(current_not_firing_data_size-no_of_notfiring_data_to_be_removed);a--)
                {
                    conflicting_data.not_firing_data.push_back(cdp->not_firing_data[a]);
                    cdp->not_firing_data.erase(cdp->not_firing_data.begin()+a);
                }
                conflicting_data.firing_neuron_index=cdp->firing_neuron_index; //this line is important 
                conflicting_data.firing_label=cdp->firing_label; //this line is important
                if(display_iterations==true)
                {
                    cout<<"\nafter erasing\n";
                    cout<<"firing_size= "<<cdp->firing_data.size()<<" not_firing_size= "<<cdp->not_firing_data.size()<<endl;
                }
                if(check_for_corrupt_cdp(cdp)==false)
                start_solver(cdp); //calling start_solver function with non conflicting data.
                else
                cout<<"\ncheck1";
            }
        }
        else
        {
            cdp->weight_matrix.clear();
            vector<double> weight_matrix_raw;
            weight_matrix_raw.clear();
            //push the raw solution in the converted data pack.
            for(int a=0;a<cdp->firing_data[0].size()*2;a++)
            {
                bool found=false;
                for(int b=0;b<st->r_id.size();b++)
                {
                    if(st->r_id[b].id==a && st->r_id[b].basic==true)
                    {
                        //cdp->weight_matrix.push_back(st.rhs[b]*st.basic_var[b][a]);
                        weight_matrix_raw.push_back(st->rhs[b]*st->basic_var[b][a]);
                        found=true;
                        break;
                    }
                }
                if(found==false)
                {
                    //cdp->weight_matrix.push_back(0);
                    weight_matrix_raw.push_back(0);
                }
            }
            //pushing the calculated solution in the weight matrix
            for(int a=0;a<cdp->firing_data[0].size()*2;a+=2)
            {
                cdp->weight_matrix.push_back(weight_matrix_raw[a]-weight_matrix_raw[a+1]);
            }
            //as cdp is an address of the original cdp obj is already present in the previous function.
        }
        delete st;
        return true;
    }
    else//this is for avoiding compilation warning
    {   return false;}
}

bool modified_simplex_solver::cyclic_bug_present()
{   return feasible_solution_calculator.cyclic_bug_present();}

void modified_simplex_solver::set_training_settings(float lower_firing_constrain_rhs1,float upper_not_firing_constrain_rhs1)
{
    lower_firing_constrain_rhs=lower_firing_constrain_rhs1; //60,150
    upper_not_firing_constrain_rhs=upper_not_firing_constrain_rhs1; //10
}


//simplex_solver_data_preparation_class
void simplex_solver_data_preparation_class::cdp_viewer(converted_data_pack* cdp)
{
    cout<<"firing neuron index= "<<cdp->firing_neuron_index<<"  label= "<<cdp->firing_label<<endl;
    cout<<"objective function coefficient= "<<endl;
    //for(int a=0;a<cdp->objective_function_coefficients.size();a++)
    //{   cout<<cdp->objective_function_coefficients[a]<<",";}
    cout<<"\n\nfiring data:-"<<endl<<endl;
    for(int a=0;a<cdp->firing_data.size()-100;a++)
    {
        //cout<<"a="<<a<<" size= "<<cdp->firing_data[a].size()<<endl;
        for(int b=0;b<cdp->firing_data[a].size();b++)
        {   cout<<cdp->firing_data[a][b]<<",";}
        cout<<endl;
    }
    cout<<"\n\nnot_firing data:-\n\n";
    for(int a=0;a<cdp->not_firing_data.size()-100;a++)
    {
        for(int b=0;b<cdp->not_firing_data[a].size();b++)
        {   cout<<cdp->not_firing_data[a][b]<<",";}
        cout<<endl;
    }
}

void simplex_solver_data_preparation_class::cdp_saver_in_mathematical_format(converted_data_pack* cdp)
{
    fstream file1("question.txt",ios::out);

    /*for(int a=0;a<cdp->objective_function_coefficients.size();a++)
    {
        file1<<cdp->objective_function_coefficients[a]<<"c"<<a+1<<"+";
        cout<<cdp->objective_function_coefficients[a]<<",";
    }*/
    file1<<"\n\n\n";
    for(int a=0;a<cdp->firing_data.size();a++)
    {
        file1<<"\n";
        for(int b=0;b<cdp->firing_data[a].size();b++)
        {
            file1<<cdp->firing_data[a][b]<<"c"<<b+1;
            if(b==cdp->firing_data[a].size()-1)
            {   continue;}
            file1<<"+";
        }
        file1<<">40";
    }
    file1<<"\n\n";
    file1.close();
    ofstream file2("question2.txt",ios::out);
    file2<<"\n\n";
    cout<<" size= "<<cdp->not_firing_data.size();
    for(int a=0;a<cdp->not_firing_data.size();a++)
    {
        file2<<"\n";
        for(int b=0;b<cdp->not_firing_data[a].size();b++)
        {
            file2<<cdp->not_firing_data[a][b]<<"c"<<b+1;
            //cout<<cdp->not_firing_data[a][b]<<"c"<<b+1<<endl;
            if(b==cdp->not_firing_data[a].size()-1)
            {   continue;}
            file2<<"+";
        }
        //cout<<a<<" ";
        file2<<"<10";
    }
}

bool simplex_solver_data_preparation_class::cyclic_bug_present()
{   return lpp_solver1.cyclic_bug_present();}

void simplex_solver_data_preparation_class::cdp_spliter(vector<converted_data_pack> &cdps,int index)
{
    //rooms
    converted_data_pack cdp_temp1=converted_data_pack();
    converted_data_pack cdp_temp2=converted_data_pack();
    //spliting process
    int a=0,b=0;
        //first half
    for(a;a<cdps[index].firing_data.size()/2;a++)
    {   cdp_temp1.firing_data.push_back(cdps[index].firing_data[a]);}
    for(b;b<cdps[index].not_firing_data.size()/2;b++)
    {   cdp_temp1.not_firing_data.push_back(cdps[index].not_firing_data[b]);}
    cdp_temp1.corupt_pack=cdps[index].corupt_pack;
    cdp_temp1.firing_label=cdps[index].firing_label;
    cdp_temp1.firing_neuron_index=cdps[index].firing_neuron_index;
    //cdp_temp1.objective_function_coefficients=cdps[index].objective_function_coefficients;
    cdp_temp1.weight_matrix=cdps[index].weight_matrix;
        //second half
    for(a=cdps[index].firing_data.size()/2;a<cdps[index].firing_data.size();a++)
    {   cdp_temp2.firing_data.push_back(cdps[index].firing_data[a]);}
    for(b=cdps[index].not_firing_data.size()/2;b<cdps[index].not_firing_data.size();b++)
    {   cdp_temp2.not_firing_data.push_back(cdps[index].not_firing_data[b]);}
    cdp_temp2.corupt_pack=cdps[index].corupt_pack;
    cdp_temp2.firing_label=cdps[index].firing_label;
    cdp_temp2.firing_neuron_index=cdps[index].firing_neuron_index;
    //cdp_temp2.objective_function_coefficients=cdps[index].objective_function_coefficients;
    cdp_temp2.weight_matrix=cdps[index].weight_matrix;
    converted_data_pack temp;
    temp=cdps[index];
    cdps[index]=cdp_temp1;
    temp.firing_data.clear();
    temp.not_firing_data.clear();
    //temp.objective_function_coefficients.clear();
    temp.weight_matrix.clear();
    cdps.push_back(cdp_temp2);
}

void simplex_solver_data_preparation_class::lp_solver()
{
    //the main conflicting_data_outer cleaner
    conflicting_data_buffer_outer.conflicting_data_buffer_vector.clear();
    int f_size=cdp[0].firing_data.size();//memory_optimization7
    int nf_size=cdp[0].not_firing_data.size();//memory_optimization7
    for(int a=0;a<cdp.size();a++)
    {
        point1:
        //cdp_spliter(cdp,a);
        if(cdp[a].firing_data.size()!=0 && cdp[a].not_firing_data.size()!=0)
        {   lpp_solver1.start_solver(&cdp[a]);}
        else
        {   continue;}
        //lpp_solver1.start_solver(cdp[a]);
        if(lpp_solver1.cyclic_bug_present()==true)
        {
            message.clear();
            message="\ncyclic bug detected...fixing...";
            print_message();
            cdp_spliter(cdp,a);
            goto point1;
        }
        if(display_iterations==true)
        {
            cout<<"\nfiring size= "<<cdp[a].firing_data.size()<<"\nnot firing size= "<<cdp[a].not_firing_data.size()<<"\n";
            cout<<"weight matrix: ";
            for(int b=0;b<cdp[a].weight_matrix.size();b++)
            {
                cout<<cdp[a].weight_matrix[b]<<",";
            }
            cout<<endl;
        }
        //adding the conflicting data to the conflicting data buffer
        //getting the conflicting data from the lpp_solver1 and adding them to the conflicting_data_buffer.
        conflicting_data_buffer.firing_data.insert(conflicting_data_buffer.firing_data.end(),lpp_solver1.conflicting_data.firing_data.begin(),lpp_solver1.conflicting_data.firing_data.end());
        conflicting_data_buffer.not_firing_data.insert(conflicting_data_buffer.not_firing_data.end(),lpp_solver1.conflicting_data.not_firing_data.begin(),lpp_solver1.conflicting_data.not_firing_data.end());
        conflicting_data_buffer.firing_neuron_index=lpp_solver1.conflicting_data.firing_neuron_index; //this line is important //the data here is directily brought from the cdp
        conflicting_data_buffer.firing_label=lpp_solver1.conflicting_data.firing_label; //this line is important //the data here is directily brought from the cdp
        //claner of conflicting_data_buffer_outer.conflicting_data_buffer_vector
        //erasing the conflicting data from the lpp_solver1
        lpp_solver1.conflicting_data.firing_data.clear();
        lpp_solver1.conflicting_data.not_firing_data.clear();
        //lpp_solver1.conflicting_data.objective_function_coefficients.clear();
        lpp_solver1.conflicting_data.weight_matrix.clear();

        //lp_optimizer may be added here.
        //handling each of the weight matrix is required.

        // 1. create new path
        //   1a. provide the output neuron id. (training for which output neuron or which label. First neuron = label 2 second neuron = label 4)
        //   2b. push the weight matrix
        if(cdp[a].corupt_pack==false)
        {   
            pthread_mutex_lock(&lock);
            network->create_new_path(cdp[a].weight_matrix,cdp[a].firing_neuron_index);
            pthread_mutex_unlock(&lock);
        }
        //enter the label of the conflicting data //I think this process is already complete...... 
        
        //pushing the conflicting_data_buffer of each label to the conflicting_data_buffer_vector.
        if(conflicting_data_buffer.firing_data.size()>0 && conflicting_data_buffer.not_firing_data.size()>0)
        {   conflicting_data_buffer_outer.conflicting_data_buffer_vector.push_back(conflicting_data_buffer);}
        
        //cleanup
        conflicting_data_buffer.weight_matrix.clear();
        //conflicting_data_buffer.objective_function_coefficients.clear();
        conflicting_data_buffer.firing_data.clear();
        conflicting_data_buffer.not_firing_data.clear();
    }
    // 2. conflicting datas must be handled here.

    if(conflicting_data_buffer_outer.conflicting_data_buffer_vector.size()>0)
    {
        //objective function coefficient calculation
        /*int d=0;
        for(int a=0;a<conflicting_data_buffer_outer.conflicting_data_buffer_vector.size();a++)
        {
            conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients.resize(data_structure->no_of_elements_in_each_record);
            for(int b=0;b<data_structure->no_of_elements_in_each_record;b++)
            {
                for(d=0;d<conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data.size();d++){
                    conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]+=conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data[d][b];
                }
                conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]=conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].objective_function_coefficients[b]/d;
            }
        }*/
        for(int a=0;a<conflicting_data_buffer_outer.conflicting_data_buffer_vector.size();a++)
        {
            if(conflicting_data_buffer_outer.conflicting_data_buffer_vector[a].firing_data.size()==0)
            {
                conflicting_data_buffer_outer.conflicting_data_buffer_vector.erase(conflicting_data_buffer_outer.conflicting_data_buffer_vector.begin()+a);
            }
        }
        message.clear();
        message="\nsolving for the conflicting data.....\n";
        print_message();
        cdp.clear();//memory_optimization7
        vector<converted_data_pack> cdp_temp;
        cdp_temp=conflicting_data_buffer_outer.conflicting_data_buffer_vector;
        for(int a=0;a<cdp_temp.size();a++)
        {   cdp.push_back(cdp_temp[a]);}
        //cdp=conflicting_data_buffer_outer.conflicting_data_buffer_vector;
        lp_solver();
    }
    else
    {   
        message.clear();
        message="\nfinished training the entire package.\n";
        print_message();
    }
}

void simplex_solver_data_preparation_class::print_message()
{
    if(display_core_events)
    {
        pthread_mutex_lock(&lock_1);
        cout<<message;
        pthread_mutex_unlock(&lock_1);
    }
}

simplex_solver_data_preparation_class::simplex_solver_data_preparation_class(vector<converted_data_pack> &cdps,datapack_structure_defination* ds,ann* network1)
{
    network=network1;
    data_structure=ds;
    lpp_solver1.set_training_settings(ds->lower_firing_constrain_rhs,ds->upper_not_firing_constrain_rhs); 
    cdp=cdps;
}

//core_class

vector<neuron> core_class::propagate(vector<float> input_attributes_value)
{
    network1.enter_data_in_the_network(input_attributes_value);
    return network1.propagate();
}

void core_class::save_core(string folder_dir)
{
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);
    int day = aTime->tm_mday;
    string day_str= to_string(day);
    if(day<10)
    {   day_str="0"+day_str;}
    int month = aTime->tm_mon + 1; // Month is 0 - 11, add 1 to get a jan-dec 1-12 concept
    string month_str=to_string(month);
    if(month<10)
    {   month_str="0"+month_str;}
    int year = aTime->tm_year + 1900; // Year is # years since 1900
    string year_str=to_string(year);
    time_t now = time(0);
    // Convert now to tm struct for local timezone
    tm* localtm = localtime(&now);
    int hr,min,sec;
    hr=localtm->tm_hour;
    min=localtm->tm_min;
    sec=localtm->tm_sec;
    string hr_str=to_string(hr),min_str=to_string(min),sec_str=to_string(sec);
    if(hr<10)
    {   hr_str="0"+hr_str;}
    if(min<10)
    {   min_str="0"+min_str;}
    if(sec<10)
    {   sec_str="0"+sec_str;}
    string core_savefile_id="";//core_aim,core_no,day,month,year
    string core_aim_str=to_string(core_aim);
    string core_no_str=to_string(core_no);
    core_savefile_id=core_aim_str+core_no_str+year_str+month_str+day_str+hr_str+min_str+sec_str;
    core_save_file_name="core-"+core_savefile_id+".csv";
    ofstream file1(folder_dir+core_save_file_name,ios::out);
    file1<<"FILE_NAME:,\n";
    file1<<"name=,"<<core_save_file_name<<",\n";
    file1<<"BASIC_SAVEFILE_INFO:,\n";
    file1<<",core_save_id,core_aim,core_no,year,month,day,hour,minute,sec,\n";
    file1<<"core_save_id,"<<core_savefile_id<<","<<core_aim<<","<<core_no<<","<<year_str<<","<<month_str<<","<<day_str<<","<<hr_str<<","<<min_str<<","<<sec_str<<",\n";
    file1<<"BASIC_NETWORK_INFO:,";
    file1<<"net_info_category,no_of_input_neuron,no_of_output_neuron,\n";
    file1<<",net_info_category,"<<network1.input_neuron_size()<<","<<network1.output_neuron_size()<<",\n";
    file1<<"DATA_LABEL_AND_OUTPUT_NEURON_INDEX:,\n";
    file1<<"[data_label~output_neuron_index],";
    for(int a=0;a<ds.elements.size();a++)
    {
        file1<<ds.elements[a]<<"~"<<a<<",";
    }
    file1<<"\n";
    file1<<"PATH_INFO:,\n";
    file1<<"no_of_path=,"<<network1.return_no_of_paths()<<",\n";
    file1<<"path_id,";
    for(int a=0;a<network1.path[0].weight_matrix.size();a++)
    {
        int weight_index=a;
        string weight_index_str=to_string(weight_index);
        weight_index_str="w"+weight_index_str;
        file1<<weight_index_str<<",";
    }
    file1<<"input_neuron_id,output_neuron_id\n";
    for(int a=0;a<network1.path.size();a++)
    {
        file1<<network1.path[a].path_id<<",";
        for(int b=0;b<network1.path[a].weight_matrix.size();b++)
        {
            file1<<network1.path[a].weight_matrix[b]<<",";
        }
        file1<<"[|";
        for(int b=0;b<network1.path[a].input_neuron_id.size();b++)
        {
            file1<<network1.path[a].input_neuron_id[b]<<"|";
        }
        file1<<"],";
        file1<<network1.path[a].output_neuron_id<<",\n";
    }
    file1.close();
}

vector<string> core_class::line_breaker(string line)
{
    vector<string> elements;
    string word="";
    for(int a=0;a<line.size();a++)
    {
        if(line.at(a)==',')
        {
            elements.push_back(word);
            word="";
            continue;
        }
        word+=(line.at(a));
    }
    return elements;
}
bool core_class::load_core(string core_file_dir)
{
    ifstream file1(core_file_dir,ios::in);
    string line="";
    int line_count=0;
    vector<string> elements;
    int no_of_path=10;
    try{
        while(file1)
        {
            file1>>line;
            if(line_count==1)
            {
                elements=line_breaker(line);
                core_save_file_name=elements[1];
            }
            else if(line_count==4)
            {
                elements=line_breaker(line);
                core_aim=stoi(elements[2]);
                core_no=stoi(elements[3]);
            }
            else if(line_count==6)
            {
                elements=line_breaker(line);
                network1.set_no_of_input_neuron(stoi(elements[2]));
                network1.set_no_of_output_neuron(stoi(elements[3]));
            }
            else if(line_count==8)
            {
                elements=line_breaker(line);
                ds.elements.resize(elements.size()-1);
                string data_label_str="",output_neuron_index_str="";
                for(int a=1;a<elements.size();a++)
                {
                    int symbol_index;
                    for(int b=0;b<elements[a].size();b++)
                    {
                        if(elements[a][b]=='~')
                        {   symbol_index=b;break;}
                    }
                    data_label_str="";
                    data_label_str.insert(data_label_str.begin(),elements[a].begin(),elements[a].begin()+symbol_index);
                    output_neuron_index_str="";
                    output_neuron_index_str.insert(output_neuron_index_str.begin(),elements[a].begin()+symbol_index+1,elements[a].end());
                    ds.elements[stoi(output_neuron_index_str)]=stoi(data_label_str);
                }
            }
            else if(line_count==10)
            {
                elements=line_breaker(line);
                no_of_path=stoi(elements[1]);
            }
            else if(line_count>11)
            {
                elements=line_breaker(line);
                path_struct path;
                path.output_neuron_id;
                for(int a=0;a<network1.input_neuron_size();a++)
                {   
                    path.input_neuron_id.push_back(a);
                    path.weight_matrix.push_back(stof(elements[a+1]));
                }
                path.output_neuron_id=stoi(elements[elements.size()-1]);
                network1.path.push_back(path);
            }
            line_count++;
            if(network1.path.size()==no_of_path)
            {   break;}
        }
    }
    catch(exception q)
    {   
        file1.close();
        return false;
    }
    file1.close();
    network_analyzer();//initialization of ns
    message="\nCore "+to_string(core_no)+" Loaded Successfully...";
    cout<<message;
    return true;
}

void core_class::network_analyzer()
{
    ns.no_of_input_neuron=network1.input_neuron_size();
    ns.no_of_output_neuron=network1.output_neuron_size();
}

void core_class::big_c_datapack_handler(vector<converted_data_pack> &cdp)//passing the vector by reference //this function might be a temporary offer //this is for preventing 0:0 bug
{
    int limit=50;//50
    converted_data_pack cdp_temp1,cdp_temp2;
    vector<converted_data_pack> cdp_vect_temp;
    for(int a=cdp.size()-1;a>=0;a--)
    {
        if(cdp[a].firing_data.size()>limit)
        {
            cdp_temp1=cdp[a];
            cdp.erase(cdp.begin()+a);
            int begin=0,end=0;
            bool end_reached=false;
            while(end_reached==false)
            {
                cdp_temp2.firing_data.clear();
                cdp_temp2.not_firing_data.clear();
                //cdp_temp2.objective_function_coefficients.clear();
                begin=end;
                end=begin+limit;
                if(end>=cdp_temp1.firing_data.size())
                {   end=cdp_temp1.firing_data.size();end_reached=true;}
                cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+begin,cdp_temp1.firing_data.begin()+end);
                if(end_reached==true)
                {   end=cdp_temp1.not_firing_data.size();}
                else if(end>=cdp_temp1.not_firing_data.size())
                {
                    int end2=cdp_temp1.firing_data.size();
                    cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+end,cdp_temp1.firing_data.begin()+end2);
                    end=cdp_temp1.not_firing_data.size();
                    end_reached=true;
                }
                cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+begin,cdp_temp1.not_firing_data.begin()+end);
                cdp_temp2.firing_label=cdp_temp1.firing_label;
                cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                //cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                cdp_vect_temp.push_back(cdp_temp2);
            }
            cdp_temp1.firing_data.clear();
            cdp_temp1.not_firing_data.clear();
            //cdp_temp1.objective_function_coefficients.clear();
        }
    }
    cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
    message.clear();
    message="\n\ncdp size after erasing in big data handler = "+to_string(cdp.size());
    print_message();
    message.clear();
    message=", cdp_vect_temp size= "+to_string(cdp_vect_temp.size());
    print_message();
    cdp_vect_temp.clear();
    
    for(int a=0;a<cdp.size();a++)
    {
        int difference=cdp[a].firing_data.size()-cdp[a].not_firing_data.size();
        if(abs(difference)>10)
        {
            cdp_temp1.firing_data.clear();
            cdp_temp1.not_firing_data.clear();
            //cdp_temp1.objective_function_coefficients.clear();
            cdp_temp2.firing_data.clear();
            cdp_temp2.not_firing_data.clear();
            //cdp_temp2.objective_function_coefficients.clear();
            cdp_temp1=cdp[a];
            cdp.erase(cdp.begin()+a);
            while(abs(difference)>10)
            {
                if(difference<0)
                {
                    limit=cdp_temp1.firing_data.size();
                    cdp_temp2.firing_data=cdp_temp1.firing_data;
                    cdp_temp2.not_firing_data.insert(cdp_temp2.not_firing_data.end(),cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                    cdp_temp2.firing_label=cdp_temp1.firing_label;
                    cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                    //cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                    cdp_temp1.not_firing_data.erase(cdp_temp1.not_firing_data.begin()+abs(difference),cdp_temp1.not_firing_data.end());
                    cdp_vect_temp.push_back(cdp_temp2);
                }
                else if(difference>0)
                {
                    limit=cdp_temp1.not_firing_data.size();
                    cdp_temp2.not_firing_data=cdp_temp1.not_firing_data;
                    cdp_temp2.firing_data.insert(cdp_temp2.firing_data.end(),cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                    cdp_temp2.firing_label=cdp_temp1.firing_label;
                    cdp_temp2.firing_neuron_index=cdp_temp1.firing_neuron_index;
                    //cdp_temp2.objective_function_coefficients=cdp_temp1.objective_function_coefficients;
                    cdp_temp1.firing_data.erase(cdp_temp1.firing_data.begin()+abs(difference),cdp_temp1.firing_data.end());
                    cdp_vect_temp.push_back(cdp_temp2);
                }
                difference=cdp_temp1.firing_data.size()-cdp_temp1.not_firing_data.size();
                cdp_temp2.firing_data.clear();
                cdp_temp2.not_firing_data.clear();
                //cdp_temp2.objective_function_coefficients.clear();
            }
            cdp.push_back(cdp_temp1);
        }
    }
    cdp.insert(cdp.end(),cdp_vect_temp.begin(),cdp_vect_temp.end());
    message.clear();
    message="\ncdp size after stabilizing extreme ratios = "+to_string(cdp.size());
    print_message();
    message.clear();
    message=", cdp_vect_temp size= "+to_string(cdp_vect_temp.size());
    print_message();
}

void core_class::load_training_data_into_core(vector<nn_core_filtered_data>& f_data_pack1,int& no_of_threads1,int& split_start1,int& split_end1)
{
    no_of_threads=no_of_threads1;
    split_start=&split_start1;
    split_end=&split_end1;
    f_data_pack=&f_data_pack1;
    ds.no_of_elements_in_each_record=f_data_pack->at(0).data[0].size();
    network_structure_modifier();
}

void core_class::train_core()
{
    vector<converted_data_pack> c_datapacks;
    converted_data_pack c_datapack;
    //ratio maintance and packing data in c_datapacks.
    int sum_total_training_data=0;
    for(int a=0;a<f_data_pack->size();a++)
    {   sum_total_training_data=sum_total_training_data+f_data_pack->at(a).data.size();}
    message.clear();
    message="\nsize of training data set= "+to_string(sum_total_training_data)+"\n";
    print_message();
    c_datapacks.clear(); //for asured cleaniness
    for(int a=0;a<f_data_pack->size();a++)
    {
        message.clear();
        message="packing data for label= "+to_string(f_data_pack->at(a).label)+"\n";
        print_message();
        //determining the c_data_pack critical info
        int sum_total_not_firing_data=sum_total_training_data-f_data_pack->at(a).data.size();
        int no_of_c_data_packs_needed=0,no_of_not_firing_data_in_each_pack=0,no_of_firing_data_in_each_pack=0;
        int additional_firing_data_in_the_last_datapack=0,additional_not_firing_data_in_the_last_datapack=0;
        if(sum_total_not_firing_data>=f_data_pack->at(a).data.size())//for not firing data > firing data
        {
            while(sum_total_not_firing_data>=f_data_pack->at(a).data.size())
            {
                //cout<<"\ncheck4="<<f_data_pack[a].data.size()<<" "<<sum_total_not_firing_data;
                sum_total_not_firing_data=sum_total_not_firing_data-f_data_pack->at(a).data.size();//cout<<"check2";
                no_of_c_data_packs_needed++;
            }
            int rem1=sum_total_not_firing_data;
            no_of_firing_data_in_each_pack=f_data_pack->at(a).data.size();
            no_of_not_firing_data_in_each_pack=f_data_pack->at(a).data.size()+rem1/no_of_c_data_packs_needed;
            additional_not_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
        }
        else if(sum_total_not_firing_data<f_data_pack->at(a).data.size()) //for firing data more than not firing data
        {
            int sum_total_firing_data=f_data_pack->at(a).data.size();
            while(sum_total_firing_data>=sum_total_not_firing_data)
            {
                sum_total_firing_data=sum_total_firing_data-sum_total_not_firing_data;//cout<<"check3";
                no_of_c_data_packs_needed++;
            }
            int rem1=sum_total_firing_data;
            no_of_not_firing_data_in_each_pack=sum_total_not_firing_data;
            no_of_firing_data_in_each_pack=sum_total_not_firing_data+rem1/no_of_c_data_packs_needed;
            additional_firing_data_in_the_last_datapack=rem1%no_of_c_data_packs_needed;
        }
        
        //packaging the data
        if(no_of_firing_data_in_each_pack==f_data_pack->at(a).data.size())//this means firing data < not firing data
        {
            int no_of_packages_created=0;
            int initial_value=0,final_value=0;
            vector<vector<float>> not_firing_data_temp;
            not_firing_data_temp.clear();
            //copying all the not firing data in not_firing_data_temp
            for(int b=0;b<f_data_pack->size();b++)
            {
                if(b!=a)
                {
                    for(int c=0;c<f_data_pack->at(b).data.size();c++)
                    {   not_firing_data_temp.push_back(f_data_pack->at(b).data[c]);}
                }
            }
            while(no_of_packages_created!=no_of_c_data_packs_needed)
            {
                //clearing the buffers
                c_datapack.firing_data.clear();
                c_datapack.not_firing_data.clear();
                //c_datapack.objective_function_coefficients.clear();
                c_datapack.weight_matrix.clear();
                //packing the firing data
                for(int b=0;b<f_data_pack->at(a).data.size();b++)
                {   c_datapack.firing_data.push_back(f_data_pack->at(a).data[b]);}
                //packing not firing data
                initial_value=final_value;
                final_value=final_value+no_of_not_firing_data_in_each_pack;
                if(no_of_packages_created==no_of_c_data_packs_needed-1)//for the last package
                {   final_value=final_value+additional_not_firing_data_in_the_last_datapack;}
                //cout<<"\nnot_firing_data_temp size= "<<not_firing_data_temp.size();
                //cout<<"\nfinal_value= "<<final_value<<" initial_value= "<<initial_value;
                //cout<<"\nadditional_not_firing_data_in_the_last_datapack= "<<additional_not_firing_data_in_the_last_datapack;
                for(int b=initial_value;b<final_value;b++)
                {
                    c_datapack.not_firing_data.push_back(not_firing_data_temp[b]);
                }
                //setting up the label and output neuron index
                c_datapack.firing_label=f_data_pack->at(a).label;
                c_datapack.firing_neuron_index=a;
                //setting up the objective function coefficient 
                /*for(int b=0;b<c_datapack.firing_data[0].size();b++)
                {
                    float summation=0;
                    for(int c=0;c<c_datapack.firing_data.size();c++)
                    {   summation=summation+c_datapack.firing_data[c][b];}
                    c_datapack.objective_function_coefficients.push_back(summation);
                }*/
                //pushing the c_datapack in c_datapacks vector
                c_datapacks.push_back(c_datapack);
                no_of_packages_created++;
            }
        }
        else if(no_of_not_firing_data_in_each_pack==sum_total_not_firing_data)//this means firing data > not firing data
        {
            int no_of_packages_created=0;
            int initial_value=0,final_value=0;
            while(no_of_packages_created!=no_of_c_data_packs_needed)
            {
                //clearing the buffers
                c_datapack.firing_data.clear();
                c_datapack.not_firing_data.clear();
                //c_datapack.objective_function_coefficients.clear();
                c_datapack.weight_matrix.clear();
                //packing the firing data
                initial_value=final_value;
                final_value=final_value+no_of_firing_data_in_each_pack;
                if(no_of_packages_created==no_of_c_data_packs_needed-1)
                {   final_value=final_value+additional_firing_data_in_the_last_datapack;}
                for(int b=initial_value;b<final_value;b++)
                {
                    c_datapack.firing_data.push_back(f_data_pack->at(a).data[b]);
                }
                //packing the not firing data
                for(int b=0;b<f_data_pack->size();b++)
                {
                    if(b!=a)
                    {
                        for(int c=0;c<f_data_pack->at(b).data.size();c++)
                        {   c_datapack.not_firing_data.push_back(f_data_pack->at(b).data[c]);}
                    }
                }
                //setting up the label and output neuron index
                c_datapack.firing_label=f_data_pack->at(a).label;
                c_datapack.firing_neuron_index=a;
                //setting up the objective function coefficient
                /*for(int b=0;b<c_datapack.firing_data[0].size();b++)
                {
                    float summation=0;
                    for(int c=0;c<c_datapack.firing_data.size();c++)
                    {   summation=summation+c_datapack.firing_data[c][b];}
                    c_datapack.objective_function_coefficients.push_back(summation);
                }*/
                //pushing the c_datapack in c_datapacks vector
                c_datapacks.push_back(c_datapack);
                no_of_packages_created++;
            }
        }
    }
    f_data_pack->clear();//memory_optimization3
    message.clear();
    message="finished packaging data in c_datapacks.";
    print_message();
    message.clear();
    message="\ntotal no of c_data_packs= "+to_string(c_datapacks.size());
    print_message();
    big_c_datapack_handler(c_datapacks);//for handling c_datapack with huge data which may create full conlflict senarios.
    message.clear();
    message="\ntotal no of c_data_packs after big c_datapacks handling= "+to_string(c_datapacks.size());
    print_message();
    //this is the place for parallelization process.
    vector<vector<converted_data_pack>> c_datapacks_vector;
    message.clear();
    message="\narranging c_datapacks for "+to_string(no_of_threads)+" threads..........";
    print_message();
    c_data_packs_division_for_multi_threading(c_datapacks_vector,c_datapacks,no_of_threads);
    point1:
    if(no_of_threads!=c_datapacks_vector.size())
    {
        message.clear();
        message="\nSetting the no_of_threads to "+to_string(c_datapacks_vector.size());
        print_message();
        no_of_threads=c_datapacks_vector.size();
    }
    //memory_optimization4 : turn  vector<simplex_solver_data_preparation_class> to  vector<simplex_solver_data_preparation_class*>       
    vector<simplex_solver_data_preparation_class> lpp_solver_vec;
    for(int a=0;a<c_datapacks_vector.size();a++)
    {
        simplex_solver_data_preparation_class lpp_solver=simplex_solver_data_preparation_class(c_datapacks_vector[a],&ds,&network1);//initializing the obj of the class   
        lpp_solver_vec.push_back(lpp_solver);
    }
    vector<thread> thread_vec(no_of_threads);
    //thread* progress_diaplay_thread;
    message.clear();
    message=" lpp_solver_vec size="+to_string(lpp_solver_vec.size());
    print_message();
    //lpp solvers will start now.........
    for(int a=0;a<no_of_threads;a++)
    {   thread_vec[a]=thread(&simplex_solver_data_preparation_class::lp_solver,lpp_solver_vec[a]);}
    //int progress_bar_error;
    //if(pds==true)
    //{   progress_diaplay_thread=new thread(&core_class::display_training_progress,this);}
    for(int a=0;a<no_of_threads;a++)
    {   thread_vec[a].join();}
    lpp_solver_vec.clear();
    thread_vec.clear();
    c_datapacks.clear();
    c_datapacks_vector.clear();
    //if(pds==true)
    //{   progress_diaplay_thread->join();}
}

int core_class::size_of_c_datapacks_vector(vector<converted_data_pack> &c_datapacks)
{
    int sum=0;
    for(int a=0;a<c_datapacks.size();a++)
    {   sum+=c_datapacks[a].firing_data.size();}
    return sum;
}

void core_class::c_data_packs_division_for_multi_threading(vector<vector<converted_data_pack>> &c_datapacks_vector,vector<converted_data_pack> &c_datapacks,int no_of_threads)
{
    int total_data=0;
    for(int a=0;a<c_datapacks.size();a++)
    {   total_data+=c_datapacks[a].firing_data.size();}
    int data_for_each_thread=total_data/no_of_threads;
    vector<converted_data_pack> c_datapacks_new;
    c_datapacks_new.clear();
    c_datapacks_vector.clear();
    message.clear();
    message="\nno of c_datapacks= "+to_string(c_datapacks.size());
    print_message();
    for(int a=0;a<no_of_threads;a++)
    {
        int b=c_datapacks.size()-1;
        if(b<0)
        {   continue;}
        while(size_of_c_datapacks_vector(c_datapacks_new)<data_for_each_thread)
        {                
            c_datapacks_new.push_back(c_datapacks[b]);
            c_datapacks.erase(c_datapacks.begin()+b);
            b--;
            if(b<0)
            {   break;}
        }
        c_datapacks_vector.push_back(c_datapacks_new);
        c_datapacks_new.clear();
    }
    int sum=0;
    for(int a=0;a<c_datapacks_vector.size();a++)
    {   sum+=c_datapacks_vector[a].size();
        message.clear();
        message="\na= "+to_string(a)+" size= "+to_string(c_datapacks_vector[a].size());
        print_message();
    }
    message.clear();
    message="\nno of c_datapacks in c_datapacks_vector= "+to_string(sum);
    print_message();
    message.clear();
    message="\nc_data_packs size= "+to_string(c_datapacks.size());
    print_message();
}

void core_class::network_structure_modifier()
{
    try{
        if(ns.no_of_input_neuron>ds.no_of_elements_in_each_record)
        {
            throw("network has more neuron than required by the data");//needs working here. UNDER CONSTRUCTION.
        }
        else
        {
            network1.set_no_of_input_neuron(ds.no_of_elements_in_each_record);
            network1.set_no_of_output_neuron(ds.no_of_labels);
        }
    }
    catch(string s)
    {   cout<<s<<endl;}
}

void core_class::print_message()
{
    if(display_core_events)
    {
        pthread_mutex_lock(&lock_1);
        cout<<message;
        pthread_mutex_unlock(&lock_1);
    }
}

void core_class::set_critical_variable(chromosome critical_variable)
{   
    network1.set_critical_variables(critical_variable);
    network1.path.clear();
    ds.lower_firing_constrain_rhs=critical_variable.rhs_lower;
    ds.upper_not_firing_constrain_rhs=critical_variable.rhs_upper;
}

core_class::core_class(
    int core_aim1,
    int core_no1,
    int parent_segment_aim1,
    int parent_segment_no1,
    string core_name1,
    datapack_structure_defination& ds1)
{
    if(id_lock==false)
    {
        core_aim=core_aim1;
        core_no=core_no1;
        parent_segment_aim=parent_segment_aim1;
        parent_segment_no=parent_segment_no1;
        core_name=core_name1;
        id_lock=true;
        ds=ds1;
    }
    else
    {  
        message.clear();
        message="Failed to set core number and core aim as id_lock=true";
        print_message();
    }
}

void core_class::clrscr()
{
    cout << "\033[2J\033[1;1H";
    //system("clear");
}
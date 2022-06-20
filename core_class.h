/*
core class handles natural cores
*/

#include<iostream>
#include<vector>
#include<fstream>
#include<sys/stat.h>
#include<string>
#include<algorithm>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<thread>
#include<dirent.h>
#include<unistd.h>
#include<sys/ioctl.h>//for the terminal size

#include"neuron_and_ann_class.h"

using namespace std;
static pthread_mutex_t lock_1;
inline bool display_iterations=false;//iteration display switch for debugging the code

struct datapack_structure_defination{
    int no_of_labels;
    int no_of_elements_in_each_record;
    vector<float> elements;
    float lower_firing_constrain_rhs;//=92; //60,150
    float upper_not_firing_constrain_rhs;//=10; //10
};

struct converted_data_pack
{
    vector<vector<float>> firing_data;
    vector<vector<float>> not_firing_data;
    //vector<float> objective_function_coefficients;
    vector<float> weight_matrix;//ans stored here.
    int firing_neuron_index;
    float firing_label;
    bool corupt_pack=false;
};

struct conflicting_data_id
{
    vector<int> id;
    bool conflict_id_present=false;
};

class modified_simplex_solver{
    private:
    //training settings
    float lower_firing_constrain_rhs; //60,150
    float upper_not_firing_constrain_rhs; //10

    string message;
    struct id{
            bool slack=false,basic=false,z=false,rhs=false,theta=false;
            int id;
        };

    struct simplex_table
    {
        vector<id> c_id;//no_of_columns-rhs-z
        vector<id> r_id; //no_of_rows-z_row
        vector<vector<float>> basic_var; //no_of_column-slack_var-z-rhs-theta*no_of_rows-z_row
        vector<vector<float>> slack_var; //no_of_columns-basic_var-rhs-theta*no_of_rows-z_row
        //vector<float> z_col;//no_of_rows-z_row
        vector<double> rhs;//no_of_rows-z_row//actual double
        vector<long double> theta;//no_of_rows-z_row//actual long double
        //vector<float> z_row;//no_of_columns
    };

    class simplex_optimizer{

    }s_optimizer;
    
    class make_solution_feasible{

        private:
        bool cyclic_bug_present_var=false;
        string message;

        void print_message();

        bool termination_condition_checker(simplex_table* st);

        void display_st(simplex_table st);

        long double round_to_zero(long double input);
        struct buffer
        {
            vector<float> basic_plus_slack_plus_z_plus_rhs_temp;
            vector<long double> sorted_theta;
            vector<int> p_row_index;
            vector<int> p_col_index;
        }buffer_obj;
        
        bool add_index_data(int p_col,int p_row);

        void simplex_table_modifier(int p_row_index,int p_col_index,simplex_table* st);

        conflicting_data_id conflict_id;

        void conflicting_data_finder(simplex_table* st);

        int iteration=0;
        void pivot_element_finder(simplex_table* st);

        public:
        bool cyclic_bug_present();

        conflicting_data_id return_conflict_id_pack();

        void start(simplex_table* st);

    }feasible_solution_calculator;

    void display_st(simplex_table st);
    
    void print_message();

    public:
    static bool check_for_corrupt_cdp(converted_data_pack* cdp);
    bool start_solver(converted_data_pack* cdp);
    bool cyclic_bug_present();
    converted_data_pack conflicting_data;
    void set_training_settings(float lower_firing_constrain_rhs1,float upper_not_firing_constrain_rhs1);
};

class simplex_solver_data_preparation_class
{
    private:
    bool display_core_events=false;
    string message;
    struct conflicting_data_buffer_vector_struct
    {
        vector<converted_data_pack> conflicting_data_buffer_vector;
    }conflicting_data_buffer_outer;

    ann* network;
    datapack_structure_defination* data_structure;
    converted_data_pack conflicting_data_buffer;
    modified_simplex_solver lpp_solver1;
    vector<converted_data_pack> cdp;//memory_optimization7 : vector<converted_data_pack> cdp turned to vector<converted_data_pack*> cdp 

    void cdp_saver_in_mathematical_format(converted_data_pack* cdp);
    
    void print_message();
    vector<converted_data_pack> fucked_up_cyclic_cdp;
    void cdp_spliter(vector<converted_data_pack> &cdps,int index);
    public:
    static void cdp_viewer(converted_data_pack* cdp);
    void lp_solver();
    bool cyclic_bug_present();
    simplex_solver_data_preparation_class(vector<converted_data_pack> &cdps,datapack_structure_defination* ds,ann* network1);
};
//int training_step_counter=0;

struct network_structure_defination{
    int no_of_input_neuron=0;
    int no_of_output_neuron=0;
};

class core_class
{
    private:
    bool display_core_events=false;
    //training data pointers
    int *split_start,*split_end;
    vector<nn_core_filtered_data>* f_data_pack;
    //core identification information
    int core_no=0,core_aim=0;//this two must be changed using a function so that proper core is loaded
    int parent_segment_aim=0,parent_segment_no=0;
    string core_name;
    string core_save_file_name="NULL";//provided if core is loaded from a core/network savefile. Not set using constructor
    bool id_lock=false;
    //training information
    ann network1;
    int no_of_threads;
    datapack_structure_defination ds;//for the data which will be processed by this particular core
    network_structure_defination ns;

    void network_structure_modifier();

    void network_analyzer();//it fills up the network_structure_defination ns based on the initialized network structure. 

    struct shuffling_data{
        vector<float> temp_data;
        int temp_label;
    };

    void big_c_datapack_handler(vector<converted_data_pack> &cdp);//passing the vector by reference //this function might be a temporary offer //this is for preventing 0:0 bug

    int size_of_c_datapacks_vector(vector<converted_data_pack> &c_datapacks);

    void c_data_packs_division_for_multi_threading(vector<vector<converted_data_pack>> &c_datapacks_vector,vector<converted_data_pack> &c_datapacks,int no_of_threads);

    nn_core_data_package_class test_data;

    string message;
    void print_message();
    void clrscr();
    vector<string> line_breaker(string line);
    
    public:
    
    bool load_core(string core_file_dir="");

    void clear_core();//deletes all the data and network present inside the core. This function is not yet implemented.
    
    vector<neuron> propagate(vector<float> input_attributes_value);//need to be implemented

    void load_training_data_into_core(vector<nn_core_filtered_data>& f_data_pack1,int& no_of_threads1,int &split_start,int &split_end);

    void train_core();//earlier called simplex_solver_data_entry_point

    void save_core(string folder_dir);
    
    network_structure_defination return_ns()
    {   return ns;}
    string return_name()
    {   return core_name;}
    string return_core_savefile_name()
    {   return core_save_file_name;}
    int return_core_no()
    {   return core_no;}
    int return_no_of_input_neuron()
    {   return network1.input_neuron_size();}
    int return_no_of_output_neuron()
    {   return network1.output_neuron_size();}
    datapack_structure_defination return_core_ds()
    {   return ds;}
    int return_core_aim()
    {   return core_aim;}
    int return_no_of_paths()
    {   return network1.path.size();}

    void set_critical_variable(chromosome critical_variable);

    core_class(int core_aim,int core_no,int parent_segment_aim,int parent_segment_no,string core_name,datapack_structure_defination& ds1);
};

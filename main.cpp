#include<iostream>
#include<string.h>

#include"segment_start.h"

using namespace std;

int main()
{
    int test_train_predict;
    string file_name="NOT_AVAILABLE",network_save_file_name="NOT_AVAILABLE";
    int no_of_threads=1;
    menu(file_name,test_train_predict,network_save_file_name,no_of_threads);
    if(test_train_predict!=-1)
    {   segment_starter(file_name,test_train_predict,network_save_file_name,no_of_threads);}
    return 0;
}
% File paths

add = 'E:/lfb research/microgel/new_tempreture_code/';
sample = 'AO12_VAHEAT-C2_Hx_O2Plasma-5min_NR-10e-10_50degC_merged_driftcor.ts.csv_xyzif.txt'
raw_pc_path = strcat(add, sample);
% raw_pc_path = strcat('sample_data.txt');
save_path_selected = "results/";

% Hyperparameters
use_spherical_information = true;
Percentage_select = 0.71;
Percentage_drop = 0.29;

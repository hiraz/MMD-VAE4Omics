"""
This is the code for classification performance & RoC graphs with AUC values
Finalised on 18/03/2020 & update on 4-12-2020
By Raz-Hira

"""
# Imports the packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import auc
#from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
from glob2 import glob
from classifier_ann_LFs_with_CrossValidation import classifier_ANN
from confusion_matrix_plotting import c_matrixPlotting

# #############################################################################

# Fixed values
latent_dim = 128

# Add noisy features
random_seed= 11

### Data loading#############

data_path = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/datasets/'

saving_path_root = "C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/results/subtypes_classification/latent_features_128/unsupervised_learned_LFs_classification/ANN/"


# Creating an empty Dataframe with column names only with metrics
df_performance = pd.DataFrame(columns=['Method','Omic_data', 'Accuracy', 'Precison', 'Recall', 'f1 score'])
df_performance.set_index('Method')

# omic_numbers
mono_omic = 1
di_omic = 2
tri_omic = 3
#NLL_lossf = 0 # if 0, we will select BCE as the loss function else NLL loss functionNLL_lossf = 0 
vae_mmd = True # if 1, it will select the files for mmd-vae & if 0, will select latent files for pca-tsne 
label_type = 1 # 1= tumour type, 2 means stage & 3 mean grade of cancer

gdc = False
# Load the tumor type or y informaiton


# Path to save performance data
## Path for the combined latent features
# witout early stopping
if vae_mmd:
    latent_dim = 128
    if label_type == 1:
        LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/'
        LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/'
        LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/'  
        saving_path = saving_path_root + 'NLL/vae-mmd/subtype/'
        ### load the labels
        label_481 = pd.read_csv(data_path + 'datasets_omics_features/common/ov_samples_with_subtypes_digit_481samples.txt', sep='\t', header=0, index_col=0)
        label_292 = pd.read_csv(data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_292samples.txt', sep='\t', header=0, index_col=0)
        label_459 = pd.read_csv(data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_459samples.txt', sep='\t', header=0, index_col=0)
        
    elif label_type == 2:
        LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/NLL-MMD-based-150-epoch/stage/'
        LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/NLL-MMD-based-150-epoch/stage/'
        LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/NLL-MMD-based-150-epoch/stage/'
        saving_path = saving_path_root + 'NLL/vae-mmd/stage/'
        
        ### load the labels for the cacner stage 
        label_481 = pd.read_csv(data_path + 'datasets_omics_features/common/stages_481samples_original_stages_digits.csv', header=0, index_col=0)
        label_292 = pd.read_csv(data_path +'datasets_omics_features/common/stages_292samples_original_stages_digits.csv', header=0, index_col=0)
        label_459 = pd.read_csv(data_path +'datasets_omics_features/common/stages_459samples_original_stages_digits.csv', header=0, index_col=0)
    else:
        LF_root_gdc = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/gdc_methylation_only/with_smote/'
        LF_path1d = LF_root_gdc + 'vae-mmd/'
        saving_path = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/results/subtypes_classification/latent_features_128/cancer_normal_cell/'
        label_481 = pd.read_csv(LF_root_gdc + 'smoted_from_613labels.csv', header=0, index_col=0) # this is for gdc samples

      
else:
    
    LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/pca-tsne/'
    LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/pca-tsne/'
    LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/pca-tsne/'       
    
    
    if label_type == 1:
        saving_path = saving_path_root + 'NLL/pca-tsne/subtype/'
        ### load the labels
        label_481 = pd.read_csv(data_path + 'datasets_omics_features/common/ov_samples_with_subtypes_digit_481samples.txt', sep='\t', header=0, index_col=0)
        label_292 = pd.read_csv(data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_292samples.txt', sep='\t', header=0, index_col=0)
        label_459 = pd.read_csv(data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_459samples.txt', sep='\t', header=0, index_col=0)
        latent_dim = 2
    elif label_type == 2:
        saving_path = saving_path_root + 'NLL/pca-tsne/stage/'
        ### load the labels for the cacner stage 
        label_481 = pd.read_csv(data_path + 'datasets_omics_features/common/stages_481samples_original_stages_digits.csv', header=0, index_col=0)
        label_292 = pd.read_csv(data_path +'datasets_omics_features/common/stages_292samples_original_stages_digits.csv', header=0, index_col=0)
        label_459 = pd.read_csv(data_path +'datasets_omics_features/common/stages_459samples_original_stages_digits.csv', header=0, index_col=0)
        latent_dim = 2
    else:
        LF_root_gdc = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/gdc_methylation_only/with_smote/'
        LF_path1d = LF_root_gdc + 'pca-tsne/'
        saving_path = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/results/subtypes_classification/latent_features_128/cancer_normal_cell/'
        label_481 = pd.read_csv(LF_root_gdc + 'smoted_from_613labels.csv', header=0, index_col=0) # this is for gdc samples
        latent_dim = 3
        
# performance saving paths    

performance_path_1 = saving_path + str(latent_dim) + 'mono-omic-performance.csv'# path to save the performance matrics for mono-omic
performance_path_2 = saving_path + str(latent_dim) +' di-omic-performance.csv'# path to save the performance matrics for mono-omic
performance_path_3 = saving_path + str(latent_dim)  + 'tri-omic-performance.csv'# path to save the performance matrics for mono-omic
    
# Load all the data using 
# Read the each data file's full path and form a list
# mono-omic
# mono-omic
all_data_mono_omic = glob(LF_path1d + "*.tsv")
sorted(all_data_mono_omic)
if gdc:
    mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic_gdc.csv', header=0)
else:
    #mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic.csv', header=0)
    mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic_mmd_vae_only.csv', header=0)
    

# # di-omics
# all_data_di_omic = glob(LF_path2d + "*.tsv")
# sorted(all_data_di_omic)
# # tri-omics
# all_data_tri_omic = glob(LF_path3d + "*.tsv")
# sorted(all_data_tri_omic)

# if vae_mmd:
#     mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic_mmd_vae_only.csv', header=0)
#     di_omic_lf_df = pd.read_csv(LF_path2d + 'latent_file_name_ids_di_omic_mmd_vae_only.csv', header=0)
#     tri_omic_lf_df = pd.read_csv(LF_path3d + 'latent_file_name_ids_tri_omic_mmd_vae_only.csv', header=0)
# else:
#     mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic.csv', header=0)
#     di_omic_lf_df = pd.read_csv(LF_path2d + 'latent_file_name_ids_di_omic.csv', header=0)
#     tri_omic_lf_df = pd.read_csv(LF_path3d + 'latent_file_name_ids_tri_omic.csv', header=0)
    


############

def load_dataset(all_data, omic_df_info, omic_num, saving_path):
    for i,name in enumerate(all_data):
        lf_path = all_data[i]
        #print (lf_path)
        
        LFs_learned_all = pd.read_csv(lf_path, sep='\t', header=0, index_col=0) # mRNA-cnv-methly
        print(LFs_learned_all)
        
        data_type = omic_df_info.loc[i].at['omic_data']
        print (data_type)
        algorithom_name = omic_df_info.loc[i].at['method']
       # print(' My Algorithm name is: ', algorithom_name )
        method = str(algorithom_name) + ' + ANN'
         
    
    #     # For mono_omic data
        if omic_num == mono_omic:
            print ("This is for mono-omic datasets")
            
            if  data_type == "RNAseq":
               
                # call the classifier 
                a, p, r, f1 = classifier_ANN(LFs_learned_all, label_292, latent_dim, label_type)
                add_row =[method, data_type, a, p, r, f1]
                df_performance.loc[len(df_performance)] = add_row
            else:
                #print (latent_code_2d.shape)
                
                if data_type == "CNV":
                    # call the classifier 
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
    
                if data_type == "mRNA":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
                    
                if data_type == "methylation":
                    #fig3, ax3 = plt.subplots()
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
                    #c_matrixPlotting(actuals_y, out_pred_y, method, data_type, saving_path)
            
            ####### For di_omic dataset##############
            
        if omic_num == di_omic :
            
            print ("This is the ROC curves for di-omics datasets")
            
            # this is for mRNA, CNV & methylation 
            if  data_type == "CNV_mRNA" or data_type == "mRNA_methylation" or data_type == "CNV_methylation": 
                if data_type == "CNV_mRNA":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row                   
                
                elif data_type == "mRNA_methylation":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row                                        
                
                else: #data_type == "CNV_methylation":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
                                         
        
            elif data_type == "RNAseq_CNV" or data_type == "RNAseq_methylation": 
                if data_type == "RNAseq_CNV":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all, label_292, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
                                                   
                    # for "RNAseq_methylation"
                else:
                    a, p, r, f1 = classifier_ANN(LFs_learned_all, label_292, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row

            else: # this is mRNA_miRNA 
                a, p, r, f1 = classifier_ANN(LFs_learned_all, label_459, latent_dim, label_type)
                add_row =[method, data_type, a, p, r, f1]
                df_performance.loc[len(df_performance)] = add_row


        ####### For tri_omic dataset##############
            
        if omic_num == tri_omic :
            print ("This is the ROC curves for tri-omics datasets")
            if data_type == "CNV_mRNA_methylation":
                    a, p, r, f1 = classifier_ANN(LFs_learned_all,label_481, latent_dim, label_type)
                    add_row =[method, data_type, a, p, r, f1]
                    df_performance.loc[len(df_performance)] = add_row
                    
            elif data_type == "RNAseq_CNV_methylation":
                a, p, r, f1 = classifier_ANN(LFs_learned_all, label_292, latent_dim, label_type)
                add_row =[method, data_type, a, p, r, f1]
                df_performance.loc[len(df_performance)] = add_row
             
            else: 
                a, p, r, f1 = classifier_ANN(LFs_learned_all, label_292, latent_dim, label_type)
                add_row =[method, data_type, a, p, r, f1]
                df_performance.loc[len(df_performance)] = add_row
                
                
    if omic_num == mono_omic:
        
        df_performance.to_csv(performance_path_1, encoding='utf-8-sig') # will save only mono-omic performances

    elif omic_num == di_omic:
        df_performance.to_csv(performance_path_2, encoding='utf-8-sig') # will save mono-omic+di-omics performances
    else:
        df_performance.to_csv(performance_path_3, encoding='utf-8-sig') # will save omic+di-omics+tri-omcs performances

### Functions calling for ROC & plotting
                
# Call the function to load latent features data and calculate performacnes, RoC with AUC
# if not loaded data for allomic combination,  PCA_FPR= roc_curves_dic.get('PCA')["FPR"] TypeError: 'NoneType' object is not subscriptable                
load_dataset(all_data_mono_omic, mono_omic_lf_df, mono_omic, saving_path)
#load_dataset(all_data_di_omic, di_omic_lf_df, di_omic)
#load_dataset(all_data_tri_omic, tri_omic_lf_df, tri_omic)
#omic_num = mono_omic




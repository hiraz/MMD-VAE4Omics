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

# #############################################################################
# Data IO and generation
#plt.style.use('seaborn')
#plt.style.available # to find the available styles

# Few fix things
latent_code_dim3=3
latent_code_dim2=2
have_label=True

### Data loading#############

root = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/'
data_common = root + 'datasets/datasets_omics_features/common/'
result_path = root + "results/"
colour_file1 = data_common + 'colour_codings_ov_subtypes.txt'
colour_file2 = data_common + 'colour_codings_ov_cancer.txt'

colour_setting = pd.read_csv(colour_file2, sep='\t')

# result savings path
saving_path1 = result_path + 'clustering/cancer_normal_cell/unsupervised/' #/ for subtypes classification results
saving_path2 = result_path + 'clustering/subtype/unsupervised/' #/ for subtypes classification results
saving_path3 = result_path + 'clustering/stage/unsupervised/' #/ for subtypes classification results

saving_path4 = result_path + 'clustering/cancer_normal_cell/supervised/' #/ for subtypes classification results
saving_path5 = result_path + 'clustering/subtype/supervised/' #/ for subtypes classification results
saving_path6 = result_path + 'clustering/stage/supervised/' #/ for subtypes classification results

#data_path_gdc = 'C:/Users/U0032456PC/OneDrive - Teesside University/Research/My-papers-submitted-published/Hira-papers/Datasets-additional/GDC-OV-cohort/'

# omic_numbers
mono_omic = 1
di_omic = 2
tri_omic = 3
NLL_lossf = 1 # if 0, we will select BCE as the loss function else NLL loss function

# Load dtasets
data_path = "C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/datasets/"

#LF_root_gdc = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/gdc_methylation_only/with_smote/' 

LF_root_gdc = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/datasets/datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/gdc_methylation_only/without-early-stopping/all/'
#saving_path_root = "C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/results/subtypes_classification/latent_features_128/unsupervised_learned_LFs_classification/SVM/"


# Load the tumor type or y informaiton
molecular_subtype_481 = data_path + 'datasets_omics_features/common/ov_samples_with_subtypes_digit_481samples.txt'
molecular_subtype_292 = data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_292samples.txt'
molecular_subtype_459 = data_path +'datasets_omics_features/common/ov_samples_with_subtypes_digit_459samples.txt'
#label_gdc = LF_root_gdc  + 'smoted_from_613labels.txt'
label_gdc = LF_root_gdc  + 'sample_id_with_cell_type613.txt'

## Path for the combined latent features
label_type = 3 # 1= tumour type, 2 means stage & 3 mean grade of cancer
supervised = True
gdc = True
# witout early stopping
if label_type == 1:
    if NLL_lossf:
        if supervised:
            LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/supervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'
            LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/supervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'
            LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/supervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'  
            saving_path = saving_path5 
        else:            
            LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'
            LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'
            LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/NLL-MMD-based-150-epoch/with-early-stopping/with-pca-tsne/'  
            saving_path = saving_path2    
    else: 
        LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/BCE-based-150-epoch/with-early-stopping/with-pca-tsne/'
        LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/BCE-based-150-epoch/with-early-stopping/with-pca-tsne/'
        LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/BCE-based-150-epoch/with-early-stopping/with-pca-tsne/' 
        saving_path = saving_path2
       
elif label_type == 2:
    if NLL_lossf:
        if supervised:
            LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/supervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'
            LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/supervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'
            LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/supervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'  
            saving_path = saving_path6   
        
        else:
                
            LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'
            LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'
            LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/NLL-MMD-based-150-epoch/stage/with-pca-tsne/'  
            saving_path = saving_path3    
    else: 
        LF_path1d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/BCE-based-150-epoch/stage/with-pca-tsne/'
        LF_path2d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_di_omic/unsupervised/BCE-based-150-epoch/stage/with-pca-tsne/'
        LF_path3d = data_path + 'datasets_with_Latent_Features/latent_feautres_128_tri_omic/unsupervised/BCE-based-150-epoch/stage/with-pca-tsne/' 
        saving_path = saving_path3
else:
   # LF_path1d = LF_root_gdc + 'all/smotesvm/'
    LF_path1d = LF_root_gdc 
    #saving_path = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/results/clustering/cancer_normal_cell/'
    saving_path = LF_root_gdc

'''
Function for scatter plot

'''

def plot_scatter(latent_code, label_file, colour_file, latent_code_dim, method, data_type, fig_path, label_type):
  
    if latent_code_dim <= 3:
        if latent_code_dim == 3:
            # Plot the 3D scatter graph of latent space
            if have_label:
                # Set sample label
                #disease_id = pd.read_csv('data/methylation/ov_subtypes_digit_481samples.txt', sep='\t', header=0, index_col=0)
                disease_id = pd.read_csv(label_file, sep='\t', header=0, index_col=0)
                latent_code_label = pd.merge(latent_code, disease_id, left_index=True, right_index=True)
                colour_setting = pd.read_csv(colour_file, sep='\t')
                fig = plt.figure(figsize=(8, 5.5))
                ax = fig.add_subplot(111, projection='3d')
                for index in range(len(colour_setting)):
                    code = colour_setting.iloc[index, 1]
                    colour = colour_setting.iloc[index, 0]
                    if code in latent_code_label.iloc[:, latent_code_dim].unique():
                        latent_code_label_part = latent_code_label[latent_code_label.iloc[:, latent_code_dim] == code]
                        ax.scatter(latent_code_label_part.iloc[:, 0], latent_code_label_part.iloc[:, 1],
                                   latent_code_label_part.iloc[:, 2], s=2, marker='o', alpha=0.8, c=colour, label=code)
                ax.legend(ncol=2, markerscale=4, bbox_to_anchor=(1, 0.9), loc='upper left', frameon=False)
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(latent_code.iloc[:, 0], latent_code.iloc[:, 1], latent_code.iloc[:, 2], s=2, marker='o',
                           alpha=0.8)
            ax.set_xlabel('First Latent Dimension')
            ax.set_ylabel('Second Latent Dimension')
            ax.set_zlabel('Third Latent Dimension')
            plt.tight_layout()
            plt.legend(loc="lower right")
            #plt.title('Molecular subtype clustering using' + str(method) + str(data_type) + ' data')
            plt.title('Cancer stage clustering using' + str(method) + str(data_type) + ' data')
            #plt.show()
            plt.savefig(fig_path, dpi=300)
            
        elif latent_code_dim == 2:
            if have_label:
                # Set sample label
                disease_id = pd.read_csv(label_file, sep='\t', header=0, index_col=0)
                #print(disease_id.shape)
                latent_code_label = pd.merge(latent_code, disease_id, left_index=True, right_index=True)
                #print (latent_code_label.shape)
                colour_setting = pd.read_csv(colour_file, sep='\t')
                
                #x= latent_code_label_part.iloc[:, 0]
                plt.figure(figsize=(8.5, 5.5))
                plt.rc('xtick', labelsize=20) 
                plt.rc('ytick', labelsize=20) 
                for index in range(len(colour_setting)):
                    #print('my name')
                    code = colour_setting.iloc[index, 1]
                    #print(code)
                    colour = colour_setting.iloc[index, 0]
                    #print(colour)
                    if code in latent_code_label.iloc[:, latent_code_dim].unique():
                        #print(code)
                        latent_code_label_part = latent_code_label[latent_code_label.iloc[:, latent_code_dim] == code]
                        #latent_code_label_part = latent_code_label
                        plt.scatter(latent_code_label_part.iloc[:, 0], latent_code_label_part.iloc[:, 1], s=2,
                                    marker='o', alpha=0.8, label=code)
                plt.legend(ncol=2, markerscale=4, bbox_to_anchor=(1, 1), loc='upper left', frameon=False)
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
            else:
                plt.scatter(latent_code.iloc[:, 0], latent_code.iloc[:, 1], s=2, marker='o', alpha=0.8)
            plt.xlabel('First Latent Dimension', fontsize=20)
            plt.ylabel('Second Latent Dimension', fontsize=20)
            plt.tight_layout()
            plt.legend(loc="lower right", fontsize=20)
            # if label_type == 1:
            #     plt.title('Molecular subtype clustering using' + str(method) + ' & ' + str(data_type) + ' data') # for subtypes
            # elif label_type == 2:
            #     plt.title('Cancer stage clustering using' + str(method) + ' & ' + str(data_type) + ' data') # for subtypes
            # else : 
            #     #plt.title('Normal vs. Cancer samples clustering using ' + str(method) + ' & ' + str(data_type) + ' data') # for subtypes with tsne
                #plt.title('Normal vs. Cancer samples clustering using ' + str(method) + ' + t-SNE ' + ' & ' + str(data_type) + ' data') # for subtypes with tsne
            #plt.show()
            plt.savefig(fig_path, dpi=300)
            

            
# Load all the data using 
# Read the each data file's full path and form a list
# mono-omic
all_data_mono_omic = glob(LF_path1d + "*.tsv")
sorted(all_data_mono_omic)
if gdc:
    mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic_gdc.csv', header=0)
else:
    mono_omic_lf_df = pd.read_csv(LF_path1d + 'latent_file_name_ids_mono_omic.csv', header=0)
    
# # # # # di-omics
# all_data_di_omic = glob(LF_path2d + "*.tsv")
# sorted(all_data_di_omic)
# di_omic_lf_df = pd.read_csv(LF_path2d + 'latent_file_name_ids_di_omic.csv', header=0)

# # # tri-omics
# all_data_tri_omic = glob(LF_path3d + "*.tsv")
# sorted(all_data_tri_omic)
# tri_omic_lf_df = pd.read_csv(LF_path3d + 'latent_file_name_ids_tri_omic.csv', header=0)


latent_code_dim = 2



############

def load_dataset(all_data, omic_df_info, omic_num):
    for i,name in enumerate(all_data):
        lf_path = all_data[i]
        #print (lf_path)
        
        latent_code_all = pd.read_csv(lf_path, sep='\t', header=0, index_col=0) # mRNA-cnv-methly
        latent_code_2d = latent_code_all[latent_code_all.columns[0:latent_code_dim]]
        #print(latent_code_2d.values)
        #print(latent_code_all)
       # latent_code_2d = latent_code_2df.round(5)
        data_type = omic_df_info.loc[i].at['omic_data']
        print (data_type)
        algorithom_name = omic_df_info.loc[i].at['method']
       # print(' My Algorithm name is: ', algorithom_name )
        method = str(algorithom_name)
        
        
         # For mono_omic data
        if omic_num == mono_omic:
            print ("This is the clustering for mono-omic datasets")
            
            if  data_type == "RNAseq":
                
                print('This for RNAseq')
                fig_path = saving_path + str(method) + ' '+ str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_292, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type) 
                                
            else:
                #print (latent_code_2d.shape)
                if gdc ==1:
                    print('This for gdc cohort')
                    fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                    plot_scatter(latent_code_2d, label_gdc, colour_file2, latent_code_dim,  method, data_type, fig_path, label_type) 
                
                else:
                        
                    print('This for non-RNAseq')
                    fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                    plot_scatter(latent_code_2d, molecular_subtype_481, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type) 
        
        # for di-omics data
        if omic_num == di_omic :
            
            print ("This is the clustering for di-omics datasets")
            
            # this is for mRNA, CNV & methylation 
            if  data_type == "CNV_mRNA" or data_type == "mRNA_methylation" or data_type == "CNV_methylation": 
                fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_481, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type)                     
        
            elif data_type == "RNAseq_CNV" or data_type == "RNAseq_methylation": 
                fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_292, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type) 
                  
            else: # this is mRNA_miRNA 
                fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_459, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type) 
                
        # ####### For tri_omic dataset##############
            
        if omic_num == tri_omic :
            print ("This is the clustering for tri-omics datasets")
            if data_type == "CNV_mRNA_methylation":
                fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_481, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type)    
            else: 
                fig_path = saving_path + str(method) + str(data_type) + '.pdf'# for methylation 
                plot_scatter(latent_code_2d, molecular_subtype_292, colour_file1, latent_code_dim,  method, data_type, fig_path, label_type) 
        
        # latent_code1_2d = latent_code_all_1[latent_code_all_1.columns[0:2]]#for 2 latent factors & 2D figure for mRNA
        #fig_path = saving_path + str(method) + str(data_type) + 'd.pdf'# for methylation 
       # plot_scatter(latent_code_2d, label_100, colour_file1, latent_code_dim2, fig_path) #
              
        

load_dataset(all_data_mono_omic, mono_omic_lf_df, mono_omic)
#load_dataset(all_data_di_omic, di_omic_lf_df, di_omic)
#load_dataset(all_data_tri_omic, tri_omic_lf_df, tri_omic)
#load_dataset(all_data_tri_omic, tri_omic_lf_df, tri_omic)

# """
# This is for integrating t-SNE with VAE or MMD-VAE
# """       
# # TSNE
# latent_space_dimension = 2
# from sklearn.manifold import TSNE
# print('TSNE')


# # # mono-omic only
# tsne = TSNE(n_components=latent_space_dimension)
# # # # Load the mono-omic mRNA data
# # # # latent_code_all_mRNA = pd.read_csv( LF_path1d +'12.tsv', sep='\t', header=0, index_col=0)
# # # # z_mRNA = tsne.fit_transform(latent_code_all_mRNA.values)
# # # # latent_code_tsne_MMD1 = pd.DataFrame(z_mRNA, index=latent_code_all_mRNA.index)
# # # # fig_path_mRNA_tsne_MMD = saving_path + str(latent_code_dim) + 'tsne-MMD_mRNA.pdf'# for methylation 
# # # # plot_scatter(latent_code_tsne_MMD1,  molecular_subtype_481, colour_file1, latent_space_dimension, 'MMD-VAE', 'mRNA', fig_path_mRNA_tsne_MMD) #
# # # # #plot_scatter(latent_code_2d, molecular_subtype_292, colour_file1, latent_code_dim,  method, data_type, fig_path) 

# # # # # # Di-omics
# # # # latent_code_all_mRNA_methyl = pd.read_csv( LF_path2d +'16.tsv', sep='\t', header=0, index_col=0)
# # # # z_mRNA_methyl = tsne.fit_transform(latent_code_all_mRNA_methyl.values)
# # # # latent_code_tsne_MMD2 = pd.DataFrame(z_mRNA_methyl, index=latent_code_all_mRNA_methyl.index)
# # # # fig_path_mRNA_methyl_tsne_MMD = saving_path + str(latent_code_dim) + 'tsne-MMD_mRNA_methyl.pdf'# for methylation 
# # # # plot_scatter(latent_code_tsne_MMD2, molecular_subtype_481, colour_file1, latent_space_dimension, 'MMD-VAE', 'mRNA-methyl', fig_path_mRNA_methyl_tsne_MMD) #
# # # tri-omics
#latent_code_all_mRNA_cnv_methyl = pd.read_csv( LF_path3d +'12.tsv', sep='\t', header=0, index_col=0)
# z_mRNA_CNV_methyl = tsne.fit_transform(latent_code_all_mRNA_cnv_methyl.values)
# latent_code_tsne_MMD3 = pd.DataFrame(z_mRNA_CNV_methyl, index=latent_code_all_mRNA_cnv_methyl.index)
# fig_path_mRNA_CNV_metyl_tsne_MMD = saving_path + str(latent_code_dim) + 'tsne-MMD_mRNA_CNV_methyl.pdf'# for methylation 
# plot_scatter(latent_code_tsne_MMD3, molecular_subtype_481, colour_file1, latent_space_dimension, 'MMD-VAE', 'mRNA-CNV-methyl', fig_path_mRNA_CNV_metyl_tsne_MMD, label_type) #

# latent_code_all_mRNA_cnv_methyl_vae = pd.read_csv( LF_path3d +'11.tsv', sep='\t', header=0, index_col=0)
# z_mRNA_CNV_methyl_vae = tsne.fit_transform(latent_code_all_mRNA_cnv_methyl_vae.values)
# latent_code_tsne_VAE3 = pd.DataFrame(z_mRNA_CNV_methyl_vae, index=latent_code_all_mRNA_cnv_methyl_vae.index)
# fig_path_mRNA_CNV_metyl_tsne_VAE = saving_path + str(latent_code_dim) + 'tsne-VAE_mRNA_CNV_methyl.pdf'# for methylation 
# plot_scatter(latent_code_tsne_VAE3, molecular_subtype_481, colour_file1, latent_space_dimension, 'VAE', 'mRNA-CNV-methyl', fig_path_mRNA_CNV_metyl_tsne_VAE, label_type) #


# # ### gdc cohort-methylation only ###############################################
#LF_root_gdc = 'C:/Research/Python-DL-for-my-research/MMD_VAE4Omics/datasets/datasets_with_Latent_Features/latent_feautres_128_mono_omic/unsupervised/gdc_methylation_only/with_smote/'
# latent_code_all_methyl_vae = pd.read_csv(LF_root_gdc +'vae_methyl_smoted.tsv', sep='\t', header=0, index_col=0)
# latent_code_all_methyl_mmd = pd.read_csv(LF_root_gdc +'mmd_methly_smoted.tsv', sep='\t', header=0, index_col=0)
# z_methyl_VAE = tsne.fit_transform(latent_code_all_methyl_vae.values)
# z_methyl_MMD = tsne.fit_transform(latent_code_all_methyl_mmd.values)
# latent_code_tsne_vae = pd.DataFrame(z_methyl_VAE, index=latent_code_all_methyl_vae.index)
# latent_code_tsne_mmd = pd.DataFrame(z_methyl_MMD, index=latent_code_all_methyl_mmd.index)

# fig_path_methyl_tsne_VAE = saving_path + 'tsne_vae_smoted.pdf'# for methylation 
# plot_scatter(latent_code_tsne_vae, label_gdc, colour_file2, latent_space_dimension, 'VAE', 'methyl', fig_path_methyl_tsne_VAE, label_type)
# fig_path_methyl_tsne_MMD = saving_path + 'tsne_MMD_smoted.pdf'# for methylation 
# plot_scatter(latent_code_tsne_mmd, label_gdc, colour_file2, latent_space_dimension, 'MMD-VAE', 'methyl', fig_path_methyl_tsne_MMD, label_type)

# latent_code_tsne_MMD3 = pd.DataFrame(z_mRNA_CNV_methyl, index=latent_code_all_13.index)
# fig_path_mRNA_CNV_metyl_tsne_MMD = saving_path + str(latent_code_dim) + 'tsne-MMD_mRNA_CNV_methyl.png'# for methylation 
# plot_scatter_2d(latent_code_tsne_MMD3, label1_file, colour_file, latent_space_dimension, fig_path_mRNA_CNV_metyl_tsne_MMD) #


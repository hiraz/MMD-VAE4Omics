# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:29:21 2020
Plotting confusion matrix and saving into a given file
@author: Raz
"""

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def c_matrixPlotting(actuals_y, pred_y, method, omic_type, fig_path_cm):
    
    cm = confusion_matrix(actuals_y, pred_y)   
    #print(cm)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cmn)
    # need to change the index range based on the class numbers
    df_cm = pd.DataFrame(cmn, index = [i for i in "0123"],
                      columns = [i for i in "0123"])
    fig = plt.figure(figsize = (10,7))
    #fig.suptitle('Accuracy ='+ accuracy, fontsize=13, y=1.04 )
    
    sn.heatmap(df_cm, annot=True)
    plt.ylabel('True Molecular Subtype', fontsize=12)
    plt.xlabel('Predicted Molecular Subtype', fontsize=12)
    plt.title('Confusion matrix for '+ str(method) + ' based classification using ' + str(omic_type) + ' data' )
    plt.tight_layout()
    figure_name = str(method) + ' based classification using ' + str(omic_type) + ' data' 
    plt.savefig(fig_path_cm, dpi=300, bbox_inches='tight')
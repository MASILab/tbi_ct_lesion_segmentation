
# coding: utf-8

# In[4]:


import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.utils import shuffle
from sklearn import metrics
from time import time
import operator
from tqdm import *

from utils.pad import pad_image


# In[26]:


# loading datasets
THIS_COMPUTER = open("host_id.cfg").read().split()[0]

if THIS_COMPUTER == "vu":
    DATA_DIRS = [os.path.join("data", "test")]
    dataset_names = ["VUMC Data"]
elif THIS_COMPUTER == "nih":
    DATA_DIRS = [os.path.join("data", "test"),
                 os.path.join("data", "screening_umd"),
                 os.path.join("data", "screening_vcu")]
    dataset_names = ["NIH Data",
                     "Screening UMD",
                     "Screening VCU"]
sns.set(rc={'figure.figsize':(10, 7.5)})


# In[57]:


for DATA_DIR, dataset_name in zip(DATA_DIRS, dataset_names):
    
    PREPROCESSING_DIR = os.path.join(DATA_DIR, "preprocessed")
    SEG_ROOT_DIR_BASE = os.path.join(DATA_DIR, "segmentations")
    MODELS = [x for x in os.listdir(SEG_ROOT_DIR_BASE) if "val" in x]
    print(MODELS)
    
    plt.title('Reciever Operating Characteristic Curves for ' + dataset_name)
    scores_dict = {}
    legend_items = []

    colors = ['r', 'p', 'g', 'b', 'o']
    
    for i, model in enumerate(MODELS):
        
        if "msl_full" in model:
            model_name = "MSL Full"
        elif "msl_half" in model:
            model_name= "MSL 1/2 A"
        elif "msl_other" in model:
            model_name = "MSL 1/2 B"
        elif "nih_ssl" in model:
            model_name = "SSL NIH"
        elif "vu_ssl" in model:
            model_name = "SSL VUMC"

        SEG_ROOT_DIR = os.path.join(SEG_ROOT_DIR_BASE, model)


        pred_filenames = [os.path.join(SEG_ROOT_DIR, x) for x in os.listdir(SEG_ROOT_DIR) 
                      if not os.path.isdir(os.path.join(SEG_ROOT_DIR, x))]
        pred_filenames.sort()

        gt_filenames = [os.path.join(PREPROCESSING_DIR, x) for x in os.listdir(PREPROCESSING_DIR) 
                          if not os.path.isdir(os.path.join(PREPROCESSING_DIR, x))]
        gt_filenames = [x for x in gt_filenames if "mask" in x]
        gt_filenames.sort()
        
        pred_filenames = pred_filenames[:3]
        gt_filenames = gt_filenames[:3]
        
        x_gt_aggr = np.empty(shape=0)
        x_pred_aggr = np.empty(shape=0)
        x_thresh_aggr = np.empty(shape=0)

        # aggregate all volumes into one big volume
        for pred, gt in tqdm(zip(pred_filenames, gt_filenames), total=len(gt_filenames)):
            x = nib.load(pred).get_data()
            x_thresh = x.copy()
            x_thresh[np.where(x_thresh >= 0.5)] = 1
            x_thresh[np.where(x_thresh < 0.5)] = 0

            x_gt = nib.load(gt).get_data()

            print(x.shape, x_gt.shape)
            if x.shape != x_gt.shape:
                x_gt = pad_image(x_gt, target_dims=x.shape)
            print(x.shape, x_gt.shape)

            x_gt_aggr = np.append(x_gt_aggr, x_gt.flatten())
            x_pred_aggr = np.append(x_pred_aggr, x.flatten())
            x_thresh_aggr = np.append(x_thresh_aggr, x_thresh.flatten())


        fpr, tpr, thresholds = metrics.roc_curve(x_gt_aggr, x_pred_aggr)
        auc = metrics.auc(fpr,tpr)
        precision = metrics.precision_score(x_gt_aggr, x_thresh_aggr)
        
        scores_dict[model_name] = {"auc": auc,
                                   "precision": precision,
                                   "thresholds": thresholds}
        
        label = model_name + " AUC = {:02f}".format(auc)
        line, = plt.plot(fpr, tpr, color=colors[i], label=label)
        
        legend_items.append((model_name, line))


        
    # sort the legend
    legend_items.sort()
    plt.legend([x[1] for x in legend_items], [x[0] for x in legend_items])
    
    # plot the y=x line
    plt.plot([0,1],[0,1], 'k--')
    
    # set the range and domain of graph to display the x=0 values
    plt.xlim([-0.01,1])
    plt.ylim([-0.01,1])
    
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    #plt.show()
    
    plt.savefig("roc_curve_"+dataset_name.replace(" ", "_")+".png")
    plt.close()
    with open("precision_"+dataset_name.replace(" ", "_")+".txt", "w") as f:
        for k, v in scores_dict.items():
            f.write("{} Precision: {:.4f}\n".format(k, v['precision']))



# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# In[2]:


os.getcwd()


# In[3]:


results_dir = "results"
#results_dir = os.path.join("..", "results")


# In[4]:


os.listdir(results_dir)


# In[5]:


training_patterns = [os.path.join(results_dir, x) for x in os.listdir(results_dir)
                    if not os.path.isdir(x)]


# In[6]:


training_patterns


# In[7]:


result_files = [[os.path.join(training_pattern, x) for x in os.listdir(training_pattern) if "result" in x]
               for training_pattern in training_patterns]
result_files = [x for x in result_files if len(x) != 0 ]
result_files


# In[8]:


result_files[0]


# In[9]:


nih_results = [x for x in result_files[0] if "nih_ssl" in x][0]
vu_results = [x for x in result_files[0] if "vu_ssl" in x][0]
msl_full_results = [x for x in result_files[0] if "msl_full" in x][0]
msl_half_results = [x for x in result_files[0] if "msl_half" in x][0]
msl_other_half_results = [x for x in result_files[0] if "msl_other_half" in x][0]


# In[10]:


nih_df = pd.read_csv(nih_results)
vu_df = pd.read_csv(vu_results)
msl_full_df = pd.read_csv(msl_full_results)
msl_half_df = pd.read_csv(msl_half_results)
msl_other_half_df = pd.read_csv(msl_other_half_results)


# In[11]:


weight_site="Training Location"
nih_df = nih_df.assign(weight_site="NIH")
vu_df = vu_df.assign(weight_site="VUMC")
msl_full_df = msl_full_df.assign(weight_site='MSL Full')
msl_half_df = msl_half_df.assign(weight_site='MSL 1/2 A')
msl_other_half_df = msl_other_half_df.assign(weight_site='MSL 1/2 B')


# In[12]:


merged = pd.concat([nih_df, vu_df, msl_full_df, msl_half_df, msl_other_half_df], 
                   keys=['NIH', 'VUMC', 'MSL Full', 'MSL 1/2 A', 'MSL 1/2 B'])


# In[13]:


merged.columns = ['filename', 'Dice Coefficient', 'thresholded volume(mm)',
       'thresholded volume ground truth(mm)',
       'largest hematoma ground truth(mm)', 'largest hematoma prediction(mm)',
       'severe hematoma ground truth', 'severe hematoma pred', 'vox dim 1(mm)',
       'vox dim 2(mm)', 'vox dim 3(mm)', 'probability vol(mm)',
       'probability volume(voxels)', 'thresholded volume(voxels)',
       'Training Location']


# In[14]:


merged


# In[28]:


sns.set(rc={'figure.figsize':(10,7.5)})

ax = sns.boxplot(x="Training Location", y="Dice Coefficient", data=merged, palette="tab10", saturation=0.75
                #).set_title("Dice Scores on NIH Data"
                ).set_title("Dice Scores on VU Data"
                )
fig = ax.get_figure()
plt.ylim(0, 1.10)


x1, x2 = 0, 2   # columns 'nih' and 'multi' (first column: 0, see plt.xticks())
y, h, col = merged['Dice Coefficient'].max() + 0.05, 0.05, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

x1, x2 = 1, 2   # columns 'nih' and 'multi' (first column: 0, see plt.xticks())
y, h, col = merged['Dice Coefficient'].max() + 0.0175, 0.025, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "ns", ha='center', va='bottom', color=col)

fig.savefig("vu_data_boxplot.png")
#fig.savefig("nih_data_boxplot.png")


# In[16]:


from scipy.stats import wilcoxon


# In[21]:


print("Wilcoxon between NIH weights, Multi weights")
print(wilcoxon(nih_df['dice'], msl_full_df['dice']))
print("Wilcoxon between NIH weights, MSL 1/2 A weights")
print(wilcoxon(nih_df['dice'], msl_half_df['dice']))
print("Wilcoxon between NIH weights, MSL 1/2 B weights")
print(wilcoxon(nih_df['dice'], msl_other_half_df['dice']))


# In[22]:


print("Wilcoxon between VU weights, Multi weights")
print(wilcoxon(vu_df['dice'], msl_full_df['dice']))
print("Wilcoxon between VU weights, MSL 1/2 A weights")
print(wilcoxon(vu_df['dice'], msl_half_df['dice']))
print("Wilcoxon between VU weights, MSL 1/2 B weights")
print(wilcoxon(vu_df['dice'], msl_other_half_df['dice']))


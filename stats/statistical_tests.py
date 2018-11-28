
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys


# # This is all with NIH data.

# In[2]:


results_dir = "results"


# In[3]:


os.listdir(results_dir)


# In[4]:


training_patterns = [os.path.join(results_dir, x) for x in os.listdir(results_dir)
                    if not os.path.isdir(x)]


# In[8]:


result_files = [[os.path.join(training_pattern, x) for x in os.listdir(training_pattern) if "result" in x]
               for training_pattern in training_patterns]

result_files = result_files[1:]
result_files


# In[12]:


nih_results = [x[0] for x in result_files if "nih" in x[0]][0]
vu_results = [x[0] for x in result_files if "vu" in x[0]][0]
multi_results = [x[0] for x in result_files if "multi" in x[0]][0]


# In[13]:


nih_df = pd.read_csv(nih_results)
vu_df = pd.read_csv(vu_results)
multi_df = pd.read_csv(multi_results)


# In[14]:


nih_df = nih_df.assign(weight_site="nih")
vu_df = vu_df.assign(weight_site="vu")
multi_df = multi_df.assign(weight_site="multi")


# In[16]:


merged = pd.concat([nih_df, vu_df, multi_df], keys=['nih', 'vu', 'multi'])


# In[17]:


merged


# In[18]:


ax = sns.boxplot(x="weight_site", y="dice", data=merged
                ).set_title("Dice Scores on NIH Data with Different Training"
                )
fig = ax.get_figure()
fig.savefig("nih_data_boxplog.png")


# In[ ]:


# This will be the best way to show the Dice scores as a box plot in the end
#ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
#                 data=tips, palette="Set3")


# In[19]:


from scipy.stats import wilcoxon


# In[20]:


print("Wilcoxon between NIH weights, Multi weights")
print(wilcoxon(nih_df['dice'], multi_df['dice']))


# In[21]:


print("Wilcoxon between VU weights, Multi weights")
print(wilcoxon(vu_df['dice'], multi_df['dice']))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np

import scipy as sp
from scipy import stats
from numpy.matlib import repmat 
from scipy.linalg import svd, schur

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from matplotlib.ticker import FormatStrFormatter


# Functions

# In[2]:


def node_strength(A):
    s = np.sum(A, axis = 0)

    return s

def node_degree(A):
    B = A > 0
    k = np.sum(B, axis = 0)
    
    return k

def ave_control(A, c = 1):
    # Bassett Lab, University of Pennsylvania, 2016.
    # Reference: Gu, Pasqualetti, Cieslak, Telesford, Yu, Kahn, Medaglia,
    #            Vettel, Miller, Grafton & Bassett, Nature Communications
    #            6:8414, 2015.

    if c is not None:
        u, s, vt = svd(A) # singluar value decomposition
        A = A/(c + s[0]) # Matrix normalization 

    T, U = schur(A,'real') # Schur stability
    midMat = np.multiply(U,U).transpose()
    v = np.matrix(np.diag(T)).transpose()
    N = A.shape[0]
    P = np.diag(1 - np.matmul(v,v.transpose()))
    P = repmat(P.reshape([N,1]), 1, N)
    ac = sum(np.divide(midMat,P))
    
    return ac


# In[3]:


# output directory
figdir = '/Users/lindenmp/Dropbox/Work/ResProjects/neurodev_cs_predictive/analysis/t1Exclude/schaefer_200_streamlineCount/figs'
os.chdir(figdir)


# In[4]:


# set seaborn figure style
sns.set(style='white', context = 'talk', font_scale = 1)


# ### Generate toy network

# In[5]:


num_nodes = 5
np.random.seed(5)

A = np.random.randint(0, 50, size = [num_nodes,num_nodes])
A = A + A.transpose()
A[np.eye(num_nodes) == 1] = 0

A


# In[6]:


f, ax = plt.subplots()
f.set_figwidth(7.5)
f.set_figheight(5)
sns.heatmap(A, center = 0, ax = ax)


# ### Calculate strength, degree, and average control on toy network

# In[7]:


node_strength(A)[0]


# In[8]:


node_degree(A)[0]


# In[9]:


ave_control(A)[0]


# ### Run simple simulation of the linear system to illustrate the impulse response

# In[10]:


# Normalize A matrix and plot
c = 1 # designate scaling factor
u, s, vt = svd(A) # SVD
# A_norm = (A/s[0])*c # Matrix normalization
A_norm = A/(c + s[0]) # Matrix normalization 

print(A_norm)
print(ave_control(A_norm, c = None)[0])


# In[11]:


f, ax = plt.subplots()
f.set_figwidth(7.5)
f.set_figheight(5)
sns.heatmap(A_norm, center = 0, ax = ax)


# In[12]:


# Simulate linear system
num_steps = 10
x = np.zeros((num_nodes,num_steps))
x[0,0] = 1 # set region 0 to be 'on' at T 0
for t in np.arange(1,num_steps): # run simulation of linear system
    x[:,t] = np.dot(A_norm,x[:,t-1])

f, ax = plt.subplots()
f.set_figwidth(7.5)
f.set_figheight(5)
sns.heatmap(x, ax = ax, cmap = 'YlGnBu_r', cbar_kws={'label': 'Activity'})
ax.set_xlabel('T')
ax.set_ylabel('Node')
f.savefig('full_toy.svg', dpi = 300, bbox_inches = 'tight')


# ### Lesion A matrix and re do simulation

# Here I lesion each non-local non-zero connection in ascending order and plot average controllability

# In[13]:


# Find the non-local non-zero connections
A_norm_tmp = A_norm.copy()

A_norm_tmp[0,:] = 0
A_norm_tmp[:,0] = 0
A_norm_tmp = np.triu(A_norm_tmp)
A_norm_tmp


# In[14]:


lesion_indices = np.nonzero(A_norm_tmp)
lesion_indices


# In[15]:


# Lesion each non-local non-zero connection and calculate average controllability
ac_lesioned = np.zeros(len(lesion_indices[0]))

for i in np.arange(0,len(lesion_indices[0])):
    A_lesioned = A_norm.copy()
    A_lesioned[lesion_indices[1][i],lesion_indices[0][i]] = 0
    A_lesioned[lesion_indices[0][i],lesion_indices[1][i]] = 0
    ac_lesioned[i] = ave_control(A_lesioned, c = None)[0]


# In[16]:


sort_idx = np.argsort(A_norm_tmp[lesion_indices])
sort_idx


# In[17]:


print(lesion_indices[0][sort_idx])
print(lesion_indices[1][sort_idx])


# In[18]:


# Plot
x_labels = list()

for i in np.arange(0,len(sort_idx)):
    x_labels.append('(' + str(lesion_indices[0][sort_idx][i]) + ',' + str(lesion_indices[1][sort_idx][i]) + ')')

sns.set(style='white', context = 'paper', font_scale = 1)

f, ax = plt.subplots()
f.set_figwidth(2.5)
f.set_figheight(1.5)

ax.bar(x = np.arange(0,len(lesion_indices[0])), height = ac_lesioned[np.argsort(A_norm_tmp[lesion_indices])], color = 'w', edgecolor = 'k', linewidth = 2)
ax.set_xlabel('Lesioned Edge')
ax.set_ylabel('Average Controllability')
ax.set_xticks(np.arange(0,len(lesion_indices[0])))
ax.set_xticklabels(x_labels)
ax.tick_params(pad = -3)
f.savefig('lesioned_ac.svg', dpi = 300, bbox_inches = 'tight')


# In[19]:


# Just for illustrative purposes, lesion a bunch of non-local connections and run the simulation
A_lesioned = A_norm.copy()
A_lesioned[2,1] = 0
A_lesioned[1,2] = 0
A_lesioned[2,3] = 0
A_lesioned[3,2] = 0
A_lesioned[3,4] = 0
A_lesioned[4,3] = 0

ave_control(A_lesioned, c = None)[0]


# In[20]:


f, ax = plt.subplots()
f.set_figwidth(7.5)
f.set_figheight(5)
sns.heatmap(A_lesioned, center = 0, ax = ax)


# In[21]:


# Simulate linear system
x = np.zeros((num_nodes,num_steps))
x[0,0] = 1 # set region 0 to be 'on' at T 0
for t in np.arange(1,num_steps): # run simulation of linear system
    x[:,t] = np.dot(A_lesioned,x[:,t-1])

f, ax = plt.subplots()
f.set_figwidth(7.5)
f.set_figheight(5)
sns.heatmap(x, ax = ax, cmap = 'YlGnBu_r', cbar_kws={'label': 'Activity'})
ax.set_xlabel('T')
ax.set_ylabel('Node')
f.savefig('lesioned_toy.svg', dpi = 300, bbox_inches = 'tight')


# # Manipulating C

# The normalization constant (c) determines the scaling of the eigenvalues in the system. By convention we set to 1 so that all modes decay over time. By increasing c we force all the system's modes to decay more quickly and thus shorten the life of impulse response. Here, I plot the activity spread from a ROI as a function of c (columns). The bottom show is the average activity at each timepoint over regions.

# In[22]:


sns.set(style='white', context = 'paper', font_scale = 0.75)

f, ax = plt.subplots(2,5)
f.set_figwidth(7.25)
f.set_figheight(2)
plt.subplots_adjust(wspace = 0.8, hspace = 0.5)

u, s, vt = svd(A) # SVD

for i, c in enumerate([1, 10, 10**2, 10**3, 10**4]):
    A_norm = A/(c + s[0]) # Matrix normalization
#     print(ave_control(A, c = c)[0])

    x = np.zeros((num_nodes,num_steps))
    x[0,0] = 1 # set region 0 to be 'on' at T 0
    for t in np.arange(1,num_steps): # run simulation of linear system
        x[:,t] = np.dot(A_norm,x[:,t-1])

    if i == 0:
        pos1 = ax[0,i].get_position() # get the original position 
        pos2 = [pos1.x0 + 0.1, pos1.y0,  pos1.width / 8, pos1.height] 
        cbar_ax = f.add_axes(pos2)
        
        sns.heatmap(x, ax = ax[0,i], cmap = 'YlGnBu_r', cbar_ax = cbar_ax, cbar_kws={'label': 'Activity'})
        ax[0,i].set_ylabel('Node')
    else:
        sns.heatmap(x, ax = ax[0,i], cmap = 'YlGnBu_r',  cbar = False)
    
    ax[0,i].set_title('c: '+str(c) + '\n ac: ' + str(np.round(ave_control(A, c = c)[0],2)))
    ax[0,i].tick_params(pad = -3)

    ax[1,i].plot(x.mean(axis = 0))
    ax[1,i].set_ylim([-0.01,0.25])
    ax[1,i].set_xlabel('Timepoint')
    ax[1,i].set_xticks(np.arange(0, x.shape[1], 2))
    ax[1,i].tick_params(pad = -3)
    
    if i == 0:
        ax[1,i].set_ylabel('Activity (average)')
    
f.savefig('full_toy_overc.svg', dpi = 300, bbox_inches = 'tight')


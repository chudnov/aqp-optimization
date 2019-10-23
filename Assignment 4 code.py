#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import matplotlib.pyplot as plt


# In[ ]:


#creates sample table of specified size
def create_sample_table (nrow, data):
    sample_table = np.empty(shape=(nrow, data.shape[1]))
    for c in range (0,sample_table.shape[1]):
        sample_table[:,c] = np.random.choice (data[:,c], nrow, replace = False)
    return sample_table


# In[ ]:


#get the MSE for one dataset, aggregate, and sample table size
def MSE_one_example (sample_table_nrows, data, aggregate, quantile):
    sample_table = create_sample_table (sample_table_nrows, data)
    MSE = 0
    for c in range (0,sample_table.shape[1]):
        if (quantile != 0):
            actual = aggregate(data[:,c], quantile)
            sample = aggregate(sample_table[:,c], quantile)
            MSE += (actual-sample)**2
        else:
            actual = aggregate(data[:,c])
            sample = aggregate(sample_table[:,c])
            MSE += (actual-sample)**2 
    MSE = MSE/sample_table.shape[1]
    return MSE


# In[ ]:


#takes in list of dataset sizes, list of aggregates, list of sample table sizes and plots MSE vs. sample table
#size for each dataset
#***dataset is ~N(0,1)
#@param dataset_sizes (list of tuples)
#@param aggregates (list of numpy aggregates on one column)
#@param sample_table_sizes (list of numbers representing ratios of sample table size to data table size)
def eval_random_sampling_MSE (dataset_sizes, aggregates, sample_table_sizes):
    for dataset_size in dataset_sizes:
        data = np.random.normal(loc = 0, scale = 1, size=dataset_size)
        for aggregate in aggregates:
            if (aggregate == np.quantile):
                for i in [.1, .25, .5, .75]:
                    aggregate_mse = []
                    for sample_table_ratio in sample_table_sizes:
                        aggregate_mse.append(MSE_one_example (round(sample_table_ratio*data.shape[0]), data, aggregate, quantile = i))
                    print ("***New Example***")
                    print ("dataset size: ")
                    print(dataset_size)
                    print ("aggregate: ")
                    print(aggregate)
                    print ("quantile: ")
                    print (i)
                    print ("sample table sizes: ")
                    print(sample_table_sizes)
                    print ("MSE values: ")
                    print(aggregate_mse)
                    plt.plot (sample_table_sizes, aggregate_mse)
                    plt.pause(0.05)
                    print ("********")
            else:
                aggregate_mse = []
                for sample_table_ratio in sample_table_sizes:
                    aggregate_mse.append(MSE_one_example (round(sample_table_ratio*data.shape[0]), data, aggregate, quantile = 0))
                print ("***New Example***")
                print ("dataset size: ")
                print(dataset_size)
                print ("aggregate: ")
                print(aggregate)
                print ("sample table sizes: ")
                print(sample_table_sizes)
                print ("MSE values: ")
                print(aggregate_mse)
                plt.plot (sample_table_sizes, aggregate_mse)
                plt.pause(0.05)
                print ("********")
    plt.show()


# In[ ]:


#Test on some examples
dataset_sizes = [(100, 100), (250, 250), (500, 500), (1000, 1000), (2500, 2500), (5000, 5000)]
aggregates = [np.mean, np.std, np.amax, np.quantile]
sample_table_sizes = [.01, .05, .1, .25, .5, .75, .9]


# In[ ]:


eval_random_sampling_MSE (dataset_sizes, aggregates, sample_table_sizes)


# In[ ]:


#test on categorical dataset.  Cat1 = 0:.25, Cat2 = .25:.5, Cat3 = . .5:.75, Cat4 =.75:1 

#Save results
sum_one_category_MSE = []

sample_table = create_sample_table (900, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)

sample_table = create_sample_table (750, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)

sample_table = create_sample_table (500, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)


sample_table = create_sample_table (250, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)


sample_table = create_sample_table (100, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)


sample_table = create_sample_table (50, data_categorical)
MSE = 0
for c in range (0,sample_table.shape[1]):
    num_Cat1_sample = 0
    num_Cat1_actual = 0
    for i in range (0, sample_table.shape[0]):
        if (sample_table[i,c] <= .25):
            num_Cat1_sample += 1
    for i in range (0, data_categorical.shape[0]):
        if (data_categorical[i,c] <= .25):
            num_Cat1_actual += 1
    sample = num_Cat1_sample/sample_table.shape[0]
    actual = num_Cat1_actual/data_categorical.shape[0]
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
sum_one_category_MSE.append(MSE)
   
plt.plot ([900, 750, 500, 250, 100, 50], sum_one_category_MSE)   
   
   


# In[8]:


#show effect of outliers  - it's minimal to none when they're distributed like this
data_outliers = np.random.normal(loc = 0, scale = 1, size=(1000,1000))
for c in range (0, data_outliers.shape[1]):
    for r in range (0, data_outliers.shape[0]):
        if (r%200 == 0):
            data_outliers[r, c] += 100


# In[10]:


#Save results
outliers_MSE = []

sample_table = create_sample_table (900, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

sample_table = create_sample_table (750, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

sample_table = create_sample_table (500, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

sample_table = create_sample_table (250, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

sample_table = create_sample_table (100, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

sample_table = create_sample_table (50, data_outliers)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
outliers_MSE.append(MSE)

plt.plot ([900, 750, 500, 250, 100, 50], outliers_MSE)   


# In[13]:


#compare outliers to no outliers
data = np.random.normal(loc = 0, scale = 1, size=(1000,1000))

#Save results
MSE_vals = []

sample_table = create_sample_table (900, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (750, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (500, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (250, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (100, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (50, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

plt.plot ([900, 750, 500, 250, 100, 50], MSE_vals)   


# In[14]:


##compare high stdev to low stdev outliers
data = np.random.normal(loc = 0, scale = 100, size=(1000,1000))

#Save results
MSE_vals = []

sample_table = create_sample_table (900, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (750, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (500, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (250, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (100, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

sample_table = create_sample_table (50, data)
MSE = 0
for c in range (0,sample_table.shape[1]):
    actual = np.quantile(data_outliers[:,c], .5)
    sample = np.quantile(sample_table[:,c], .5)
    MSE += (actual-sample)**2
MSE = MSE/sample_table.shape[1]
MSE_vals.append(MSE)

plt.plot ([900, 750, 500, 250, 100, 50], MSE_vals)   


# In[ ]:


#can't provide guaruntees/confidence withotu knowing underlyign distirbution - here stdev scales error


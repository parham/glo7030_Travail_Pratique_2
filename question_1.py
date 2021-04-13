
#%%[markdown]
# ![Université Laval](img/ulaval.jpg)
# # <center><b>GLO-4030/GLO-7030 : Apprentissage par réseaux de neurones profonds</b></center>
# # <center><b>Travail Pratique 2</b></center>
# ***
# __Course__: GLO-4030/GLO-7030 : Apprentissage par réseaux de neurones profonds <br>
# __Title__: Travail Pratique 2 <br>
# __Semester__: Winter 2021 <br>
# __Lecturer__: Dr. Pascal Germain <br>
# __Author__: Parham Nooralishahi <br>
# __Organization__: Université Laval <br>
# ***
# 
# ## Question 1 - Fine-tuning and Normalization (50%)

#%%
import os
import parham_core as phm
from torchviz import make_dot

import torchvision.models as models
import torchvision.datasets as ds
import torchvision.transforms as trans


#%%[markdown]
# |__Directory__|__Path__|
# |----------|-----|
# |__Dataset__|./datasets|
# |__Used Data__|./data_split|
# |__Images for the report__| ./img|

#%%[markdown]
# ## Dataset Preparation & Processing
# In this question, you have to perform fine-grained classification of bird species. To do so, download the images of the dataset <a href="http://www.vision.caltech.edu/visipedia/CUB-200.html">CUB-200</a> <br\>.

# ### CUB-200 Dataset
# Caltech-UCSD Birds 200 (CUB-200) is an image dataset with photos of 200 bird species (mostly North American).
# * __Number Of Categories__: 200
# * __Number Of Images__: 6,033
# * __Annotations__: Bounding Box, Rough Segmentation, Attributes
# 
# @techreport{WelinderEtal2010, <br/>
# &emsp;&emsp; Author = {P. Welinder and S. Branson and T. Mita and C. Wah and F. Schroff and S. Belongie and P. Perona}, <br/>
# &emsp;&emsp; Institution = {California Institute of Technology}, <br/>
# &emsp;&emsp; Number = {CNS-TR-2010-001}, <br/>
# &emsp;&emsp; Title = {{Caltech-UCSD Birds 200}}, <br/>
# &emsp;&emsp; Year = {2010} <br/>
# } <br/>
# 
# ![CUB-200 Dataset](img/cub200.jpg)
# 
# ### Download and Unzip CUB-200 Dataset
# 

#%%
orig_dataset_dir = 'orig_dataset'
orig_dataset_dir = phm.get_cub200(orig_dataset_dir)

#%%[markdown]
# ### Data Preparation (Training & Testing Subset)
# For each class, we sort the images in ascending order by file name and use the first 15 images as a test set. Then, we use the other images for training. 

#%%
dataset_dir='dataset'
training_ds_dir, testing_ds_dir = phm.prepare_subsets(orig_dataset_dir, dataset_dir)

#%%[markdown]
# ### Data Normalization
# Experiments are conducted twice with two different data normalization strategies. The first time, use the values from the training dataset to normalize the data. The second time, use the following values that were used for the ImageNet training :
# 
# ||__R__|__G__|__B__|
# |------|-------|-------|-------|
# | __Mean__ | 0.485 | 0.456 | 0.406 |
# | __Std__ | 0.229 | 0.224 | 0.225 |

# #### Data Normalization using the values of the training dataset
# As mentioned, for the first data normalization strategy, the values calculated based on the training dataset.

#%%
mean_stg_1, std_stg_1 = phm.calculate_normalization_params(training_ds_dir)

print('Training Dataset Calculated Normalization parameters:')
print('Mean: %s' % str(mean_stg_1))
print('Std: %s' % str(std_stg_1))

norm_stg1_transform = trans.Compose([
    trans.Resize((224,224)),
    trans.ToTensor(),
    trans.Normalize(
        mean_stg_1, 
        std_stg_1
    )
])

norm_stg2_transform = trans.Compose([
    trans.Resize((224,224)),
    trans.ToTensor(),
    trans.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

#%%[markdown]
# ## Neural Network Model Definition

#%%
print('Define Neural network model ...')
model = phm.create_model()
print(model)

_, data_loader = phm.get_datasets(training_ds_dir, norm_stg1_transform)

sample_data = next(iter(data_loader))
sample_data = sample_data[0].cuda() if phm.is_cuda_available() else sample_data[0]
out_test = model(sample_data)
make_dot(out_test.mean(), params=dict(model.named_parameters()))

#%%[markdown]
# The figure is a presentation of the model. If a node represents a backward function, it is gray. Otherwise, the node represents a tensor and is either blue, orange, or green. Blue: reachable leaf tensors that requires grad (tensors whose .grad fields will be populated during .backward()) Orange: saved tensors of custom autograd functions as well as those saved by built-in backward nodes Green: tensor passed in as outputs Dark green: if any output is a view, we represent its base tensor with a dark green node. <br>
# 
# |__Parameters__|__Values__|
# |--------------|----------|
# |__Epoch Number__|20|
# |__Batch Size__|68|
# |__Learning Rate__|0.1|
# |__Momentum__|0.9|

# ## Question 1-a : Using the default random initialization
# ### (a) Normalization using the values from dataset

#%% 

epoch = 31
batch_size = 68
learning_rate = 0.1
momentum = 0.9

# Create Models
model = phm.create_model(pretrained=False)

phm.run_experiment('q1_a_norm_data',
    model,
    norm_stg1_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]
# ### (b) Normalization using ImageNet values

#%% 

# Create Models
model = phm.create_model(pretrained=False)

phm.run_experiment('q1_a_norm_imagenet',
    model,
    norm_stg2_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]

# ## Question 1-b : Using the pre-trained model, but freezing all convolution parameters.
# ### (a) Normalization using the values from dataset

#%%
model = phm.create_model(pretrained=True)
model = phm.freeze_conv_params(model)

phm.run_experiment('q1_b_norm_data',
    model,
    norm_stg1_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]
# ### (b) Normalization using ImageNet values

#%%
model = phm.create_model(pretrained=True)
model = phm.freeze_conv_params(model)

phm.run_experiment('q1_b_norm_imagenet',
    model,
    norm_stg2_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]

# ## Question 1-c : Using the pre-trained model, but only freezing the parameters in "layer1".
# ### (a) Normalization using the values from dataset

#%%
model = phm.create_model(pretrained=True)
model = phm.freeze_layer1_params(model)

phm.run_experiment('q1_c_norm_data',
    model,
    norm_stg1_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]
# ### (b) Normalization using ImageNet values

#%%
model = phm.create_model(pretrained=True)
model = phm.freeze_layer1_params(model)

phm.run_experiment('q1_c_norm_imagenet',
    model,
    norm_stg2_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]

# ## Question 1-d : Using the pre-trained model, but letting all the parameters (including the convolution layers) be adjusted by backprop.

# ### (a) Normalization using the values from dataset

model = phm.create_model(pretrained=True)

phm.run_experiment('q1_d_norm_data',
    model,
    norm_stg1_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)

#%%[markdown]
# ### (b) Normalization using ImageNet values

#%%
model = phm.create_model(pretrained=True)

phm.run_experiment('q1_d_norm_imagenet',
    model,
    norm_stg2_transform, 
    training_dir=training_ds_dir,
    testing_dir=testing_ds_dir,
    learning_rate=learning_rate,
    batch_size=batch_size,
    num_epochs=epoch,
    momentum=momentum
)
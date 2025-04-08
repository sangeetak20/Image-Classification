#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
from textwrap import wrap
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


classifications = pd.read_csv('training_classifications.csv')

training_images_path = '/Users/iisan1/Downloads/training_images'
training_images = os.listdir(training_images_path)

#only run once 
#this removes the galaxy ID as a label 
labels = list(classifications.columns)
labels_noID = labels[1:]

#need this because otherwise the generator will select the galaxy ID as the highest value
classifications_noID = classifications.drop('GalaxyID', axis = 1)

#csv that is indexed by the galaxy ID
ID = pd.read_csv('training_classifications.csv', index_col = 'GalaxyID')

#from this generator we are looping through each of the galaxies and yielding their highest label value and label 
def image_batch(batch_no, batch_size): 
    
    #initializing dictionary 
    stored_galaxy = {} 
    resized_path = r'/Users/iisan1/Downloads/resized_images/'
    
    #takes in int
    label_arrays = np.zeros((batch_size, 37), dtype = np.float32)
    tensor_image_array = np.zeros((batch_size, 3, 77, 77), dtype = np.float32)
    galaxies = np.zeros((batch_size))
    i = 0
    for image in batch_no: 
        i = int(i)
        #getting current galaxy and adding to array
        current_galaxy = int(image.split('.jpg')[0])
        galaxies[i] = current_galaxy
        
        #creating array of label values for current galaxy 
        label_array = np.array(np.array(ID.loc[current_galaxy]))
        label_arrays[i] = label_array
        
        #getting image and converting to tensor and adding to array 
        img = Image.open(resized_path + str(current_galaxy) + '_resized.jpg')
        convert_tensor = transforms.ToTensor()
        tensor_image = convert_tensor(img)
        tensor_image_array[i] = tensor_image
        
        i+= 1
        
    #converting array of images into tensor  
    tensor_image_array = torch.from_numpy(tensor_image_array)
    
    #converting label array to tensor 
    label_arrays = torch.from_numpy(label_arrays)
    
    yield label_arrays, tensor_image_array, galaxies
    
#creating a list of indexes for batches 
batches = np.arange(0, len(classifications), step = 200)
batches = np.append(batches, len(classifications)%200 + batches[-1])

#looping through the entire classifications array in batches of 200 
for i in range(len(batches)-1): 
    batch = np.arange(batches[i], batches[i+1])
    yielded_labels = image_batch(batch)
    
#creating training and cross validation (testing) sets 
images = training_images
labels = list(classifications.columns)
image_train, image_test, label_train, label_test = \
train_test_split(images, classifications, test_size = 0.20, train_size = 0.80)

def mean_square_error(y_true, y_pred):
    return tf.keras.backend.square(y_true-y_pred)

#creates array for average values per label 
average_labels = [] 
for i in labels_noID: 
    average_labels.append(np.average(label_train[i]))
avg_test_labels = []
for i in labels_noID: 
    avg_test_labels.append(np.average(label_test[i]))

#creating matrix for label array to input into LMSE function
def label_matrix(galaxy_ID, arr):
    
    #takes in galaxy ID and finds index in the csv file that matches 
    galaxy_index = np.where(arr['GalaxyID'] == galaxy_ID)[0][0]
    
    #gets the row of the galaxy ID (this does NOT include the galaxy ID)
    row = label_train[galaxy_index : galaxy_index+1]
    
    #creates an array of values 
    label_array = np.array(row)[0][1:]
    #print(len(label_array))
    
    return label_array

#getting LMSE values for training set 
lmse_val = []
mean = 0
testing_ID = np.array(label_test['GalaxyID'])
for i in testing_ID: 
    galaxy_ID = i
    y_true = label_matrix(galaxy_ID, label_test)
    y_pred = avg_test_labels
    mean += mean_square_error(y_true, average_labels)
mean = tf.keras.backend.mean(mean, axis=-1)
np.sqrt(mean)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First 2D convolutional layer, taking in 1 input channel (image),
        # outputting 32 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 5), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        # Second 2D convolutional layer, taking in the 32 input layers,
        # outputting 64 convolutional features, with a square kernel size of 3
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), 
                                   nn.ReLU(), 
                                   nn.MaxPool2d(2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3), 
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        
#         self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3), 
#                                    nn.ReLU(),
#                                    nn.MaxPool2d(2))

        # Designed to ensure that adjacent pixels are either all 0s or all active
        # with an input probability
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.dropout3 = nn.Dropout2d(0.1)
        self.dropout4 = nn.Dropout2d(0.1)

        #First fully connected layer
        self.fc1 = nn.Linear(6272, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 37)
#         #Fourth fully connected layer that outputs our 37 labels
        
        # x represents our data
    def forward(self, x):
        #FIRST LAYER 
        # Pass data through conv1
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        #flatten to make 1D
        x = torch.flatten(x, 1)
        
        # Pass data through ``fc1``
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        
        output = torch.sigmoid(x)
        return output
def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

# Instantiate the model
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay =0.0001)
loss_history = []

nn_images_validation = list(divide_chunks(image_test, 150))
nn_images_train = list(divide_chunks(image_train, 150))

loss_history_validation = []

epochs = 15
for _ in range(epochs): 
    #This is for the training 
    model.train()
    for batch in range(len(nn_images_train)): 
        
        runningloss = []
        #from generator pull the images and labels for each image 
        current_batch = nn_images_train[batch]
        #generator returns tensor labels, tensor images, and galaxy names (not used)
        generator_function= next(image_batch(current_batch, len(current_batch)))
        
        
        for i in range(len(current_batch)):
            optimizer.zero_grad()
            #getting tensor from generator function 
            labels_tensor = generator_function[0][i]
            labels_tensor= labels_tensor.reshape(1, 37)
            #getting images from generator function and reshaping
            inputs = generator_function[1][i]
            inputs = inputs.reshape(1, 3, 77, 77)


            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.reshape(1, 37)
            loss = torch.sqrt(criterion(labels_tensor, outputs))
            loss.backward()
            optimizer.step()

            # get loss
            runningloss.append(loss.item())
    
        loss_history.append(np.array(runningloss).mean())
    
    #this is for the validation 
    running_loss = 0.0
    model.eval()
    runningloss_validation = [] 
    with torch.no_grad(): 
        for batch in range(len(nn_images_validation)):

            #from generator pull the images and labels for each image 
            current_batch = nn_images_validation[batch]
            generator_function= next(image_batch(current_batch, len(current_batch)))


            for i in range(len(current_batch)):
                labels_tensor = generator_function[0][i]
                labels_tensor = labels_tensor.reshape(1, 37)
                inputs = generator_function[1][i]
                inputs = inputs.reshape(1, 3, 77, 77)
                #print("inputs",inputs.shape)


                # forward + backward + optimize
                outputs = my_nn(inputs)
                outputs = outputs.reshape(1, 37)
                #print("outputs:", outputs.shape)
                loss = torch.sqrt(criterion(labels_tensor, outputs))

                # print statistics
                running_loss += loss.item()
                runningloss_validation.append(loss.item())

    loss_history_validation.append(np.array(runningloss_validation).mean())
    
    
# print(running_loss/len(image_train))

torch.save(model.state_dict, r'/Users/iisan1/Downloads/Lab3')

import torchvision.models as models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
    model.fc = nn.Sequential(nn.Flatten(),
    nn.Linear(2048, 128),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(128, 37),
    nn.Sigmoid())
    #loss_fn = torch.sqrt(criterion(prediction, y))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay =0.0001)
    return model.to(device), optimizer

def train_batch(x, y, model, opt):
    model.train()
    prediction = model(x)
    batch_loss = torch.sqrt(criterion(prediction, y))
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item(), prediction
def validation_batch(x, y, model):
    model.eval()
    prediction = model(x)
    return prediction

train_epoch_losses = []
val_epoch_losses = []
train_losses = []
model, optimizer = get_model()
for epoch in range(10):
    #training
    model.train()
    #prediction_train = []
    for batch in range(len(nn_images_train)): 
        
        #from generator pull the images and labels for each image 
        current_batch = nn_images_train[batch]
        generator_function= next(image_batch(current_batch, len(current_batch)))
        
        for i in range(len(current_batch)):
            labels_tensor = generator_function[0][i]
            labels_tensor = labels_tensor.reshape(1, 37)
            inputs = generator_function[1][i]
            inputs = inputs.reshape(1, 3, 77, 77)
            
            output = train_batch(inputs, labels_tensor, model, optimizer)
            
            batch_loss = output[0]
            train_losses.append(batch_loss)

        #getting the average for this epoch and adding it to array
        train_epoch_loss = np.array(train_losses).mean()
        train_epoch_losses.append(train_epoch_loss)

    #validation 
    model.eval()
    val_loss = []
    prediction_validation = []
    with torch.no_grad():
        for batch in range(len(nn_images_validation)): 

            #from generator pull the images and labels for each image 
            current_batch = nn_images_validation[batch]
            generator_function= next(image_batch(current_batch, len(current_batch)))

            for i in range(len(current_batch)):
                labels_tensor = generator_function[0][i]
                labels_tensor = labels_tensor.reshape(1, 37)
                inputs = generator_function[1][i]
                inputs = inputs.reshape(1, 3, 77, 77)

                output = validation_batch(inputs, labels_tensor, model)
                output = output.reshape(1, 37)

                batch_loss = criterion(output, labels_tensor)
                val_loss.append(batch_loss) 
                prediction_validation.append(output)
            
    #getting the average for this epoch and adding it to array         
    val_epoch_loss = np.array(val_loss).mean()
    val_epoch_losses.append(val_epoch_loss)

resnetloss_CNN = np.zeros(len(train_epoch_losses))
epochh  = 0
for i in range(len(resnetloss_CNN)): 
    print(epochh)
    if i%329 == 0 and i!= 0: 
        resnetloss_CNN[i] = val_epoch_losses[epochh]
        epochh += 1
    resnetloss_CNN[i] = val_epoch_losses[epochh]
#for prediction array
def highest_val(label, arr):
    val = [] 
    
    indices = np.arange(0, 61579, step = 1)
    
    for i in range(len(arr)): 
        value = arr[i][0][label]
        val.append(value)
        
    val = np.array(val)
    
    sorted_indices = np.argsort(val)
    val_sorted = val[sorted_indices]
    indices_sorted = indices[sorted_indices]
    
    index_max = indices_sorted[-6:-1]
    
    return index_max
#for classfiications csv file 
def highest_classification(label, df, df_columns):
    #defining indices
    classifications = pd.read_csv('training_classifications.csv', header = 0)
    indices = np.arange(0, 61578, step = 1)
    path = os.listdir(r'/Users/iisan1/Downloads/training_images')
    current_label = list(df[df_columns[label]])#finding the maximum in that label
#     maxes = sorted(current_label)
    GalaxyID = list(classifications['GalaxyID'])
    
    test_list = df[df_columns[label]]
    N = 5
    index5 = sorted(range(len(test_list)), key = lambda sub: test_list[sub])[-N:]
    galaxyID5 = [GalaxyID[i] for i in index5]
    path_index5 = [path[i] for i in galaxyID5]
    return galaxyID5, path_index5

smooth = highest_val(0, prediction_validationno20)
star =  highest_val(2, prediction_validationno20)
edge = highest_val(3, prediction_validationno20)
ring = highest_val(18, prediction_validationno20)
lens = highest_val(19, prediction_validationno20)
spiral2 = highest_val(32, prediction_validationno20)
merger = highest_val(23, prediction_validationno20)
#print("indices:",smooth, star, spiral2)


classification_labels = list(classifications.columns)[1:]
real_smooth = highest_classification(0, classifications_noID, classification_labels)
real_star = highest_classification(2, classifications_noID, classification_labels)
real_edge = highest_classification(3, classifications_noID, classification_labels)
real_ring = highest_classification(18, classifications_noID, classification_labels)
real_lens = highest_classification(19, classifications_noID, classification_labels)
real_spiral2 = highest_classification(32, classifications_noID, classification_labels)
real_merger = highest_classification(23, classifications_noID, classification_labels)

real_no23 = [real_smooth, real_star, real_edge, real_ring, real_lens, real_spiral2, real_merger]
no23 = [smooth, star, edge, ring, lens, spiral2, merger]
label_indices = [0, 2, 3, 18, 19, 32, 23]
no23_labels = ['smooth', 'star or artifact', 'edge-on', 'ring', 'lens', '2 spiral arms', 'merger']

#need this because otherwise the generator will select the galaxy ID as the highest value
classifications_noID = classifications.drop('GalaxyID', axis = 1)

#csv that is indexed by the galaxy ID
ID = pd.read_csv('training_classifications.csv', index_col = 'GalaxyID')

from torchvision.transforms import v2

#from this generator we are looping through each of the galaxies and yielding their highest label value and label 
def no24_batch(batch_no, batch_size): 
    
    #initializing dictionary 
    stored_galaxy = {} 
    resized_path = r'/Users/iisan1/Downloads/resized_test_images/'
    
    #takes in int
    #label_arrays = np.zeros((batch_size, 37), dtype = np.float32)
    tensor_image_array = np.zeros((batch_size, 3, 77, 77), dtype = np.float32)
    #galaxies = np.zeros((batch_size))
    i = 0
    for image in batch_no: 
        i = int(i)
        #getting current galaxy and adding to array
        current_galaxy = int(image.split('_resized.jpg')[0])
        #galaxies[i] = current_galaxy
        
        
        #creating array of label values for current galaxy 
        #label_array = np.array(np.array(ID.loc[current_galaxy]))
        #label_arrays[i] = label_array
        
        
        #getting image and converting to tensor and adding to array 
        img = Image.open(resized_path + str(current_galaxy) + '_resized.jpg')
        #converts image to tensor
        convert_tensor = transforms.ToTensor()
        tensor_image = convert_tensor(img)
        #randomly rotates image 
        rotater = v2.RandomRotation(degrees=(0, 90))
        rotated_img = rotater(tensor_image)
        tensor_image_array[i] = rotated_img
        
        i+= 1
        
    #converting array of images into tensor  
    tensor_image_array = torch.from_numpy(tensor_image_array)
    
    #converting label array to tensor 
    #label_arrays = torch.from_numpy(label_arrays)
    
    yield tensor_image_array
val_epoch_lossesno24 = []
for epoch in range(15):
    model.eval()
    prediction_validationno24 = []
    with torch.no_grad():
        for batch in range(len(testing_img)): 
            val_lossno24 = []
            #from generator pull the images and labels for each image 
            current_batch = testing_img[batch]
            generator_function= next(no24_batch(current_batch, len(current_batch)))

            for i in range(len(current_batch)):
                inputs = generator_function[i]
                inputs = inputs.reshape(1, 3, 77, 77)

                prediction = validation_batch(inputs, labels_tensor, model)
                prediction_validationno24.append(prediction)
def comparison_figure(comparison, real, specific_label): 
    fig, axs = plt.subplots(2, 5, figsize = (15, 10))
    fig.tight_layout()
    label = 0 
    label_real = 0
    path = r'/Users/iisan1/Downloads/training_images'
    image_path = os.listdir(path)
    for i in range(2): 
        for j in range(5): 
            if i%2 == 0: 
                image = Image.open(path + '/' + image_path[real[j]])
                axs[i, j].imshow(image)
                plt.suptitle(f'Real Classification: {no23_labels[specific_label]}')
                axs[i, j].set_title(f'ID: {real[label_real]}')
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                axs[i,j].set_ylabel('Real')
                label_real += 1
                
            else: 
                image = Image.open(path + '/' + image_path[comparison[j]])
                axs[i, j].imshow(image)
                plt.suptitle(f'Real vs. My Model Classification: {no23_labels[specific_label]}')
                axs[i,j].set_ylabel('Model')
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                axs[i, j].set_title(f'ID: {comparison[label]}')
                label += 1
    plt.subplots_adjust(wspace=0)
    fig.savefig(f'No23{no23_labels[specific_label]}.png')
    
    
for l in range(len(real_no23)): 
    comparison_figure(no23[l], real_no23[l], l)


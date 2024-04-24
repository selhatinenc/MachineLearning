#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report


# # ASSIGNMENT 2 
# 
# libraries pd , np , os  and cv2 is legal 
# 
# time is for just showing how much time does it takes to run my code

# # FEATURE EXTRACTION 
# 
# we extract feature by  
# 
# appliying resize
# 
# applying multiple instances of Gabor Filter
# 
# applying gray scale image
# 

# In[2]:


def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree

    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters


# In[3]:


def apply_filter(img, filters):
    # This general function is designed to apply filters to our image

    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)

    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1  # remain depth same as original image

    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  # Apply filter to image

        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage


# # ASSIGMENT FOLDER  FOR FLOWERS
# 
# below I take my images from my local folder for assignments

# In[4]:
print("hello")

image_folder = r".\flowers\train"
image_foldertest = r".\flowers\test"
image_foldervalid = r".\flowers\validation"


# # creating dataframe for Train and Test data

# In[5]:


dftrain = pd.DataFrame()
dftest = pd.DataFrame()
dfvalid= pd.DataFrame()


# # reading function for images

# In[6]:


def read(imagefolder,data):
    
    for flower_type in os.listdir(imagefolder):

        type_folder = os.path.join(imagefolder, flower_type)

        for image_file in os.listdir(type_folder):
            # 1 path of image

            image_path = os.path.join(type_folder, image_file)

            # 2 label name extracted from
            labelname = os.path.basename(os.path.dirname(image_path))

            # 3 burada resize et ve target size belirle
            resized_img = cv2.resize(cv2.imread(image_path), (8, 8))

            # gabor filter
            gfilters = create_gaborfilter()

            # gabor rgb
            image_g = apply_filter(resized_img, gfilters)

            gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            a = np.array(gray_image)

            result = a.flatten()

            df_image = pd.DataFrame(result).T
            df_image.columns = [f'pixel_{i}' for i in range(len(result))]
            df_image['label'] = labelname

            # Ana DataFrame'e ekle
            data = pd.concat([data, df_image], ignore_index=True)
    
    return data   


# # read train

# In[7]:


start_time1 = time.time()
dftrain = read(image_folder,dftrain)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(elapsed_time1)


# # Show first 5 column

# In[8]:


print(dftrain.head(5))


# # read test

# In[9]:


start_time1 = time.time()
dftest = read(image_foldertest,dftest)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(elapsed_time1)


# # Show first 5 column

# In[10]:


print(dftest.head(5))


# # read valid

# In[11]:


start_time1 = time.time()
dfvalid = read(image_foldervalid,dfvalid)
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(elapsed_time1)


# # Show first 5 column

# In[12]:


print(dfvalid.head(5))


# # create attribute list from data

# In[13]:


attributelist=[]
for column in dftrain.columns:
    if column !="label":
        attributelist.append(column)


# # count how many unique value of target

# In[14]:


unique_count = dftrain['label'].nunique()


# # ID3 MODEL IMPLEMENTATION

# # create function for entropy

# In[15]:


def Entropy(datatarget):

    total_instances = len(datatarget)
    # how many unique values
    label_counts = datatarget.value_counts()

    entropiSlist = []

    for label, count in label_counts.items():
        probability = count / total_instances

        tolerans = 1e-10
        log_result=np.emath.logn(unique_count, probability)
        if abs(log_result + 1) < tolerans:
            log_result = -1
        entropiSlist.append(probability * log_result)

    return -sum(entropiSlist)


# # create function for GAİN , information gain

# In[16]:


def information_gain(data, attribute, target,totalentropy):
    # Calculate total entropy before the split

    column1_array = data[attribute].to_numpy()
    uniqarray = np.unique(column1_array)
    new_array1 = np.insert(uniqarray, 0, 0)
    new_array2 = np.append(uniqarray, 0)
    new_array1 = (new_array1 + new_array2) / 2
    new_array1[0] = -float("inf")
    new_array1[-1] = float("inf")
    intervallist = [(new_array1[i], new_array1[i + 1]) for i in range(len(new_array1) - 1)]

    for interval in intervallist:

        subset = data[(data[attribute] >= interval[0]) & (data[attribute] < interval[1])] 
        totalentropy = totalentropy - ( (len(subset) / len(data)) * Entropy(subset[target])  )


    return totalentropy,attribute,intervallist



# # create algorithm function for ID3
# 
# some explanation here

# In[17]:


def ID3Algorith(data,attributes,target):



    #1 if all labels same return the label
    if len(np.unique(data[target])) == 1:
        node = DecisionNode(label=data[target].iloc[0])
        return node
    #2 if no attributes left return most common label
    if len(attributes) == 0:

        unique_values, counts = np.unique(data[target],return_counts=True)
        most_common_value = unique_values[np.argmax(counts)]
        node = DecisionNode(label=most_common_value)
        return node
    #3

    infogain=-1
    best_attribute = ""
    intervallist=[]
    total_entropy = Entropy(data[target])
    for attribute in attributes:

        infogainx,attribute,intervallistx= information_gain(data, attribute, target,total_entropy)

        if infogainx > infogain:
            infogain = infogainx
            best_attribute = attribute
            intervallist = intervallistx

    attributes.remove(best_attribute)
    
    node = DecisionNode(attribute=best_attribute, intervallist=intervallist)

    for subsets in intervallist:

        subsetx = data[(data[best_attribute] >= subsets[0]) & (data[best_attribute] < subsets[1])]
        nodex =ID3Algorith(subsetx, attributes, target)
        node.nodelist.append(nodex)
    #return node
    return node


# # create Node Class for decision tree

# In[18]:


class DecisionNode:
    def __init__(self, label=None, attribute=None, intervallist=None,nodelist=None):
        self.label = label
        self.attribute = attribute
        self.intervallist = intervallist if intervallist is not None else []
        self.nodelist = nodelist if nodelist is not None else []


# # create function for printing rules so we can see better and examine our code

# In[19]:


def print_rules(node, condition):

   if node is not None:

        # Node sonuna geldik
        if node.label is not None:
            condition = condition + " then " + node.label
            print(condition)
            return


        if condition is None:
            condition = "If "
        else:
            condition = condition + " ∧ "


        condition = condition + node.attribute + "--> "

        for i, interval_node in enumerate(node.nodelist):

            new_condition = condition +  str(node.intervallist[i])

            print_rules(interval_node, condition=new_condition)


# # create function for printing tree left to rigt nodes

# In[20]:


def print_tree(node, level=0, prefix="Root: ", node_number=1, condition=None):
    arrow = "↪"
    if node is not None:
        condition_text = f"Condition: {condition}" if condition else ""
        print(" " * (
                    level * 6)+ f"{arrow}" + f"{prefix}Node {node_number} (Level {level}): Label: {node.label}, Attribute: {node.attribute} {condition_text}")

        for i, interval_node in enumerate(node.nodelist):
            interval_condition = f"Interval {i + 1}"
            print_tree(interval_node, level + 1, prefix=f"Interval {i+1}: ", node_number=node_number * 2 + i, condition=interval_condition)


# # what we did so far :
# 
# we crated functions for applying feature extraction
# 
# we created functions for reading our image data and created dataframes
# 
# we defined some variables and list from dataframes
# 
# we created functions for our ID3 model implementation
# 
# we created node class for better showcasing our tree
# 
# we created functions for printing tree and rules for our trained data with ID3 functions

# # TRAINING OUR DATA WITH ID3 MODEL

# In[21]:


start_time1 = time.time()
DecisionTree=ID3Algorith(dftrain,attributelist,"label")
end_time1 = time.time()
elapsed_time1 = end_time1 - start_time1
print(elapsed_time1)


# # SHOW CASING THE RULES 

# In[22]:


print_rules(DecisionTree,None)


# # SHOW CASING TREE

# In[23]:


print_tree(DecisionTree)


# # CLASSIFICATION OF TEST

# In[24]:


def classification(row, node):
    if node is not None:
        # Node sonuna geldik
        if node.label is not None:
            return node.label

        for i, interval_node in enumerate(node.nodelist):
            if node.intervallist[i][0] <= row[node.attribute] < node.intervallist[i][1]:
                return classification(row, interval_node)




# In[25]:


def predict(data, node):

    data['Predictions'] = data.apply(lambda row: classification(row, node), axis=1)


# In[26]:


dftest2 = pd.DataFrame()
dftest2=read(image_foldertest,dftest2)
predict(dftest2, DecisionTree)

print(dftest2)


# In[27]:


# Gerçek etiketler ve tahminler
y_true = dftest2['label']
y_pred = dftest2['Predictions']

# Accuracy (Doğruluk)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision (Hassasiyet)
precision = precision_score(y_true, y_pred, average='weighted')
print("Precision:", precision)

# Recall (Duyarlılık)
recall = recall_score(y_true, y_pred, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)


# # CLASSIFICATION OF TRAIN

# In[28]:


dftrain2 = pd.DataFrame()
dftrain2=read(image_folder,dftrain2)
predict(dftrain2, DecisionTree)

print(dftrain2)


# In[29]:


print(dftrain)


# In[30]:


# Gerçek etiketler ve tahminler
y_true = dftrain2['label']
y_pred = dftrain2['Predictions']

# Accuracy (Doğruluk)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision (Hassasiyet)
precision = precision_score(y_true, y_pred, average='weighted')
print("Precision:", precision)

# Recall (Duyarlılık)
recall = recall_score(y_true, y_pred, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)


# # Error Analysis for Classification
# 
# • Find a few misclassified images and comment on why you think they were hard to
# classify.
# 
# • Compare the performance of different ID3 model variation choices for your dataset.
# Wherever relevant, feel free to discuss computation time in addition to the classification rate.

# # PRUNING 

# # CLASSIFICATION OF VALIDATION

# In[31]:


dfvalid2 = pd.DataFrame()
dfvalid2=read(image_foldervalid,dfvalid2)
predict(dfvalid2, DecisionTree)

print(dfvalid2)


# In[32]:


# Gerçek etiketler ve tahminler
y_true = dfvalid2['label']
y_pred = dfvalid2['Predictions']

# Accuracy (Doğruluk)
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# Precision (Hassasiyet)
precision = precision_score(y_true, y_pred, average='weighted')
print("Precision:", precision)

# Recall (Duyarlılık)
recall = recall_score(y_true, y_pred, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 Score:", f1)

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true, y_pred)
print("Classification Report:")
print(report)


# # TWIG FINDER FUNCTION 

# In[33]:


def twigfinder(root,data,target,giriş):

    result = []
    queue = [(root, "None", (0, 0))] 

    infogaina = float("inf")

    while queue:

        current_node, parentattribute, parentinterval = queue.pop(0)

        if current_node.label is None and current_node.nodelist is not None:

            if all(not child.nodelist for child in current_node.nodelist):
                # eğer root node twig olduysa boş döndür ve pruning içinde hesaplama yapma , print ay
                if parentattribute !=  "None":

                    # info gain hesapla
                    subsetx = data[
                        (data[parentattribute] >= parentinterval[0]) & (data[parentattribute] < parentinterval[1])]
                    total_entropy = Entropy(subsetx[target])
                    infogainx, attribute, intervallistx = information_gain(subsetx, current_node.attribute, target,
                                                                           total_entropy)
                    print(infogainx)
                    print(parentattribute)
                    print(current_node.attribute)
                    print(intervallistx)
                    giriş=giriş+1
                    print(giriş)
                    print()
                    if infogainx < infogaina:
                        infogaina = infogainx
                        if len(result) != 0:
                            result.pop(0)
                        result.append((current_node, infogainx,parentattribute,parentinterval,subsetx))
                        
                    


            queue.extend(
                (child, current_node.attribute, current_node.intervallist[i])
                for i, child in enumerate(current_node.nodelist)
                if child.label is None
            )


    
    return result


# # PRUNING ALGORITHM

# In[34]:


giriş=0


# In[35]:


def pruning(LastAccuracy,nodex,data,dfvalid,prunnedlist,target,giriş):
   
    print("giriş")
    dfvalid2=dfvalid
    #CATALOG ALL TWIGS IN THE TREE
    catalogtwigs = twigfinder(nodex,data,target,giriş)

    nodelistorg=catalogtwigs[0][0].nodelist
    attributeorg=catalogtwigs[0][0].attribute
    intervallistorg=catalogtwigs[0][0].intervallist

    catalogtwigs[0][0].nodelist = []
    catalogtwigs[0][0].attribute = None
    catalogtwigs[0][0].intervallist=[]

    catalogtwigs[0][0].label = catalogtwigs[0][4][target].value_counts().idxmax()

    prunnedlist.append(attributeorg)

    predict(dfvalid2,nodex)

    y_true = dfvalid2['label']
    y_pred = dfvalid2['Predictions']

    # Accuracy (Doğruluk)
    CurrentAccuracy = accuracy_score(y_true, y_pred)
    print(CurrentAccuracy)
    print(LastAccuracy)
    if CurrentAccuracy >= LastAccuracy:
        print("girdi")
        pruning(CurrentAccuracy, nodex, data, dfvalid, prunnedlist, target,giriş)
    else:
        prunnedlist.pop()
        catalogtwigs[0][0].nodelist= nodelistorg
        catalogtwigs[0][0].attribute=attributeorg
        catalogtwigs[0][0].intervallist=intervallistorg
        catalogtwigs[0][0].label = None

    return nodex,prunnedlist


# # NEW PRUNNED DECİSİON TREE

# In[36]:


dftrain3 = pd.DataFrame()
dftrain3=read(image_folder,dftrain3)
attributelist=[]
for column in dftrain3.columns:
    if column !="label":
        attributelist.append(column)
dftrain4 = pd.DataFrame()
dftrain4=read(image_folder,dftrain4)

NewDecisionTree=ID3Algorith(dftrain4,attributelist,"label")
prunnedlist=[]
PrunnedDecisionModel,prunnedlist= pruning(accuracy,NewDecisionTree,dftrain3,dfvalid,prunnedlist,"label",0)
print("prunned list:" + ', '.join(map(str, prunnedlist)))


# # TEST PRE PRUNING DECISION TREE

# In[37]:


dftest4 = pd.DataFrame()
dftest4=read(image_foldertest,dftest4)
predict(dftest4, DecisionTree)

y_true4 = dftest4['label']
y_pred4 = dftest4['Predictions']

# Accuracy (Doğruluk)
accuracytestPRE = accuracy_score(y_true4, y_pred4)
print("Accuracy: {:.5f}".format(accuracytestPRE))

# Precision (Hassasiyet)
precision = precision_score(y_true4, y_pred4, average='weighted')
print("Precision:", precision)

# Recall (Duyarlılık)
recall = recall_score(y_true4, y_pred4, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true4, y_pred4, average='weighted')
print("F1 Score:", f1)

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_true4, y_pred4)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true4, y_pred4)
print("Classification Report:")
print(report)


# In[38]:


print_rules(DecisionTree,None)


# In[39]:


print(dftest4)


# # TEST PRUNNED DECISION TREE

# In[40]:


dftest5 = pd.DataFrame()
dftest5=read(image_foldertest,dftest5)
predict(dftest5, PrunnedDecisionModel)

y_true5 = dftest5['label']
y_pred5 = dftest5['Predictions']

# Accuracy (Doğruluk)
accuracytestPOST = accuracy_score(y_true5, y_pred5)
print("Accuracy: {:.5f}".format(accuracytestPOST))


# Precision (Hassasiyet)
precision = precision_score(y_true5, y_pred5, average='weighted')
print("Precision:", precision)

# Recall (Duyarlılık)
recall = recall_score(y_true5, y_pred5, average='weighted')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_true5, y_pred5, average='weighted')
print("F1 Score:", f1)

# Confusion Matrix (Karmaşıklık Matrisi)
conf_matrix = confusion_matrix(y_true5, y_pred5)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(y_true5, y_pred5)
print("Classification Report:")
print(report)


# In[41]:


print_rules(PrunnedDecisionModel,None)


# In[42]:


print(dftest5)


# In[ ]:





# In[ ]:





# In[ ]:





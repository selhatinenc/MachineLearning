# Import necessary libraries
import numpy as np
from DecisionTree import DecisionTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import cv2
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Load your dataset and perform train/test split
# Replace the placeholder paths with the actual paths to your dataset
nameofclass = "astilbe" 
assignment_path = "C:/Users/encse/OneDrive/Masaüstü/MachineLearning/Assignment2"
train_images_path = "./flowers/train/" + nameofclass
test_images_path = "./flowers/test/" + nameofclass
train_images_Output= "./Outputs/train/" + nameofclass+"/"
desired_width = 50
desired_height = 50
def resize_image(input_path, label2, width, height,images,Image_Normal,imagesRgb,imagesCannyEdge):
    # Read the input image
    image = cv2.imread(input_path)
    imageCannyEdge = canny_edge_detection(image)
    imagesCannyEdge.append(imageCannyEdge)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imagesRgb.append(rgb_image)
    Image_Normal.append(image)
    # Resize the image
    resized_image = cv2.resize(image, (width, height))

    # Save the resized image
    cv2.imwrite(train_images_Output+"resize/"+label2+ '.jpg', resized_image)
    cv2.imwrite(train_images_Output+"canny/"+ label2+'.jpg', imageCannyEdge)
    cv2.imwrite(train_images_Output+"rgb/"+ label2+'.jpg', rgb_image)
    images.append(resized_image)
def canny_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help the edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    return edges

# import cv2

# def resize_image(input_path, output_path, width, height):
#     # Read the input image
#     image = cv2.imread(input_path)

#     # Resize the image
#     resized_image = cv2.resize(image, (width, height))

#     # Save the resized image
#     cv2.imwrite(output_path, resized_image)

# if __name__ == "__main__":
#     # Input and output file paths
#     input_image_path = "path/to/your/input/image.jpg"
#     output_image_path = "path/to/your/output/resized_image.jpg"

#     # Specify the desired width and height
#     desired_width = 500
#     desired_height = 300

#     # Resize the image
#     resize_image(input_image_path, output_image_path, desired_width, desired_height)

#     print("Image resized successfully.")
test_labels = []
train_labels = []




Train_ReSize = []
Test_ReSize = []

Train_RGB = []
Test_RGB = []

Train_CannyEdge = []
Test_CannyEdge = []
def load_dataset(images_path,imagesPath2,images,labels,imagesRgb,imagesCannyEdge,limit):

    # Counter to keep track of iterations
    iteration_count = 0   
    print(images_path)
    Image_Normal = []
    for file in os.listdir(images_path):
        label = file.split("_")[0]  # Use the class name as the label
        label2=label
        label = int(label)  # Convert the label to an integer if needed

        image_path = os.path.join(images_path, file)
        resize_image(image_path, label2, desired_width, desired_height,images,Image_Normal,imagesRgb,imagesCannyEdge)

        labels.append(0)
        # Increment the counter
        iteration_count += 1

        # Check if the limit is reached
        if iteration_count >= limit:
            break
    
    # Counter to keep track of iterations
    iteration_count = 0   
    print(imagesPath2)
    Image_Normal = []
    for file in os.listdir(imagesPath2):
        label = file.split("_")[0]  # Use the class name as the label
        label2=label
        label = int(label)  # Convert the label to an integer if needed

        image_path = os.path.join(imagesPath2, file)
        resize_image(image_path, label2, desired_width, desired_height,images,Image_Normal,imagesRgb,imagesCannyEdge)

        labels.append(1)
        # Increment the counter
        iteration_count += 1

        # Check if the limit is reached
        if iteration_count >= limit:
            break
    
    return images, labels
    for root, dirs, files in os.walk(images_path):

        for file in files:
            # Assuming image names are in the format "label_image_number.jpg"
            label = file.split("_")[0]  # Extract the label from the filename
            label = int(label)  # Convert the label to an integer if needed

            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)  # Use OpenCV to read the image
            images.append(image)
            labels.append(label)
            # You may need to resize or preprocess the image based on your requirements
            resize_image(images_path, train_images_Output, desired_width, desired_height)
         

    return images, labels


# Load training data
train_images, train_labels = load_dataset("./flowers/train/astilbe","./flowers/train/bellflower",Train_ReSize,train_labels,Train_RGB,Train_CannyEdge,30)
# Load test data
test_images, test_labels = load_dataset("./flowers/test/astilbe","./flowers/test/bellflower",Test_ReSize,test_labels,Test_RGB,Test_CannyEdge,4)














def extract_features_from_image(image):
    # This is a placeholder function. You should replace this with your actual feature extraction logic.
    # For simplicity, let's assume the image is grayscale, and we flatten it as a feature.
    flattened_image = image.flatten()
    return flattened_image
def extract_color_features(image):
    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Calculate the mean and standard deviation of each color channel
    mean_color = np.mean(rgb_image, axis=(0, 1))
    std_color = np.std(rgb_image, axis=(0, 1))

    # Concatenate mean and standard deviation as features
    color_features = np.concatenate([mean_color, std_color])

    return color_features




# Function to extract features
def extract_features(images):
    # Implement feature extraction logic here
    # You can use libraries like OpenCV to resize, change color space, apply filters, etc.
    # For simplicity, let's assume we have a function called extract_features_from_image

    features = []
    for image in images:
        feature = extract_features_from_image(image)
        features.append(feature)

    return np.array(features)


def extract_features_ColorRgb(images):
    # Implement feature extraction logic here
    # You can use libraries like OpenCV to resize, change color space, apply filters, etc.
    # For simplicity, let's assume we have a function called extract_features_from_image

    features = []
    for image in images:
        feature = extract_color_features(image)
        features.append(feature)

    return np.array(features)




#*********************************************************************************************************************
# Function to train the decision tree model
def train_decision_tree(train_features, train_labels):
    model =  DecisionTree(max_depth=10)

    model.fit(train_features, train_labels)
    # plot tree
    plt.figure(figsize=(20,16))# set plot size (denoted in inches)
    tree.plot_tree(model,fontsize=10)                                                                                      
    return model

# Function to evaluate the model
def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')

    confusion = confusion_matrix(test_labels, predictions)

    return accuracy, precision, recall, f1, confusion


def print_decision_tree_rules(model, feature_names):
    tree_rules = []
    tree_ = model.tree_

    def recurse(node):
        nonlocal tree_rules
        if tree_.feature[node] != -2:  # Check if not a leaf node
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            rule = f"{name} <= {threshold:.4f}"
            tree_rules.append(rule)
            recurse(tree_.children_left[node])

            rule = f"{name} > {threshold:.4f}"
            tree_rules.append(rule)
            recurse(tree_.children_right[node])
        else:
            tree_rules.append(f"Class {np.argmax(tree_.value[node])}")

    recurse(0)
    return tree_rules

# # Load your dataset and perform train/test split
# # Replace the placeholder paths with the actual paths to your dataset
# nameofclass = "astilbe"
# train_images_path = "flower_photos/train/"+nameofclass
# test_images_path = "flower_photos/test/"+nameofclass


# def load_dataset(images_path):
#     images = []
#     labels = []

#     for root, dirs, files in os.walk(images_path):
#         for file in files:
#             # Assuming image names are in the format "label_image_number.jpg"
#             label = file.split("_")[0]  # Extract the label from the filename
#             label = int(label)  # Convert the label to an integer if needed

#             image_path = os.path.join(root, file)
#             image = cv2.imread(image_path)  # Use OpenCV to read the image
#             # You may need to resize or preprocess the image based on your requirements

#             images.append(image)
#             labels.append(label)

# #     return images, labels


# # Load training data
# train_images, train_labels = load_dataset(train_images_path)

# # Load test data
# test_images, test_labels = load_dataset(test_images_path)

# Main script
if __name__ == "__main__":
    # Step 1: Read your classification data
    # Assume you have loaded your data into train_features, train_labels, test_features, test_labels

    # Step 2: Extract features for each image in the training set
    ReSizetrain_features = extract_features(Train_ReSize)
    ReSizetest_features = extract_features(Test_ReSize)
    #******
    CannyEdgetrain_features = extract_features(Train_CannyEdge)
    CannyEdgetest_features = extract_features(Test_CannyEdge)
    #******
    RGBtrain_features = extract_features(Train_RGB)
    RGBtest_features = extract_features(Test_RGB)
    #******
    train_features = CannyEdgetrain_features
    test_features = CannyEdgetest_features
    print("ReSize")
    
    #train_features = np.concatenate((ReSizetrain_features, CannyEdgetrain_features, RGBtrain_features), axis=1)
    #test_features = np.concatenate((ReSizetest_features, CannyEdgetest_features, RGBtest_features), axis=1)
    #******
    # Step 3: Train your ID3 decision tree model with respect to your features
    model = train_decision_tree(train_features, train_labels)
    
# Print feature importances
    feature_importances = model.feature_importances_
    feature_names = ["ReSize","CannyEdge","RGB"]  # Replace with actual feature names
    
    for feature_name, importance in zip(train_images, feature_importances):
        print(f"Feature: {feature_name}, Importance: {importance}")
        feature_names.append(feature_name)
    # Step 4: For each given test sample, measure the features' quality
    # Step 5: Compute and report accuracy, precision, recall, and F1 Score
    accuracy, precision, recall, f1, confusion = evaluate_model(model, test_features, test_labels)
    predictions = model.predict(test_features)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    num_pixels = len(train_features[0])
    ftNames = [f"Pixel{i}" for i in range(1, num_pixels + 1)]
    #ftNames=["ReSize","CannyEdge","RGB"]
    # Step 6: Write the rules of your best-performing decision tree model
    rules = print_decision_tree_rules(model, ftNames)
    for rule in rules:
        print(rule)

    # Step 7: Error Analysis for Classification
    # Find a few misclassified images and comment on why you think they were hard to classify
    misclassified_indices = np.where(test_labels != predictions)[0]
    for idx in misclassified_indices[:5]:  # Display the first 5 misclassified images
        print(f"Misclassified Image: {idx}")
        # Display the misclassified image and any additional information for analysis

   
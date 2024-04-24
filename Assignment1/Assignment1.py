import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import string
from nltk.corpus import stopwords  
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import numpy as np
data = pd.read_csv('English Dataset.csv')
print(data.head())
data['Text'] = data['Text'].str.lower()
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

data['Text'] = data['Text'].apply(remove_punctuation)


nltk.download('stopwords')
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['Text'] = data['Text'].apply(remove_stopwords)


stemmer = PorterStemmer()

def stem_text(text):
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

data['Text'] = data['Text'].apply(stem_text)
# You may need to clean and preprocess your text data depending on its quality.
# You can use text cleaning techniques like removing punctuation, stopwords, and stemming.

# Step 2: Create a Bag of Words (BoW) representation

# Initialize the CountVectorizer with the desired options
vectorizer = CountVectorizer(
    lowercase=True,          # Convert text to lowercase
    stop_words='english',    # Remove common English stopwords
    max_features=1000,       # Limit the vocabulary size (adjust as needed)
    binary=False             # Use word counts instead of binary values
)

# Fit the vectorizer on your text data and transform your text data into a BoW representation
X = vectorizer.fit_transform(data['Text'])

# Access the feature names (vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Convert X to a dense array (for better readability)
X_array = X.toarray()
print(feature_names )

print(X_array)



#--------------------------------------------------------------
print("---------------------------------------------------")    
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
 ]
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names_out())
print(X2.toarray())
# Now, X_array contains your BoW representation of the text data
# Each row corresponds to a document, and each column corresponds to a word in the dictionary.
# The values in X_array are word counts for each word in each document.

# You can proceed with k-NN classification using X_array as your feature matrix.
# Don't forget to   plit your data into training and testing sets and use labels (categories) for supervised learning.

# Example: Splitting your data into X (features) and y (labels)


# You can now use X and y for training and testing your k-NN classifier.

# Further analysis and model implementation can be performed based on your requirements.

# If you have other specific requirements or questions, feel free to ask.



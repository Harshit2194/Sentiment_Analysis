# %%
import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import csv
import math
import nltk
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# %% [markdown]
# **Reading the csv file**

# %%
#Creating a data frame out of the csv file
df = pd.read_csv(r"/Users/harshitpatel/Sem_5/IBM_PBEL/Sentiment_DATA_Set.csv")
df.head() #head function is used to show the top 5 rows of dataframe

# %%
#Some information about the dataset
df.info()

# %%
df["reviews.rating"].value_counts().sort_values(ascending = False)

# %% [markdown]
# **Converting the emojis to text**

# %%
#Converting Emojis to their Respective Emotions
df["reviews.text"] = df["reviews.text"].replace([r"\:\)",r"\:\-\)", r"\:\-\}",r"\;\-\}",r"\:\-\>",r"\;\-\)"], ["Happy","Happy","Happy","Happy","Happy","Happy"], regex=True)
df["reviews.text"] = df["reviews.text"].replace([r"\:\-\(",r"\:\(",r"\:\-\|",r"\;\-\(",r"\;\-\<",r"\|\-\{"], ["Sad", "Sad", "Sad", "Sad", "Sad", "Sad",], regex=True)
df["reviews.text"] = df["reviews.text"].replace([r"\:\D",r"\:\'\-\)",r"\:\`\-\("], ["laugh", "tear of joy", "tear of sadness"], regex=True)

# %%

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


# %%
# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords and stem
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# %%
data_set = df[["reviews.text", "reviews.rating"]]
data_set.columns = ["reviews", "score"]
data_set

# %%
data_set["sentiment"] = np.where(data_set["score"] >= 4, 1, -1)
data_set


# %%
df_majority = data_set[data_set['sentiment'] == 1]
df_minority = data_set[data_set['sentiment'] == -1]


# %%
print(len(df_majority))
print(len(df_minority))


# %%
nltk.download('punkt_tab')
# Balance the dataset
df_majority = data_set[data_set['sentiment'] == 1]
df_minority = data_set[data_set['sentiment'] == -1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,n_samples=len(df_minority),random_state=42)



# Combine balanced dataset
balanced_data = pd.concat([df_majority_downsampled, df_minority])

# Preprocess all reviews
balanced_data['processed_reviews'] = balanced_data['reviews'].apply(preprocess_text)


# %%
# Split into train and test sets
train, test = train_test_split(balanced_data, test_size=0.3, random_state=42)

# %%
# Save to CSV
train[['processed_reviews', 'sentiment']].to_csv('train.csv', index=False,header=False)
test[['processed_reviews', 'sentiment']].to_csv('test.csv', index=False,header=False)


# %%
# Read training data
with open("train.csv", 'r') as file:
    reviews = list(csv.reader(file))

# %%
# Get positive and negative texts
def get_text(reviews, score):
    return " ".join([r[0] for r in reviews if r[1] == str(score)])

negative_text = get_text(reviews, -1)
positive_text = get_text(reviews, 1)

# %%
# Count words
def count_text(text):
    words = re.split(r"\s+", text)
    return Counter(words)

negative_counts = count_text(negative_text)
positive_counts = count_text(positive_text)

# %%
# Get class counts
def get_y_count(score):
    return len([r for r in reviews if r[1] == str(score)])

positive_review_count = get_y_count(1)
negative_review_count = get_y_count(-1)

# Class probabilities
prob_positive = positive_review_count / len(reviews)
prob_negative = negative_review_count / len(reviews)


# %%
print(prob_positive)
print(prob_negative)

# %%
# Improved prediction function with log probabilities
def make_class_prediction(text, counts, class_prob, class_count):
    log_prediction = math.log(class_prob) 
    text_counts = Counter(re.split(r"\s+", text))
    total_words = sum(counts.values()) + class_count
    
    for word in text_counts:
        word_prob = (counts.get(word, 0) + 1) / total_words
        log_prediction += text_counts[word] * math.log(word_prob)
    
    return log_prediction


# %%
def evaluate_model(test_data):
    actual = []
    predictions = []
    
    for review in test_data:
        if len(review) < 2:  # Skip malformed rows
            continue
            
        text = review[0]
        try:
            true_label = int(review[1])
        except ValueError:
            continue  # Skip header row if it exists
            
        actual.append(true_label)
        
        neg_pred = make_class_prediction(text, negative_counts, prob_negative, negative_review_count)
        pos_pred = make_class_prediction(text, positive_counts, prob_positive, positive_review_count)
        
        pred = -1 if neg_pred > pos_pred else 1
        predictions.append(pred)
    
    if not actual:  # Check if we have any valid data
        print("No valid test data found")
        return
    
    print("\nClassification Report:")
    print(classification_report(actual, predictions, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(actual, predictions))
    
    fpr, tpr, thresholds = roc_curve(actual, predictions, pos_label=1)
    print(f"\nAUC: {auc(fpr, tpr):.2f}")

# %%
# Load test data properly
test_data = []
with open("test.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= 2:  # Only take rows with both text and label
            test_data.append(row)


# %%
# Evaluate model
evaluate_model(test_data)


# %%
# Interactive prediction
def predict_sentiment():
    while True:
        review_text = input("\nEnter a review to analyze (or 'quit' to exit): ")
        if review_text.lower() == 'quit':
            break
        
        processed_text = preprocess_text(review_text)
        neg_pred = make_class_prediction(processed_text, negative_counts, prob_negative, negative_review_count)
        pos_pred = make_class_prediction(processed_text, positive_counts, prob_positive, positive_review_count)
        
        print("\nPredicted sentiment:", "POSITIVE" if pos_pred > neg_pred else "NEGATIVE")

# Run interactive prediction
print("\nReview Sentiment Analysis Tool")
predict_sentiment()

# %%




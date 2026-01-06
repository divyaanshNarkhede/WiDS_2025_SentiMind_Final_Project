import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
import sys

INPUT_FILE = "Reviews.csv"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cleaned_data.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
nltk.download('stopwords', quiet=True)

# Load Data
print("Loading Data")
try:
    df = pd.read_csv(INPUT_FILE, nrows=50000)
    print(f"Dataset shape: {df.shape}")
    print(df.head(3))
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
    sys.exit(1)

print("\nMissing values:\n", df.isnull().sum())
df.dropna(subset=['Text', 'Score'], inplace=True)

# Label Sentiment
def map_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score <= 2:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Score'].apply(map_sentiment)

print("\nClass Distribution:")
print(df['Sentiment'].value_counts())

# Visualize Sentiment Distribution
plt.figure(figsize=(8, 6))
sentiment_order = ['Negative', 'Neutral', 'Positive']
ax = sns.countplot(x='Sentiment', data=df, order=sentiment_order, hue='Sentiment', 
                   palette='viridis', legend=False)

plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Reviews')

# Add counts on bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), textcoords='offset points')

# Save plot to outputs folder
plot_path = os.path.join(OUTPUT_DIR, "sentiment_distribution.png")
plt.savefig(plot_path)
print(f"\nSentiment histogram saved to: {plot_path}")
plt.show()

# Text Cleaning
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def clean_text(text):
    # Handle non-string input
    if not isinstance(text, str):
        text = str(text)
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords and stem
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(words)

print("\nCleaning Text")
df['Cleaned_Text'] = df['Text'].apply(clean_text)
print("Sample Cleaned Text:", df['Cleaned_Text'].iloc[0])

# Save Data
cleaned_df = df[["Cleaned_Text", "Sentiment"]]
cleaned_df.to_csv(OUTPUT_FILE, index=False)

print("\nPreprocessing completed successfully.")
print(f"Saved cleaned data to: {OUTPUT_FILE}")
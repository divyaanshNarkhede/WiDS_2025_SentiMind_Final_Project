"""
Feature Extraction Module for Sentiment Analysis
================================================
This module implements various feature extraction techniques:
1. Bag-of-Words (BoW)
2. TF-IDF Vectorization
3. Word2Vec Embeddings
4. BERT Embeddings

Uses the cleaned data from preprocessing.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# BERT imports are optional - only needed if extracting BERT features
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Note: transformers/torch not installed. BERT features will not be available.")

# Configuration
INPUT_FILE = os.path.join("outputs", "cleaned_data.csv")
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load cleaned data from preprocessing step."""
    print("Loading cleaned data...")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} samples")
        return df
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run preprocessing.py first.")
        exit()


def split_data(df):
    """Split data into train and test sets."""
    print("\nSplitting data into train/test sets...")
    X = df['Cleaned_Text']
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# 1. Bag-of-Words (BoW)
# =============================================================================
def extract_bow_features(X_train, X_test, max_features=5000):
    """
    Extract Bag-of-Words features.
    
    BoW creates a vocabulary of all unique words and represents each document
    as a vector of word counts.
    
    Parameters:
    -----------
    X_train : pd.Series - Training text data
    X_test : pd.Series - Test text data
    max_features : int - Maximum vocabulary size
    
    Returns:
    --------
    X_train_bow, X_test_bow, bow_vectorizer
    """
    print("\n" + "="*50)
    print("Extracting Bag-of-Words Features")
    print("="*50)
    
    bow_vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,  # Ignore terms that appear in less than 2 documents
        max_df=0.95  # Ignore terms that appear in more than 95% of documents
    )
    
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(bow_vectorizer.vocabulary_)}")
    print(f"Train matrix shape: {X_train_bow.shape}")
    print(f"Test matrix shape: {X_test_bow.shape}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(OUTPUT_DIR, "bow_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(bow_vectorizer, f)
    print(f"BoW vectorizer saved to: {vectorizer_path}")
    
    return X_train_bow, X_test_bow, bow_vectorizer


# =============================================================================
# 2. TF-IDF Vectorization
# =============================================================================
def extract_tfidf_features(X_train, X_test, max_features=5000):
    """
    Extract TF-IDF features.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) weighs words by their
    importance - common words get lower weights, rare words get higher weights.
    
    Parameters:
    -----------
    X_train : pd.Series - Training text data
    X_test : pd.Series - Test text data
    max_features : int - Maximum vocabulary size
    
    Returns:
    --------
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer
    """
    print("\n" + "="*50)
    print("Extracting TF-IDF Features")
    print("="*50)
    
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Train matrix shape: {X_train_tfidf.shape}")
    print(f"Test matrix shape: {X_test_tfidf.shape}")
    
    # Show top features
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print(f"\nSample features: {list(feature_names[:10])}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"TF-IDF vectorizer saved to: {vectorizer_path}")
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer


# =============================================================================
# 3. Word2Vec Embeddings
# =============================================================================
def extract_word2vec_features(X_train, X_test, vector_size=100, window=5, min_count=2):
    """
    Extract Word2Vec embeddings.
    
    Word2Vec learns dense vector representations where semantically similar
    words have similar vectors. Document vectors are created by averaging
    word vectors.
    
    Parameters:
    -----------
    X_train : pd.Series - Training text data
    X_test : pd.Series - Test text data
    vector_size : int - Dimensionality of word vectors
    window : int - Context window size
    min_count : int - Minimum word frequency
    
    Returns:
    --------
    X_train_w2v, X_test_w2v, word2vec_model
    """
    print("\n" + "="*50)
    print("Extracting Word2Vec Features")
    print("="*50)
    
    # Tokenize texts
    train_tokens = [text.split() for text in X_train]
    test_tokens = [text.split() for text in X_test]
    
    # Train Word2Vec model
    print("Training Word2Vec model...")
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=RANDOM_STATE,
        epochs=10
    )
    
    print(f"Vocabulary size: {len(w2v_model.wv)}")
    print(f"Vector dimensionality: {vector_size}")
    
    def document_vector(tokens, model):
        """Average word vectors to get document vector."""
        vectors = []
        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    # Create document vectors
    print("Creating document vectors...")
    X_train_w2v = np.array([document_vector(tokens, w2v_model) for tokens in train_tokens])
    X_test_w2v = np.array([document_vector(tokens, w2v_model) for tokens in test_tokens])
    
    print(f"Train matrix shape: {X_train_w2v.shape}")
    print(f"Test matrix shape: {X_test_w2v.shape}")
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "word2vec_model.bin")
    w2v_model.save(model_path)
    print(f"Word2Vec model saved to: {model_path}")
    
    # Show similar words example
    try:
        sample_word = "good"
        if sample_word in w2v_model.wv:
            similar = w2v_model.wv.most_similar(sample_word, topn=5)
            print(f"\nWords similar to '{sample_word}': {similar}")
    except KeyError:
        pass
    
    return X_train_w2v, X_test_w2v, w2v_model


# =============================================================================
# 4. BERT Embeddings
# =============================================================================
def extract_bert_features(X_train, X_test, max_length=128, batch_size=32, max_samples=None):
    """
    Extract BERT embeddings.
    
    BERT (Bidirectional Encoder Representations from Transformers) provides
    contextualized embeddings that capture semantic meaning based on context.
    
    Note: This is computationally expensive. Consider using a subset of data
    or a GPU for faster processing.
    
    Parameters:
    -----------
    X_train : pd.Series - Training text data
    X_test : pd.Series - Test text data
    max_length : int - Maximum sequence length
    batch_size : int - Batch size for processing
    max_samples : int - Maximum samples to process (None for all)
    
    Returns:
    --------
    X_train_bert, X_test_bert
    """
    print("\n" + "="*50)
    print("Extracting BERT Features")
    print("="*50)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained BERT
    print("Loading BERT model (this may take a moment)...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()
    
    def get_bert_embeddings(texts, desc="Processing"):
        """Get BERT embeddings for a list of texts."""
        embeddings = []
        
        # Limit samples if specified
        if max_samples:
            texts = texts[:max_samples]
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size].tolist()
            batch_num = i // batch_size + 1
            
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"  {desc}: Batch {batch_num}/{total_batches}")
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding as document representation
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        return np.vstack(embeddings)
    
    print(f"\nProcessing training data ({len(X_train)} samples)...")
    X_train_bert = get_bert_embeddings(X_train, "Train")
    
    print(f"\nProcessing test data ({len(X_test)} samples)...")
    X_test_bert = get_bert_embeddings(X_test, "Test")
    
    print(f"\nTrain matrix shape: {X_train_bert.shape}")
    print(f"Test matrix shape: {X_test_bert.shape}")
    
    return X_train_bert, X_test_bert


# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Run all feature extraction methods."""
    print("="*60)
    print("FEATURE EXTRACTION FOR SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load and split data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save labels
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.values)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.values)
    print(f"\nLabels saved to outputs folder")
    
    # 1. Bag-of-Words
    X_train_bow, X_test_bow, bow_vec = extract_bow_features(X_train, X_test)
    
    # Save BoW features (sparse matrices)
    from scipy import sparse
    sparse.save_npz(os.path.join(OUTPUT_DIR, "X_train_bow.npz"), X_train_bow)
    sparse.save_npz(os.path.join(OUTPUT_DIR, "X_test_bow.npz"), X_test_bow)
    
    # 2. TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vec = extract_tfidf_features(X_train, X_test)
    
    # Save TF-IDF features (sparse matrices)
    sparse.save_npz(os.path.join(OUTPUT_DIR, "X_train_tfidf.npz"), X_train_tfidf)
    sparse.save_npz(os.path.join(OUTPUT_DIR, "X_test_tfidf.npz"), X_test_tfidf)
    
    # 3. Word2Vec
    X_train_w2v, X_test_w2v, w2v_model = extract_word2vec_features(X_train, X_test)
    
    # Save Word2Vec features
    np.save(os.path.join(OUTPUT_DIR, "X_train_word2vec.npy"), X_train_w2v)
    np.save(os.path.join(OUTPUT_DIR, "X_test_word2vec.npy"), X_test_w2v)
    
    # 4. BERT (Optional - can be slow without GPU)
    # Skip BERT by default - uncomment below to enable
    use_bert = 'n'
    # use_bert = input("\nExtract BERT features? This can be slow without GPU. (y/n): ").lower()
    if use_bert == 'y':
        # Use a subset for demonstration if dataset is large
        max_bert_samples = None
        if len(X_train) > 5000:
            max_bert_samples = 5000
        
        X_train_bert, X_test_bert = extract_bert_features(
            X_train, X_test, max_samples=max_bert_samples
        )
        
        # Save BERT features
        np.save(os.path.join(OUTPUT_DIR, "X_train_bert.npy"), X_train_bert)
        np.save(os.path.join(OUTPUT_DIR, "X_test_bert.npy"), X_test_bert)
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)
    print("\nFeatures saved to outputs folder:")
    print("  - Bag-of-Words: X_train_bow.npz, X_test_bow.npz")
    print("  - TF-IDF: X_train_tfidf.npz, X_test_tfidf.npz")
    print("  - Word2Vec: X_train_word2vec.npy, X_test_word2vec.npy")
    if use_bert == 'y':
        print("  - BERT: X_train_bert.npy, X_test_bert.npy")
    print("  - Labels: y_train.npy, y_test.npy")
    print("  - Vectorizers: bow_vectorizer.pkl, tfidf_vectorizer.pkl")
    print("  - Word2Vec model: word2vec_model.bin")


if __name__ == "__main__":
    main()

import numpy as np
import spacy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

class BOWpipe:
  def __init__(self, data, n=1, tfidf=False):
    self.dataset = data
    self.nlp = spacy.blank("en")
    self.processed_dataset = self.process_document(self.dataset)
    self.encoding, self.vocab = self.encode_corpus(self.processed_dataset, n, tfidf)
    self.shape = self.encoding.shape
    self.nbytes = self.encoding.nbytes

  def bag_of_ngrams(self, documents, n=1):
    vocabulary = {}
    
    for doc in documents:
      if len(doc) < n: continue # Skip if doc is shorter than n-gram
      for i in range(len(doc) - n + 1):
        ngram = tuple(doc[i:i+n])
        if ngram not in vocabulary:
          vocabulary[ngram] = len(vocabulary)

    dataset = []
    for doc in documents:
      vector = [0] * len(vocabulary)
      # Check if doc is valid for n-gram generation
      if len(doc) >= n:
          for i in range(len(doc) - n + 1):
            ngram = tuple(doc[i:i+n])
            if ngram in vocabulary:
              vector[vocabulary[ngram]] += 1
      dataset.append(vector)
    
    return dataset, vocabulary
  
  def tfidf_transformation(self, feature_vectors):
    feature_vectors = np.array(feature_vectors, dtype=float)
    n_docs = feature_vectors.shape[0]
    n_vocab = feature_vectors.shape[1]
    
    # 1. Calculate IDF
    # Count how many docs contain each term
    doc_freq = np.sum(feature_vectors > 0, axis=0)
    
    # Avoid division by zero by adding 1 (Smoothing)
    IDF = np.log((n_docs + 1) / (doc_freq + 1)) + 1

    # 2. Calculate TF * IDF
    total_terms = feature_vectors.sum(axis=1, keepdims=True)
    total_terms[total_terms == 0] = 1  # avoid division by zero for empty docs
    tf = feature_vectors / total_terms

    tfidf = tf * IDF

    # 3. Normalize TF-IDF vectors
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return tfidf / norms
  
  def process_document(self, corpus):
    processed_corpus = []
    for d in corpus:
      doc = self.nlp(d) # Tokenise
      doc_tokens = []
      for token in doc:
        if not token.is_stop: # Check for stop words
          doc_tokens.append(token.text)
      processed_corpus.append(doc_tokens)
    return processed_corpus
  
  def encode_corpus(self, corpus, n, tfidf):
    encoding, vocab = self.bag_of_ngrams(corpus, n)
    if tfidf:
      encoding = self.tfidf_transformation(encoding)
    return np.array(encoding), vocab
  
class Gaussian_Classifier:
    def __init__(self, title, encoded_corpus, labels):
        self.title = title
        self.train, self.test, self.labels_train, self.labels_test = train_test_split(
        encoded_corpus, labels, test_size=0.33, random_state=42)
        self.model = self.train_model(self.train, self.labels_train)

    def train_model(self, encoded_corpus, labels):
        print(f"Training {self.title} model...")
        model = GaussianNB()
        model.fit(encoded_corpus, labels)
        return model

    def evaluate_model(self):
        predictions = self.model.predict(self.test)
        precision = precision_score(self.labels_test, predictions)
        recall = recall_score(self.labels_test, predictions)
        accuracy = accuracy_score(self.labels_test, predictions)
        print(f"{self.title}: Acc. - {accuracy * 100:.2f}%, Prec. - {precision * 100:.2f}%, Rec. - {recall * 100:.2f}%")
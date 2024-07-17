import os, string, re, pickle
import pandas as pd
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english')
@st.cache_data
def load_cache_resources():
    return Predictor()

class Utils:
    def path_exists(self, path: str):
        return True if os.path.exists(path) else False
    
    def create_corpus(self, series: pd.Series):
        corpus = []
        for sentence in series.unique():
            words = word_tokenize(sentence)
            corpus.extend(words)
        return set(corpus)


class FeatureEngineer:

    def create_features(self, dataframe):
        dataframe['length'] = dataframe['message'].apply(self.compute_len)
        dataframe['sentences'] = dataframe['message'].apply(self.count_sentences)
        dataframe['words'] = dataframe['message'].apply(self.count_words)
        dataframe['average_words_in_sentence'] = dataframe['message'].apply(self.count_average_sentence_words)
        dataframe['stopwords'] = dataframe['message'].apply(self.count_stopwords)
        dataframe['dollar_sign'] = dataframe['message'].apply(self.count_dollar_sign)
        return dataframe

    def compute_len(self, text: str):
        return len(text)
    
    def count_sentences(self, text: str):
        text = sent_tokenize(text)
        return len(text)
    
    def count_words(self, text: str):
        return len(word_tokenize(text))

    def count_average_sentence_words(self, text: str):
        text = sent_tokenize(text)
        sentences_word_count = [len(word_tokenize(sentence)) for sentence in text]
        return np.average(sentences_word_count)

    def count_stopwords(self, text: str):
        text = word_tokenize(text)
        stopwords_count = len([word for word in text if word in stop_words])
        return stopwords_count
    
    def count_dollar_sign(self, text: str):
        dollar_sign_count = len([char for char in text if char=='$'])
        return dollar_sign_count
    
    def compute_number_count_with_more_than_3_chars(self, text):
        number_pattern = r'\b\d{4,}\b'
        numbers_found = re.findall(number_pattern, text)
        return len(numbers_found)


class Transformer:
    def __init__(self):
        self.punctuations = string.punctuation
        self.stemmer = PorterStemmer()

    def apply_transformation(self, text: str):
        text = self.remove_links(text)
        text = self.remove_punctuations(text)
        text = word_tokenize(text)
        text = self.remove_stopwords(text)
        text = self.do_stemming(text)
        return " ".join(text)
        
    def remove_punctuations(self, text: str):
        text = [char for char in text if char not in self.punctuations]
        return "".join(text)
    
    def remove_links(self, text):
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)
    
    def remove_stopwords(self, text):
        return [word for word in text if word not in stop_words]
    
    def do_stemming(self, text: str):
        return [self.stemmer.stem(word) for word in text]


utils = Utils()
transformer = Transformer()
feature_engineer = FeatureEngineer()

class Predictor:
    def __init__(self):
        cwd = os.getcwd()
        self.model_filepath = os.path.join(cwd, "src", "resources", "model.pkl")
        self.vectorizer_filepath = os.path.join(cwd, "src", "resources", "vectorizer.pkl")
        self._load_resources()

    def _load_resources(self):
        self.model = pickle.load(open(self.model_filepath, "rb"))
        self.vectorizer = pickle.load(open(self.vectorizer_filepath, "rb"))

    def predict_class(self, message: str):
        message = self._prepare_input(message)
        predicted_class = self._inference_get_class(message)
        return predicted_class
    
    def _prepare_input(self, message: str):
        message = transformer.apply_transformation(message)
        input_message = self._create_combine_features(message)
        return input_message
    
    def _create_combine_features(self, message: str):
        length = feature_engineer.compute_len(message)
        number_count = feature_engineer.compute_number_count_with_more_than_3_chars(message)
        vector = self.vectorizer.transform([message]).toarray()

        feats = pd.DataFrame([[length, number_count]], columns=['length', 'count_number_>3_chars']).reset_index(drop=True)
        vector = pd.DataFrame(vector).reset_index(drop=True)
        input_message = pd.concat([feats, vector], axis=1)
        input_message.columns = input_message.columns.astype('str')
        return input_message
    
    def _inference_get_class(self, message: pd.DataFrame):
        prediction = self.model.predict(message)
        prediction = prediction[0]
        return "Spam" if prediction==1 else "Not Spam"
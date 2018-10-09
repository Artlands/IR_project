#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import re
import time

import numpy as np
import pandas as pd
import nltk
import codecs
from sklearn import feature_extraction
import mpld3

import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer("english")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib



# stopwords = nltk.corpus.stopwords.words('english')

class Cluster:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        cluster = Cluster()
        cluster.cluster()

    def tokenize_and_stem(self, raw_text):
        # remove unicode
        text = "".join([x for x in raw_text if ord(x) < 128])
        # use preprocessor to clean the text
        cleaned_text = p.clean(text)
        # convert to lower case and split
        words = cleaned_text.lower().split()
        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stopword_set]
        # join the cleaned words in a list
        cleaned_word_list = " ".join(meaningful_words)

        tokens = [word for sent in nltk.sent_tokenize(cleaned_word_list) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems

    def tokenize_only(self, raw_text):
        # remove unicode
        text = "".join([x for x in raw_text if ord(x) < 128])
        # use preprocessor to clean the text
        cleaned_text = p.clean(text)
        # convert to lower case and split
        words = cleaned_text.lower().split()
        # remove stopwords
        stopword_set = set(stopwords.words("english"))
        meaningful_words = [w for w in words if w not in stopword_set]
        # join the cleaned words in a list
        cleaned_word_list = " ".join(meaningful_words)

        tokens = [word for sent in nltk.sent_tokenize(cleaned_word_list) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    def cluster(self):
        print (f"Reading the data...")
        start = time.time()

        # Read data
        for line in self.in_file:
            data = json.loads(line)
            documents = data['documents']
            texts = data['text']

        print (f"Tokenizing and stemming vocabularies...")
        totalvocab_stemmed = []
        totalvocal_tokenized = []

        for i in texts:
            allwords_stemmed = self.tokenize_and_stem(i)
            totalvocab_stemmed.extend(allwords_stemmed)

            allwords_tokenized = self.tokenize_only(i)
            totalvocal_tokenized.extend(allwords_tokenized)

        print (f"Creating a DataFrame for tokenized vocab and stemmed vocab...")
        vocab_frame = pd.DataFrame({'words': totalvocal_tokenized}, index = totalvocab_stemmed)
        print(f"There are {str(vocab_frame.shape[0])} items in vocab_frame.")
        # Return the first 10 rows
        # print(vocab_frame.head(n = 10))

        # Convert a collection of raw documents to a matrix of TF-IDF features.
        print (f'Creating tfidf matrix...')
        tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, max_features = 100000, min_df = 0.10, use_idf=True, tokenizer = self.tokenize_and_stem, ngram_range=(1, 3))
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        print(f'Tfidf matrix shape: {tfidf_matrix.shape}')
        terms = tfidf_vectorizer.get_feature_names()
        dist = 1 - cosine_similarity(tfidf_matrix)

        #K-means Clustering
        print (f'K-means Clustering...')
        num_clusters = 5
        km = KMeans(n_clusters = num_clusters)
        km.fit(tfidf_matrix)

        joblib.dump(km, 'doc_cluster.pkl')
        clusters = km.labels_.tolist()

        tweets = {'documents': documents, 'texts': texts, 'cluster': clusters}
        frame = pd.DataFrame(tweets, index = [clusters], columns = ['documents', 'cluster'])
        # Write dataframe to a json file
        out_file = self.get_new_file()
        out_file.write('%s\n' % json.dumps(frame.to_json(orient='records')))
        out_file.close()
        # frame['cluster'].value_counts()
        #
        # grouped = frame['documents'].groupby(frame['cluster'])
        # grouped.mean()
        #
        end = time.time()
        print (f"It takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data/dataframes{self.file_ext}"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file {new_file_path}")
        return open(new_file_path, "w")


    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name = argv[1]
            self.in_file = open(self.file_name, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Cluster the data

        Usage:

            $ python clustering.py <file_name>

        """

if __name__ == "__main__":
    Cluster.run()

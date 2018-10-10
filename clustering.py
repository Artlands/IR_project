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
# from sklearn.cluster import KMeans
# from sklearn.externals import joblib

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# import matplotlib as mpl
# from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

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

    def cluster(self):
        start = time.time()

        # Read data
        print (f"Reading the data...")
        for line in self.in_file:
            data = json.loads(line)
            documents = data['documents']
            texts = data['text']

        # Convert a collection of raw documents to a matrix of TF-IDF features.
        print (f'Creating tfidf matrix...')
        tfidf_vectorizer = TfidfVectorizer(max_df = 0.90, max_features = 100000, min_df = 0.10, use_idf=True, tokenizer = self.tokenize_and_stem, ngram_range=(1, 3))
        # Apply tfidf normalization to a sparse matrix of occurrence counts
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        print(f'Tfidf matrix shape: {tfidf_matrix.shape}')
        terms = tfidf_vectorizer.get_feature_names()
        dist = 1 - cosine_similarity(tfidf_matrix)

        #hierarchy document clustering------------------------------------------
        #Ward clustering is an agglomerative clustering method, meaning that at each stage,
        #the pair of clusters with minimum between-cluster distance are merged. Used the precomputed
        # cosine distance matrix (dist) to calclate a linkage_matrix, which then plot as a dendrogram.

        print (f'Hierarchy document clustering...')
        linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(15, 20)) # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=documents);

        plt.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom= False,      # ticks along the bottom edge are off
            top= False,         # ticks along the top edge are off
            labelbottom= False)

        plt.tight_layout() #show plot with tight layout

        #Save figure
        plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
        #-----------------------------------------------------------------------

        end = time.time()
        print (f"It takes {end-start}s")

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

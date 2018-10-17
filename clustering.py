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
# import mpld3

import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer("english")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.cluster import KMeans
from sklearn.externals import joblib

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
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

    def get_cmap(self, n, name='tab10'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def cluster(self):
        start = time.time()

        # Read data
        print (f"Reading the data...")
        for line in self.in_file:
            data = json.loads(line)
            documents = data['place']
            texts = data['text']

        print (f'Tokenizing and stemming words...')
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in texts:
            allwords_stemmed = self.tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
            totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

            allwords_tokenized = self.tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)

        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
        print (f"there are {str(vocab_frame.shape[0])} items in vocab_frame")

        # Convert a collection of raw documents to a matrix of TF-IDF features.
        print (f'Creating tfidf matrix...')
        tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, max_features = 100000, min_df = 0.05, use_idf=True, tokenizer = self.tokenize_and_stem, ngram_range=(1, 3))
        # Apply tfidf normalization to a sparse matrix of occurrence counts
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        print(f'Tfidf matrix shape: {tfidf_matrix.shape}')
        terms = tfidf_vectorizer.get_feature_names()
        similarity_matrix = cosine_similarity(tfidf_matrix)
        dist = 1 - cosine_similarity(tfidf_matrix)

        #K-means clustering
        from sklearn.cluster import KMeans
        # num_clusters = 6
        for i in range(2, 10):
            num_clusters = i
            km = KMeans(n_clusters=num_clusters, n_init=10, max_iter=500, tol=0.0001)
            km.fit(tfidf_matrix)
            joblib.dump(km, 'doc_cluster.pkl')
            # km = joblib.load('doc_cluster.pkl')
            clusters = km.labels_.tolist()

            books = {'book': documents, 'verses': texts}
            frame = pd.DataFrame(books, index = [clusters], columns = ['book', 'cluster'])

            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

            #set up cluster name and color
            cmap = self.get_cmap(num_clusters)
            cluster_colors = {}
            cluster_names = {}
            for i in range(num_clusters):
                cluster_colors[i] = cmap(i)
                cluster_names[i] = ''
                for ind in order_centroids[i, :5]:
                    cluster_names[i] = cluster_names[i] + vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0] +' '

            #Multidimensional scaling
            from sklearn.manifold import MDS
            mds = MDS(n_components = 2, dissimilarity="precomputed", random_state = 1)
            pos = mds.fit_transform(dist) # shape(n_components, n_samples)
            xs, ys = pos[:, 0], pos[:, 1]
            #Visualizing document clusters


            df = pd.DataFrame(dict(x = xs, y = ys, label=clusters, title = documents))
            groups = df.groupby('label')

            # set up plot
            fig, ax = plt.subplots(figsize=(17, 9)) # set size
            ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

            #iterate through groups to layer the plot
            #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
            for name, group in groups:
                ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                        label=cluster_names[name], color=cluster_colors[name],
                        mec='none')
                ax.set_aspect('auto')
                ax.tick_params(\
                    axis = 'x',          # changes apply to the x-axis
                    which ='both',      # both major and minor ticks are affected
                    bottom = False,      # ticks along the bottom edge are off
                    top = False,         # ticks along the top edge are off
                    labelbottom=False)
                ax.tick_params(\
                    axis= 'y',         # changes apply to the y-axis
                    which='both',      # both major and minor ticks are affected
                    left = False,      # ticks along the bottom edge are off
                    top = False,         # ticks along the top edge are off
                    labelleft=False)

            ax.legend(numpoints=1)  #show legend with only 1 point

            #add label in x,y position with the label as the film title
            for i in range(len(df)):
                ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=12)

            # plt.show() #show the plot
            kfile_name = 'kmeans-' + str(num_clusters) + '.png'
            plt.savefig(kfile_name, dpi=200)

        Plot similarity--------------------------------------------------------
        sns.set(style="white")
        sns.set(font_scale=1)
        mask = np.zeros_like(similarity_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = False
        f, ax = plt.subplots(figsize=(11, 9))

        c = sns.heatmap(similarity_matrix, mask=mask, cmap="YlGnBu", vmax=1,
                    square=True, linewidths=0.01,  ax=ax)
        c.set(xlabel='Document ID', ylabel='Document ID')
        # plt.show()
        fig = c.get_figure()
        fig.suptitle('TF-IDF Document Similarity Matrix')

        fig.savefig("similarity.png")

        #hierarchy document clustering------------------------------------------
        #Ward clustering is an agglomerative clustering method, meaning that at each stage,
        #the pair of clusters with minimum between-cluster distance are merged. Used the precomputed
        # cosine distance matrix (dist) to calclate a linkage_matrix, which then plot as a dendrogram.
        # from scipy.cluster import hierarchy
        # print (f'Hierarchy document clustering...')
        # # linkage_matrix = hierarchy.ward(dist) #define the linkage_matrix using ward clustering pre-computed distances
        # # linkage_matrix = hierarchy.single(dist)
        # # linkage_matrix = hierarchy.centroid(dist)
        # linkage_matrix = hierarchy.median(dist)
        # fig, ax = plt.subplots(figsize=(10, 20)) # set size
        # ax = hierarchy.dendrogram(linkage_matrix, orientation="right", labels=documents);
        #
        # plt.tick_params(\
        #     axis= 'x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom= False,      # ticks along the bottom edge are off
        #     top= False,         # ticks along the top edge are off
        #     labelbottom= False)
        #
        # plt.tight_layout() #show plot with tight layout
        # #
        # # #Save figure
        # plt.savefig('median_clusters.png', dpi=200) #save figure as ward_clusters
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

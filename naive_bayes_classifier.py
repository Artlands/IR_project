file_name_train_train#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import re
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

class NBClassifier:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        classifier = NBClassifier()
        classifier.calc_class_word()
        classifier.classify()

    #Clean and preprocess text in tweet
    def preprocess(self, raw_text):
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
        #tokenize strings
        word_tokens = TextBlob(cleaned_word_list).words
        #singularize words
        inflection_words = []
        for word in word_tokens:
            inflection_words.append(word.singularize().lemmatize('v').lower())
         #delete  words with only one letter
        filtered_words = [ w for w in inflection_words if len(w) > 1]
        # clean_data['tokens'] = third_filtered
        cleaned_word_tokens = filtered_words
        return cleaned_word_tokens


    #calucate P(Class), word counter, class vocabulary, create word cloud
    def calc_class_word(self):
        self.negative_class = 0
        self.neutral_class = 0
        self.positive_class = 0

        self.negative_tokens = []
        self.neutral_tokens = []
        self.positive_tokens = []

        self.negative_voc = 0
        self.neutral_voc = 0
        self.positive_voc = 0

        with open(self.file_name_train, "r") as file1:
            for line in file1:
                tweet = json.loads(line)

                if tweet['Class'] == '-1':
                    self.negative_class += 1
                    self.negative_tokens = self.negative_tokens + self.preprocess(tweet['Text'])
                if tweet['Class'] == '0':
                    self.neutral_class += 1
                    self.neutral_tokens = self.neutral_tokens + self.preprocess(tweet['Text'])
                if tweet['Class'] == '1':
                    self.positive_class += 1
                    self.positive_tokens = self.positive_tokens + self.preprocess(tweet['Text'])

        self.p_negative = self.negative_class / (self.negative_class + self.neutral_class + self.positive_class)
        self.p_neutral = self.neutral_class / (self.negative_class + self.neutral_class + self.positive_class)
        self.p_positive = self.positive_class / (self.negative_class + self.neutral_class + self.positive_class)

        self.negative_counter = Counter(self.negative_tokens)
        self.neutral_counter = Counter(self.neutral_tokens)
        self.positive_counter = Counter(self.positive_tokens)

        self.negative_voc = len(set(self.negative_tokens))
        self.neutral_voc = len(set(self.neutral_tokens))
        self.positive_voc = len(set(self.positive_tokens))

        most_occur_neg = self.negative_counter.most_common(100)
        most_occur_neu = self.neutral_counter.most_common(100)
        most_occur_pos = self.positive_counter.most_common(100)
        text_neg = ' '.join([word[0] for word in most_occur_neg])
        text_neu = ' '.join([word[0] for word in most_occur_neu])
        text_pos = ' '.join([word[0] for word in most_occur_pos])

        # lower max_font_size
        wordcloud_neg = WordCloud(max_font_size=50).generate(text_neg)
        wordcloud_neu = WordCloud(max_font_size=50).generate(text_neu)
        wordcloud_pos = WordCloud(max_font_size=50).generate(text_pos)

        plt.figure()
        plt.imshow(wordcloud_neg, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('negative_wordcloud', dpi=200)

        plt.figure()
        plt.imshow(wordcloud_neu, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('neutral_wordcloud', dpi=200)

        plt.figure()
        plt.imshow(wordcloud_pos, interpolation="bilinear")
        plt.axis("off")
        plt.savefig('positive_wordcloud', dpi=200)


    def make_predictions(self, cls, text):
        negative_prediction = 1
        neutral_prediction = 1
        positive_prediction = 1

        text_tokens = self.preprocess(text)

        for word in text_tokens:
            negative_prediction *= self.p_negative * (self.negative_counter[word] + 1)/(len(self.negative_tokens) + self.negative_voc)
            neutral_prediction *= self.p_neutral * (self.neutral_counter[word] + 1)/(len(self.neutral_tokens) + self.neutral_voc)
            positive_prediction *= self.p_positive * (self.positive_counter[word] + 1)/(len(self.positive_tokens) + self.positive_voc)

        # maximum = max(negative_prediction, neutral_prediction, positive_prediction)
        if cls == "positive":
            return positive_prediction/(negative_prediction + neutral_prediction + positive_prediction)
        if cls == "neutral":
            return neutral_prediction/(negative_prediction + neutral_prediction + positive_prediction)
        if cls == "negative":
            return negative_prediction/(negative_prediction + neutral_prediction + positive_prediction)
        return 0

    def classify(self):
        self.actual = []
        self.prediction_pos = []
        self.prediction_neu = []
        self.prediction_neg = []
        with open(self.file_name_test, "r") as file2:
            for line in file2:
                tweet = json.loads(line)

                # record the actual class
                self.actual.append(int(tweet['Class']))

                # record the predictions score for three classes
                self.prediction_pos.append(self.make_predictions("positive",tweet['Text']))
                self.prediction_neu.append(self.make_predictions("neutral",tweet['Text']))
                self.prediction_neg.append(self.make_predictions("negative",tweet['Text']))

        # Generate the roc curve using scikits-learn.
        fpr_pos, tpr_pos, thresholds_pos = metrics.roc_curve(self.actual, self.prediction_pos, pos_label= 1)
        fpr_neu, tpr_neu, thresholds_neu = metrics.roc_curve(self.actual, self.prediction_neu, pos_label= 0)
        fpr_neg, tpr_neg, thresholds_neg = metrics.roc_curve(self.actual, self.prediction_neg, pos_label= -1)

        plt.figure()
        lw = 2
        plt.plot(fpr_pos, tpr_pos, color='coral',
                 lw=lw, label='ROC curve of Positive class(area = %0.2f)' % metrics.auc(fpr_pos,tpr_pos))
        plt.plot(fpr_neu, tpr_neu, color='olivedrab',
                 lw=lw, label='ROC curve of Neutral class(area = %0.2f)' % metrics.auc(fpr_neu, tpr_neu))
        plt.plot(fpr_neg, tpr_neg, color='steelblue',
                 lw=lw, label='ROC curve of Negative class(area = %0.2f)' % metrics.auc(fpr_neg, tpr_neg))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.axes().set_aspect('equal')
        plt.legend(loc="lower right")
        plt.show()

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name_train = argv[1]
            self.file_name_test = argv[2]
            self.working_dir = os.getcwd()
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Naive Bayes Classifier

        Usage:

            $ python naive_bayes_classifier.py <training_file_name> <testing_file_name>

        """

if __name__ == "__main__":
    NBClassifier.run()

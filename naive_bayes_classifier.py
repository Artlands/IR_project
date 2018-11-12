#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import re
import time
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
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
            inflection_words.append(word.singularize().lower())
         #delete  words with only one letter
        filtered_words = [ w for w in inflection_words if len(w) > 1]
        # clean_data['tokens'] = third_filtered
        cleaned_word_tokens = filtered_words
        return cleaned_word_tokens


    #calucate P(Class), word counter, class vocabulary
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

        for line in self.in_file:
            file = json.loads(line)
            if file['Class'] == '-1':
                self.negative_class += 1
                self.negative_tokens = self.negative_tokens + self.preprocess(file['Text'])
            if file['Class'] == '0':
                self.neutral_class += 1
                self.neutral_tokens = self.neutral_tokens + self.preprocess(file['Text'])
            if file['Class'] == '1':
                self.positive_class += 1
                self.positive_tokens = self.positive_tokens + self.preprocess(file['Text'])
        self.in_file.close()

        self.p_negative = self.negative_class / (self.negative_class + self.neutral_class + self.neutral_class)
        self.p_neutral = self.neutral_class / (self.negative_class + self.neutral_class + self.neutral_class)
        self.p_positive = self.positive_class / (self.negative_class + self.neutral_class + self.neutral_class)

        self.negative_counter = Counter(self.negative_tokens)
        self.neutral_counter = Counter(self.neutral_tokens)
        self.positive_counter = Counter(self.positive_tokens)

        self.negative_voc = len(set(self.negative_tokens))
        self.neutral_voc = len(set(self.neutral_tokens))
        self.positive_voc = len(set(self.positive_tokens))

    def classify(self):
        negative_prediction = 1
        neutral_prediction = 1
        positive_prediction = 1

        text = "this is the most happiest moment in my life"

        text_tokens = self.preprocess(text)
        for word in text_tokens:
            negative_prediction *= self.p_negative * (self.negative_counter[word] + 1)/(len(self.negative_tokens) + self.negative_voc)
            neutral_prediction *= self.p_neutral * (self.neutral_counter[word] + 1)/(len(self.neutral_tokens) + self.neutral_voc)
            positive_prediction *= self.p_positive * (self.positive_counter[word] + 1)/(len(self.positive_tokens) + self.positive_voc)

        print(f"Negative prediction:\t{negative_prediction}")
        print(f"Neutral prediction:\t{neutral_prediction}")
        print(f"Positive prediction:\t{positive_prediction}")

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
        Naive Bayes Classifier

        Usage:

            $ python naive_bayes_classifier.py <training_file_name>

        """

if __name__ == "__main__":
    NBClassifier.run()

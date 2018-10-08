#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time
import math
from collections import defaultdict

class TfidfCalculator:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        calculator = TfidfCalculator()
        calculator.calculate()

    # calculate term counts
    def set_weights(self, tokens):
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        return(token_counts)

    def computeTF(self, wordDict, allterms):
        tfDict = {}
        tokenCount = len(allterms)
        for word, count in wordDict.items():
            tfDict[word] = 1 + math.log10(count) #log frequent weight
        return tfDict

    def computeIDF(self, docList):
        idfDict = {}
        N = len(docList)

        # collect all tokens from every documents, delete duplicated one
        all_tokens_tmp = []
        for i in range(0, N):
            all_tokens_tmp = all_tokens_tmp + docList[i]["tokens"]
        all_tokens_set = set(all_tokens_tmp)
        tokens = [i for i in all_tokens_set]

        # calculate number of documents containing token.
        for token in tokens:
            idfDict[token] = 0
            for doc in docList:
                if token in doc["tokens"]:
                    idfDict[token] += 1
        for token, val in idfDict.items():
            idfDict[token] = math.log10(N / float(val))

        return idfDict

    def computeTFIDF(self, tfDict, idfDict):
        tfidf = {}
        for token, val in tfDict.items():
            tfidf[token] = val * idfDict[token]
        return tfidf


    def calculate(self):
        print (f"Calculating tf-idf score of each document in {os.path.join(self.working_dir, self.file_base_name+self.file_ext)}")
        start = time.time()
        out_file = self.get_new_file()
        docList = []
        with self.in_file as f:
            line = f.readline()
            documents = json.loads(line)
            idfDict = self.computeIDF(documents)
            for doc in documents:
                new_doc={}
                new_doc['place'] = doc['place']
                wordDict = self.set_weights(doc['tokens'])
                tfDict = self.computeTF(wordDict, doc['tokens'] )
                new_doc['tfidf'] = self.computeTFIDF(tfDict, idfDict)
                new_doc['sorted_terms'] = sorted(wordDict.items(), key = lambda kv: kv[1], reverse = True)
                new_doc['sorted_tfidf'] = sorted(new_doc['tfidf'].items(), key = lambda kv: kv[1], reverse = True)
                docList.append(new_doc)

        out_file.write('%s\n' % json.dumps(docList))

        out_file.close()

        end = time.time()
        print (f"Calculated all terms tfidf, it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"{self.file_base_name}_tfidf{self.file_ext}"
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
        Calculate tf-idf score of each document in a file.

        Usage:

            $ python tfidf_calculator.py <file_name>

        """

if __name__ == "__main__":
    TfidfCalculator.run()

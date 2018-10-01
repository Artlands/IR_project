#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time
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
            tfDict[word] = count/float(tokenCount)
        return tfDict

    # def computeIDF(self, docList):
    #     import math
    #     idfDict = {}
    #     N = len(docList)
    #     idfDict = dict.fromkeys(docList['tokens'].keys(), 0)
    #     for doc in docList:
    #         for word, val in doc.items():
    #             if val > 0:
    #                 idfDict[word] += 1
    #     for word, val in idfDict.items():
    #         idfDict[word] = math.log10(N/float(val))
    #     return idfDict

    def calculate(self):
        print ("Calculating tf-idf score of each document in %s\n" % (os.path.join(self.working_dir, self.file_base_name+self.file_ext)))
        start = time.time()
        out_file = self.get_new_file()
        docList = []
        with self.in_file as f:
            line = f.readline()
            documents = json.loads(line)
            for doc in documents:
                new_doc={}
                new_doc['place'] = doc['place']
                wordDict = self.set_weights(doc['tokens'])
                new_doc['token_tf'] = self.computeTF(wordDict, doc['tokens'] )
                docList.append(new_doc)

        out_file.write('%s\n' % json.dumps(docList))

        out_file.close()

        end = time.time()
        print ("Calculated all terms accounts, it takes %fs\n" % (end-start))

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = "%s_tfidf%s" % (self.file_base_name, self.file_ext)
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print ("Creating file %s" % (new_file_path))
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

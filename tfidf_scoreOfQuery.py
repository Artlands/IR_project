#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time

import matplotlib.pyplot as plt

class QueryScore:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        calculater = QueryScore()
        calculater.cal()
        calculater.plot()

    def cal(self):

        print ("Calculating tf-idf score of query in the documents...")
        start = time.time()

        out_file = self.get_new_file()
        queryScoreList = []
        for line in self.in_file:
            data = json.loads(line)
            for doc in data:
                score = 0.0
                queryScore = {}
                for item in self.querylist:
                    if item in doc['tfidf']:
                        score += doc['tfidf'][item]
                queryScore['document'] = doc['place']
                queryScore['score'] = score
                queryScoreList.append(queryScore)
                self.sortedList = sorted(queryScoreList, key = lambda k: k['document'])

        out_file.write('%s\n' % json.dumps(self.sortedList))
        out_file.close()

        end = time.time()
        print ("The query scores for %s is saved to one json file. it takes %fs\n" % (self.querylist, (end-start)))

    def plot(self):
        x = [x['document'] for x in self.sortedList]
        y = [y['score'] for y in self.sortedList]
        plt.bar(x, y, label='terms tfidf score')
        plt.xlabel('documents')
        plt.ylabel('Tfidf-score')
        plt.title("Tfidf score for the term: " + ' '.join(self.querylist))
        # plt.legend()
        plt.show()


    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = "%s_queryScore%s" % (self.file_base_name, self.file_ext)
        new_file_path = os.path.join(self.working_dir, new_file_name)
        strCreatFile = "Creating file %s" % (new_file_path)
        print (strCreatFile)
        return open(new_file_path, "w")

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name = argv[1]
            self.in_file = open(self.file_name, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
            self.querylist = []
            for i in range(2, len(argv)):
                self.querylist.append(argv[i].lower())
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Calculate tf-idf score of query in the documents

        Usage:

            $ python tf-idf.py <tfidf_file_name> [query definition]

        """

if __name__ == "__main__":
    QueryScore.run()

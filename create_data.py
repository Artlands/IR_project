#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time

class CreateData:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        creator = CreateData()
        creator.create()

    def create(self):

        print (f"Create data for dataFrame")
        start = time.time()

        out_file = self.get_new_file()
        result = {}
        documents = []
        text = []
        for line in self.in_file_1:
            data1 = json.loads(line)
            for doc in data1:
                documents.append(doc['place'] + '-0')
                text.append(doc['text'])

        for line in self.in_file_2:
            data2 = json.loads(line)
            for doc in data2:
                documents.append(doc['place'] + '-1')
                text.append(doc['text'])

        result['documents'] = documents
        result['text'] = text
        print(f"Documents number is {len(documents)}")

        out_file.write('%s\n' % json.dumps(result))
        out_file.close()

        end = time.time()
        print (f"Created data for dataFrame in one json file. it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data/data{self.file_ext}"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file {new_file_path}")
        return open(new_file_path, "w")

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name_1 = argv[1]
            self.in_file_1 = open(self.file_name_1, "r")
            self.file_name_2 = argv[2]
            self.in_file_2 = open(self.file_name_2, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name_1)
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Seperate file into several documents

        Usage:

            $ python tfidf_documents.py <file_name>

        """

if __name__ == "__main__":
    CreateData.run()

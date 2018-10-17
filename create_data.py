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

        print (f"Creating data for analysis")
        start = time.time()

        out_file = self.get_new_file()
        documents = {}
        docList = []
        place_list = []
        text_list = []
        data_last = {}
        for line in self.in_file:
            data = json.loads(line)
            document = {}
            document['place'] = data['place']
            document['text'] = data['text']

            if document['place'] in documents:
                documents[document['place']]= documents[document['place']] + document['text']
            else:
                documents[document['place']] = document['text']

        for key, value in documents.items():
            docu_item = {}
            docu_item['place'] = key
            docu_item['text'] = value
            docList.append(docu_item)
            docList = sorted(docList, key = lambda kv: kv['place'], reverse = False)

        for doc in docList:
            place_list.append(doc['place'])
            text_list.append(doc['text'])
        data_last['place'] = place_list
        data_last['text'] = text_list

        print(f"Documents number is {len(place_list)}")
        print(f"Texts number is {len(text_list)}")

        out_file.write('%s\n' % json.dumps(data_last))
        out_file.close()

        end = time.time()
        print (f"Created data in one json file. it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data/data{self.file_ext}"
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
        Create data for analysis

        Usage:

            $ python tfidf_documents.py <file_name>

        """

if __name__ == "__main__":
    CreateData.run()

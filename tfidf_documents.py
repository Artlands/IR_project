#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import time

class TfidfCal:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        calculater = TfidfCal()
        calculater.cal()

    def cal(self):

        strInProcess = "Seperating %s, into documents..." % (os.path.join(self.working_dir, self.file_base_name + self.file_ext))
        print (strInProcess)
        start = time.time()

        out_file = self.get_new_file()
        documents = {}
        docList = []
        for line in self.in_file:
            data = json.loads(line)
            document = {}
            document['place'] = data['place']
            document['tokens'] = data['tokens']

            if document['place'] in documents:
                documents[document['place']].extend(document['tokens'])
            else:
                documents[document['place']] = document['tokens']

        for key, value in documents.items():
            docu_item = {}
            docu_item['place'] = key
            docu_item['tokens'] = value
            docList.append(docu_item)

        out_file.write('%s\n' % json.dumps(docList))
        out_file.close()

        end = time.time()
        print ("Created %s documents in one json file. " % (len(documents)))
        print ("it takes %fs\n" % (end-start))

    def find_record(self, documents, place):
        for dict in documents:
            if dict['place'] == place:
                return dict['tokens']
            return False

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = "%s_documents%s" % (self.file_base_name, self.file_ext)
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
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Seperate file into several documents

        Usage:

            $ python tf-idf.py <file_name>

        """

if __name__ == "__main__":
    TfidfCal.run()

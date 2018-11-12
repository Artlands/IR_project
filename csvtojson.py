#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import csv
import time

class Csvtojson:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        creator = Csvtojson()
        creator.create()

    def create(self):

        print (f"CSV to JSON")
        start = time.time()

        out_file = self.get_new_file()

        fieldnames = ("Place", "Text", "Class")
        reader = csv.DictReader(self.in_file, fieldnames)
        for row in reader:
            json.dump(row, out_file)
            out_file.write('\n')

        out_file.close()

        end = time.time()
        print (f"Convert csv to json. it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data/classed_json.json"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file {new_file_path}")
        return open(new_file_path, "w")

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name = argv[1]
            self.in_file = open(self.file_name, "r", newline='', encoding='utf-8')
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Convert csv to json

        Usage:

            $ python csvtojson.py <cvs_file_name>

        """

if __name__ == "__main__":
    Csvtojson.run()

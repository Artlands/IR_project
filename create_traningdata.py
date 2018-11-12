#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import csv
import time

class CreateData:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        creator = CreateData()
        creator.create()

    def create(self):

        print (f"Creating Training data")
        start = time.time()

        out_file = self.get_new_file()

        with self.in_file as file:
            training_data = [json.loads(next(file)) for x in range(0, self.training_num)]

        csvwriter = csv.writer(out_file)
        count = 0
        for data in training_data:
            if count == 0:
                header = data.keys()
                csvwriter.writerow(header)
                count += 1
            csvwriter.writerow(data.values())

        print(f"Created {self.training_num} documents of training dataset")

        out_file.close()

        end = time.time()
        print (f"Created Training data in one json file. it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"data/training_data.csv"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file {new_file_path}")
        return open(new_file_path, "w")

    def parse_args(self,argv):
        """parse args and set up instance variables"""
        try:
            self.file_name = argv[1]
            self.training_num = int(argv[2])
            self.in_file = open(self.file_name, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
        except:
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Create training data for analysis

        Usage:

            $ python create_trainingdata.py <file_name> <documents_num>

        """

if __name__ == "__main__":
    CreateData.run()

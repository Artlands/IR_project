#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import sys
import os
import json

class JsonCleaner:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        cleaner = JsonCleaner()
        cleaner.clean()

    def clean(self):
        print "Cleaning %s, only keep attribute(s): %s" % (os.path.join(self.working_dir, self.file_base_name + self.file_ext), self.keep_attr)

        out_file = self.get_new_file()
        for line in self.in_file:
            data = json.loads(line)
            clean_data = {}
            for attr in self.keep_attr:
                if attr in data:
                    clean_data[attr] = data[attr]
            out_file.write('%s\n' % json.dumps(clean_data))

        out_file.close()

        print "Created a cleaned json!"

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = "%s_cleaned%s" % (self.file_base_name, self.file_ext)
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print "creating file %s" % (new_file_path)
        return open(new_file_path, "w")

    def parse_args(self, argv):
        """parse args and set up instance variables"""
        try:
            self.keep_attr=[]
            len_argv = len(argv)
            if len_argv > 2:
                for i in range(2, len_argv):
                    self.keep_attr.append(argv[i])
            self.file_name = argv[1]
            self.in_file = open(self.file_name, "r")
            self.working_dir = os.getcwd()
            self.file_base_name, self.file_ext = os.path.splitext(self.file_name)
        except:
            print self.usage()
            sys.exit(1)

    def usage(self):
        return """
        Clean the json file, only keep specific attributes assigned by user.
        Usage:

            $ python json_cleaner.py <file_name> [attr1, attr2, ...]

        """


if __name__ == "__main__":
    JsonCleaner.run()

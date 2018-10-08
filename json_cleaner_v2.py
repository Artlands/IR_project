#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import sys
import os
import json
import re
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.NUMBER)

class JsonCleaner:

    def __init__(self):
        self.parse_args(sys.argv)

    @staticmethod
    def run():
        cleaner = JsonCleaner()
        cleaner.clean()

    # def preprocess(self, raw_text):
    #     # remove unicode
    #     text = "".join([x for x in raw_text if ord(x) < 128])
    #     # use preprocessor to clean the text
    #     cleaned_text = p.clean(text)
    #     # convert to lower case and split
    #     words = cleaned_text.lower().split()
    #     # remove stopwords
    #     stopword_set = set(stopwords.words("english"))
    #     meaningful_words = [w for w in words if w not in stopword_set]
    #     # join the cleaned words in a list
    #     cleaned_word_list = " ".join(meaningful_words)
    #     #tokenize strings
    #     word_tokens = TextBlob(cleaned_word_list).words
    #     #singularize and lemmatization words
    #     inflection_words = []
    #     for word in word_tokens:
    #         inflection_words.append(word.singularize().lemmatize('v').lower())
    #      #delete  words with only one letter
    #     filtered_words = [ w for w in inflection_words if len(w) > 1]
    #     # clean_data['tokens'] = third_filtered
    #     cleaned_word_tokens = filtered_words
    #     return cleaned_word_tokens


    def clean(self):
        print(f"Cleaning {os.path.join(self.working_dir, self.file_base_name + self.file_ext)}, only keep attribute(s): {self.keep_attr}")
        start = time.time()
        out_file = self.get_new_file()
        for line in self.in_file:
            data = json.loads(line)
            clean_data = {}
            if 'text' in data and 'place' in data and data['place'] != None and self.check_place(data['place']['full_name']): # only keep tweets with valid text and place
                for attr in self.keep_attr:
                    if attr == 'text':
                        clean_data['text'] = data['text']
                        # clean_data['tokens'] = self.preprocess(data['text'])
                    elif attr == 'place':
                        clean_data['place'] = self.trans_place(data['place']['full_name'])  #simplify place info
                    else:
                        clean_data[attr] = data[attr]
                out_file.write('%s\n' % json.dumps(clean_data))

        out_file.close()
        end = time.time()
        print(f"Created a cleaned json! it takes {end-start}s")

    def get_new_file(self):
        """return a new file object ready to write to """
        new_file_name = f"{self.file_base_name}_cleaned{self.file_ext}"
        new_file_path = os.path.join(self.working_dir, new_file_name)
        print (f"Creating file{new_file_path}")
        return open(new_file_path, "w")

    def check_place(self, place):
        """check if place is in USA"""
        validPlace = ["AL","AK","AZ","AR","CA","CO" "CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI",
            "MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK",
            "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY", "USA"]
        strSplited = place.split(", ")
        if len(strSplited) == 2 and strSplited[1] in validPlace:
            return True
        else:
            return False

    def trans_place(self, place):
        """nomralize state name, eg Iowa, USA -> IA; San Francisco, CA -> CA"""
        states = {
            'Alabama': "AL","Alaska" : "AK","Arizona" : "AZ","Arkansas" : "AR","California" : "CA","Colorado" : "CO","Connecticut" : "CT",
            "Delaware" : "DE","Florida" : "FL","Georgia" : "GA","Hawaii" : "HI","Idaho" : "ID","Illinois" : "IL","Indiana" : "IN","Iowa" : "IA",
            "Kansas" : "KS","Kentucky" : "KY","Louisiana" : "LA","Maine" : "ME","Maryland" : "MD","Massachusetts" : "MA","Michigan" : "MI",
            "Minnesota" : "MN","Mississippi" : "MS","Missouri" : "MO","Montana" : "MT","Nebraska" : "NE","Nevada" : "NV","New Hampshire" : "NH",
            "New Jersey" : "NJ","New Mexico" : "NM","New York" : "NY","North Carolina" : "NC","North Dakota" : "ND","Ohio" : "OH","Oklahoma" : "OK",
            "Oregon" : "OR","Pennsylvania" : "PA","Rhode Island" : "RI","South Carolina" : "SC","South Dakota" : "SD","Tennessee" : "TN",
            "Texas" : "TX","Utah" : "UT","Vermont" : "VT","Virginia" : "VA","Washington" : "WA","West Virginia" : "WV","Wisconsin" : "WI","Wyoming" : "WY"
        }
        strSplited = place.split(", ")
        if strSplited[1] == 'USA':
            if strSplited[0] != "District of Columbia" and strSplited[0] in states :
                newplace = states[strSplited[0]]
            else:
                newplace = 'MD'
        else:
            newplace = strSplited[1]

        return newplace

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
            print (self.usage())
            sys.exit(1)

    def usage(self):
        return """
        Clean the json file, only keep specific attributes assigned by user.
        Usage:

            $ python json_cleaner.py <file_name> [attr1, attr2, ...]

        """


if __name__ == "__main__":
    JsonCleaner.run()

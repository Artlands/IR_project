# IR_project

This Project is written for Information Retrieval course (CS5364). The main purpose of this project is to practice diferent analysis methods on a dataset.  We choose 1million of tweets crawled from internet as our dataset. To analysis this big dataset(the data size is about 5.09 GB) more efficiently, we build several tools.

- file_splitter.py. 
  - Split a large file into many smaller files with set number of rows.
- json_cleaner.py. 
  - Clean the json file(tweets are stored in json format), only keep specific attributes assigned by user, basically we keep "place" and "text" informtion. 
  - Normalize the state name to uniform format, e.g. Iowa, USA -> IA, San Francisco, CA -> CA.
  - Preprocess (including remove unicode, hashtag, link and emoij, remove stopwords, singularize and lemmatize words etc.) text and store them in tokens.
  - Save the output file as 'filename'_cleaned.json. 
- tfidf_documents.py.
  - Classify the documents according to the states' name.
  - Save the output file as 'filename'_cleaned_documents.json.
- tfidf_calculator.py.
  - Calculator tf-idf score of every word of each document.
  - Save the output file as 'filename'_cleaned_documents_tfidf.json.
- tfidf_scoreOfQuery.py
  - Calculate the tf-idf score of a query specified by user.
  - Save the output file as 'filename'_cleaned_documents_tfidf_queryScore.json.
  - Plot the tf-idf score of each document.
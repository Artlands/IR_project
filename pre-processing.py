import json
import nltk
import preprocessor as pre
pre.set_options(pre.OPT.URL, pre.OPT.MENTION, pre.OPT.EMOJI,
    pre.OPT.SMILEY, pre.OPT.HASHTAG, pre.OPT.NUMBER, pre.OPT.RESERVED)

from textblob import TextBlob
from nltk.corpus import stopwords

outfile = open('./data/tweets_cleaned.json', 'w')
with open('./data/tweets_09_15_10000.json', 'r') as f:

#-------------- show single tweet structure --------------
    # line = f.readline() # read only the first tweet/line
    # tweet = json.loads(line) # load it as Python dict
    # print(json.dumps(tweet, indent=4)) # pretty-print
    # print(tweet['text'])
#-------------- simplify json files  --------------
    for line in f:
        tweet = json.loads(line)
        data = {}
        # data['created_at'] = tweet['created_at']
        # data['text'] = tweet['text']
        if tweet.get('text'):
            data['text'] = tweet['text']
        # data['location'] = tweet['user']['location']
        # data['country'] = tweet['place']['country']
        # data['coordinates'] = tweet['place']['bounding_box']['coordinates']
        json.dump(data, outfile)

        # if 'text' in tweet:
        #     text = tweet['text'].lower()
        #     cleanedText = pre.clean(text)
        #     tokens = TextBlob(cleanedText).words.singularize().lemmatize()
        #     clean_tokens = tokens[:]
        #     for token in tokens:
        #         if token in stopwords.words('english'):
        #             clean_tokens.remove(token)
        #     print(clean_tokens)
outfile.close()

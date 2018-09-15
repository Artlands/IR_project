import json

with open('tweets_2018_0.json', 'r') as file:
    line = file.readline()
    tweet = json.loads(line)
    print(json.dumps(tweet, indent=4))

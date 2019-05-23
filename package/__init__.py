import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import WordPunctTokenizer
from urllib3.connectionpool import xrange

df = pd.read_csv("../testingData/training1600000processednoemoticon.csv", header=None, usecols=[0, 5],
                 names=['sentiment', 'text'], encoding="ISO-8859-1")
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                 "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                 "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                 "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                 "mustn't": "must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in xrange(0, len(df)):
    if (i + 1) % 100000 == 0:
        print("Tweets %d of %d has been processed" % (i + 1, len(df)))
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

clean_df = pd.DataFrame(clean_tweet_texts, columns=['text'])
clean_df['target'] = df.sentiment
clean_df.to_csv('clean_tweet.csv', encoding='utf-8')

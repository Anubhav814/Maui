from __future__ import division
from collections import defaultdict
import csv

import json
import os
import glob
import re
import math
import operator
import time
import numpy
import tensorflow
import scipy
import sklearn

dirs=os.listdir("fao")

dirs=sorted(dirs)
keyfiles = [f for f in dirs  if f.endswith('.key')]
txtfiles = [f for f in dirs if f.endswith('.txt')]

content=[]
allkeys=[]

# Read all keywords and all the words in all the files

for i in range(len(keyfiles)):
	allkeys.append(open("fao/"+keyfiles[i],"r").read())
	#content.append(open("fao/"+txtfiles[i],"r", encoding="utf-8").read())
	content.append(open("fao/"+txtfiles[i],"rb").read().decode("ascii","ignore"))


#content=content.decode("utf-8")
#content=content.encode("ascii","ignore")
# Split to get individual keywords in all .key files
keywords=[]
for j in range(len(keyfiles)):
	keywords.append(allkeys[j].split('\n'))

# Remove non alphabetical characters
cleantext=[]
regex=re.compile('[^a-zA-Z\ ]')
#regex = re.compile('[,\.!?/:()0-9;\n]')
for k in range(len(txtfiles)):
	cleantext.append(regex.sub('',content[k].lower()))
	#cleantext=re.sub('[^a-z\ ]+','',open('/home/anubhav/Desktop/naum/saurav-tagging/data/fao/'+txtfiles[k],"r").read())

'''allwords=[]
for n in range(len(txtfiles)):
	allwords.append(cleantext[n].split())

print(allwords[1][1])

wordsall=[]
for n in range(len(txtfiles)):
	for m in range(len(allwords[n])):
		wordsall.append(allwords[n][m])
'''
	



from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0, stop_words = 'english')


tfidf_matrix =  tf.fit_transform(content)
feature_names = tf.get_feature_names()

#print (len(feature_names))

dense = tfidf_matrix.todense()
#print(len(dense[0].tolist()[0]))

fileid = dense[0].tolist()[0]
phrase_scores = [pair for pair in zip(range(0, len(fileid)), fileid) if pair[1] > 0]

#print(len(phrase_scores))



#print(sorted(phrase_scores, key=lambda t: t[1] * -1)[:5])

sorted_phrase_scores = sorted(phrase_scores, key=lambda t: t[1] * -1)
for phrase, score in [(feature_names[word_id], score) for (word_id, score) in sorted_phrase_scores][:20]:
   print('{0: <20} {1}'.format(phrase, score))


with open("tfid.csv", "w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["FileId", "Phrase", "Score"])

    doc_id = 0
    for doc in tfidf_matrix.todense():
        print ("Document %d" %(doc_id))
        word_id = 0
        for score in doc.tolist()[0]:
            if score > 0.1:
                word = feature_names[word_id]
                writer.writerow([doc_id+1, word.encode("utf-8"), score])
            word_id +=1
        doc_id +=1
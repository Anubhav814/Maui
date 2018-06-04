from __future__ import division
import json
import os
import glob
import time
import re
import math
import operator
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

import rake


####################### TRAINING ###############################

num_files = 140
# Preparing Data to be used
dirs=os.listdir("fao")
print ("Retrieving data...")


start = time.time()

dirs=sorted(dirs)
keyfiles = [f for f in dirs  if f.endswith('.key')]
txtfiles = [f for f in dirs if f.endswith('.txt')]

content=[]
allkeys=[]

for i in range(len(keyfiles)):
	allkeys.append(open("fao/"+keyfiles[i],"r").read())
	content.append(open("fao/"+txtfiles[i],"rb").read().decode("ascii","ignore"))

keywords=[]
for j in range(len(keyfiles)):
	keywords.append(allkeys[j].split('\n'))

cleantext=[]
regex=re.compile('[^a-zA-Z\ ]')
#regex = re.compile('[,\.!?/:()0-9;\n]')
for k in range(len(txtfiles)):
	cleantext.append(regex.sub('',content[k].lower()))
	#cleantext=re.sub('[^a-z\ ]+','',open('/home/anubhav/Desktop/naum/saurav-tagging/data/fao/'+txtfiles[k],"r").read())

allwords=[]
for n in range(len(txtfiles)):
	allwords.append(cleantext[n].split())

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
# Stemming the words
## Comment this if stemming not needed ##
'''stemmed_words=[]
for i in range(len(txtfiles)):
	b1=[]
	for word in allwords[i]:
		word = stemmer.stem(word)
		if word not in stop_words:
			b1.append(word)
	stemmed_words.append(b1)
allwords = stemmed_words
## Stemming ends here ##
'''
print(time.time()-start)

print ("Data retrieved and pre-processed")





rake_object = rake.Rake("SmartStoplist.txt", 5, 3, 4)

keywords = rake_object.run(allwords[0])
#print ("Keywords:", keywords)

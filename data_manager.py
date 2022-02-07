"""
This file is meant to read in a selection of documents from the 20 news database and 
	format that data into a dataframe
This way we are able to preform data cleaning and Topical Modeling on this data 

Sources used:
	https://towardsdatascience.com/end-to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0

Usage:
	dm.parseGroup("./directory_name") - reads in data in target directory
	dm.tokenize() - tokenizes the data into a corpus
	dm.load() - helper function for pickle load
	dm.save() - helper function for pickle dump
	dm.genSimLDA(num_topics) - runs LDA and generates HTML representation of results


"""
import pandas as pd
import pickle
import os
import gensim
from gensim.utils import simple_preprocess
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
from pprint import pprint
import pickle 
import pyLDAvis
import pyLDAvis.gensim_models 
import IPython

class data_manager(object):
	def __init__(self,name):
		self.name = name
		self.df = pd.DataFrame(columns = ["group","id", "text"])
		self.stop_words = stopwords.words('english')
		self.stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'com', 'org'])
		self.corpus = None
		self.id2word = None
	
	def add(self,group,ID,text):#adds a new entry to the data frame
		newData = {"group":group,"id":ID,"text":text}
		self.df = self.df.append(newData,ignore_index = True)

	def parseGroup(self,group):#parses the read in of a directory full of text data
		for doc in os.listdir(group):
			self.add(group,doc,self.getText(doc,group))

	def getText(self,doc,group):#helper function to get the text from a target file and strips bad chars
		docText = ""
		f = open(os.path.join(group, doc), "r")
		f.seek(0)
		for line in f: # goes through all lines
			for word in line.lower().split(): # goes throuugh all words per line
				pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
				docText += pureWord
				docText += " "
		return docText

	def print(self):#data_manager print function
		print(str(self.name) + " dataframe:")
		print(self.df)

	def output(self):#returns the df directly 
		return self.df

	def save(self, fileName = None):#helper function for pickle dump
		if fileName is None:
			fileName = self.name
		pickle.dump(self.df, open( str(fileName)+"_df.p", "wb" ) )
		pickle.dump(self.corpus, open( str(fileName)+"_corpus.p", "wb" ) )
		pickle.dump(self.id2word, open( str(fileName)+"_id2word.p", "wb" ) )
		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.df = pickle.load(open( str(fileName)+"_df.p", "rb" ) )
		self.corpus = pickle.load( open( str(fileName)+"_corpus.p", "rb" ) )
		self.id2word = pickle.load( open( str(fileName)+"_id2word.p", "rb" ) )
		print(str(fileName) + " loaded")

	def sent_to_words(self,sentences):#helper function to ensure proper formatting of corpus 
		for sentence in sentences:
			# deacc=True removes punctuations
			yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

	def remove_stopwords(self,texts):#helper function to remove stopwords
		return [[word for word in simple_preprocess(str(doc)) 
			if word not in self.stop_words] for doc in texts]

	def tokenize(self):#function to tokenize data in prep for LDA
		data = self.df['text'].values.tolist()
		data_words = list(self.sent_to_words(data))
		# remove stop words
		data_words = self.remove_stopwords(data_words)
		# print(data_words[:1][0][:300])
		# Create Dictionary
		self.id2word = corpora.Dictionary(data_words)
		# Create Corpus
		texts = data_words
		# Term Document Frequency
		self.corpus = [self.id2word.doc2bow(text) for text in texts]
		# View
		# print(corpus[:1][0][:30])

	def genSimLDA(self,topics=10,vis=True):#Runs the LDA using Gensim and returns as a LDAvis HTML file
		# number of topics
		num_topics = topics
		# Build LDA model
		lda_model = gensim.models.LdaMulticore(corpus=self.corpus,id2word=self.id2word,num_topics=num_topics)
		# Print the Keyword in the 10 topics
		pprint(lda_model.print_topics())
		doc_lda = lda_model[self.corpus]
		if (vis == True):
			# Visualize the topics
			# pyLDAvis.enable_notebook()
			LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(self.name)+"_"+str(num_topics))
			# # this is a bit time consuming - make the if statement True
			# # if you want to execute visualization prep yourself
			if 1 == 1:
				LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, self.corpus, self.id2word)
				with open(LDAvis_data_filepath, 'wb') as f:
					pickle.dump(LDAvis_prepared, f)
			# load the pre-prepared pyLDAvis data from disk
			with open(LDAvis_data_filepath, 'rb') as f:
				LDAvis_prepared = pickle.load(f)
			pyLDAvis.save_html(LDAvis_prepared, './results/ldavis_prepared_'+str(self.name)+"_"+ str(num_topics) +'.html')
			LDAvis_prepared








		
		




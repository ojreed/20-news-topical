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
import numpy as np

"""
Class setup to manage the import, processing, and LDA of text data in a directory 
	uses a 3rd party LDA from gensim 
"""

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

	def print_corpus(self):
		print(self.corpus)

	def get_corpus(self):
		return self.corpus

	def print_id(self):
		print(self.id2word)

	def get_id(self):
		return self.id2word

	def output(self):#returns the df directly 
		return self.df

	def save(self, fileName = None):#helper function for pickle dump
		if fileName is None:
			fileName = self.name
		pickle.dump(self.df, open( "./pkl/"+str(fileName)+"_df.p", "wb" ) )
		pickle.dump(self.corpus, open( "./pkl/"+str(fileName)+"_corpus.p", "wb" ) )
		pickle.dump(self.id2word, open( "./pkl/"+str(fileName)+"_id2word.p", "wb" ) )
		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.df = pickle.load(open( "./pkl/"+str(fileName)+"_df.p", "rb" ) )
		self.corpus = pickle.load( open( "./pkl/"+str(fileName)+"_corpus.p", "rb" ) )
		self.id2word = pickle.load( open( "./pkl/"+str(fileName)+"_id2word.p", "rb" ) )
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

	def LDA(self,topics=10,vis=True):#Runs the LDA using Gensim and returns as a LDAvis HTML file
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

"""
Custom version of the data manager class that implements a custom version of LDA rather than using a gensim version
"""

class custom_manager(data_manager):
	def __init__(self,name):
		data_manager.__init__(self,name)
		self.matrix_final = None# 0 = prob_topics, 1 = word_topic_count, 2 = topic_doc_assign, 3 = doc_topic_count
		self.theta_final = None

	def LDA(self,topics=5,alpha = 0.2, beta = 0.001, num_iter = 100):
	#initilize hyperparamters
		#topics, alpha, beta and number of iterations managed by function defualt vals
		#Vocabulary Size	
		V = len(self.id2word)
		#number of documents
		D = len(self.corpus)
	#practical count matricies
		# Initialize word-topic count matrix (size K x V, K = # topics, V = # vocabulary)
		word_topic_count = np.zeros((topics,V))

		# Initialize topic-document assignment matrix
		topic_doc_assign = [np.zeros(len(doc)) for doc in self.corpus] 

		# Initialize document-topic matrix
		doc_topic_count = np.zeros((D,topics))
	#randomize init word-topic
		for doc_ind in range(D):
			for word_ind in range(len(self.corpus[doc_ind])):
				topic_doc_assign[doc_ind][word_ind] = np.random.choice(topics,1)
				# Record word-topic and word-ID
				word_topic = int(topic_doc_assign[doc_ind][word_ind])
				word_doc_ID = self.corpus[doc_ind][word_ind]
				# Increment word-topic count matrix
				word_topic_count[word_topic, word_doc_ID] += 1 
	#randomize init document-topics
		# Loop over documents (D = numb. docs)
		for doc_ind in range(D):
			
			# Loop over topics (K = numb. topics)
			for topic_ind in range(topics):
				
				# topic-document vector
				topic_doc_vector = topic_doc_assign[doc_ind]
				
				# Update document-topic count
				doc_topic_count[doc_ind][topic_ind] = sum(topic_doc_vector == topic_ind)
	#Run LDA - Main Segement
		#loop num iter
		for it in range(num_iter):
			#loop documents
			for doc_ind in range(D):
				#loop words
				for word_ind in range(len(self.corpus[doc_ind])):
				#setup
					#Initial topic-word assignment
					init_topic = int(topic_doc_assign[doc_ind][word_ind])
					#Initial word ID of word
					wordID = self.corpus[doc_ind][word_ind][0]
					# Before finiding posterior probabilities, remove current word from count matrixes
					doc_topic_count[doc_ind][init_topic] -= 1
					word_topic_count[init_topic, wordID] -= 1
				#Find probability for reassingment based on the big equations
					# Denominator in first term (Numb. of words in doc + numb. topics * alpha)
					denom1 = sum(doc_topic_count[doc_ind]) + topics*alpha
					
					# Denominator in second term (Numb. of words in topic + numb. words in vocab * beta)
					denom2 = np.sum(word_topic_count, axis = 1) + V*beta
					

					# Numerators, number of words assigned to a topic + prior dirichlet param
					numerator1_temp = [doc_topic_count[doc_ind][col] for col in range(topics)] 
					numerator1 = np.array(numerator1_temp) + alpha
					numerator2_temp = [word_topic_count[row, wordID] for row in range(topics)]
					numerator2 = np.array(numerator2_temp) + beta

				# Compute conditional probability of assigning each topic
					# Recall that this is obtained from gibbs sampling
					prob_topics_temp = (numerator1/denom1) * (numerator2/denom2)
					prob_topics = prob_topics_temp/sum(prob_topics_temp)
				
					# Update topic assignment (topic can be drawn with prob. found above)
					update_topic_assign = np.random.choice(topics,1,list(prob_topics))
					topic_doc_assign[doc_ind][word_ind] = update_topic_assign
					
					# Add in current word back into count matrixes
					doc_topic_count[doc_ind][init_topic] += 1
					word_topic_count[init_topic ,wordID] +=1
	#post process
		# Compute posterior mean of document-topic distribution
		theta = (doc_topic_count+alpha)
		theta_row_sum = np.sum(theta, axis = 1)
		theta = theta/theta_row_sum.reshape((D,1))

		# Print document-topic mixture
		print('Subset of document-topic mixture matrix: \n%s' % theta[0:30])
		print("Word Topics")
		print(word_topic_count)
		print("Prob Topics")
		print(prob_topics)
		self.matrix_final = [prob_topics, word_topic_count, topic_doc_assign, doc_topic_count]
	
	def most_frequent_index(self, lst, n):
		n_most = []
		for word_index in range(len(lst)):#an index in the target list
			if (len(n_most) == 0):
					n_most = [word_index]
			for rank_index in range(min((len(n_most),n))):#an index in the n_most ranking list
				if (lst[word_index] > lst[n_most[rank_index]]):#compare target list value w/ current ranks value
					n_most.insert(rank_index,word_index)
					break 
		return n_most[:n]


	def visualize(self):
		print("Most Common Words by Topics")
		word_depth = 10 #number of most frequent words shown
		for topic in range(len(self.matrix_final[0])): #for each topic
			print("Topics #" + str(topic) + ":")
			n_most = self.most_frequent_index(self.matrix_final[1][topic],word_depth)
			for word in range(word_depth):
				print("	" + str(word+1) + ") " + str(self.id2word[n_most[word]]) + " at " + str(round(self.matrix_final[1][topic][n_most[word]]/sum(self.matrix_final[1][topic])*100,4)) + "%")


	def save(self, fileName = None):#helper function for pickle dump
		if fileName is None:
			fileName = self.name
		pickle.dump(self.df, open( "./pkl/"+ str(fileName)+"_df.p", "wb" ) )
		pickle.dump(self.corpus, open( "./pkl/"+ str(fileName)+"_corpus.p", "wb" ) )
		pickle.dump(self.id2word, open( "./pkl/"+ str(fileName)+"_id2word.p", "wb" ) )
		pickle.dump(self.theta_final, open( "./pkl/"+ str(fileName)+"_theta.p", "wb" ) )
		pickle.dump(self.matrix_final, open( "./pkl/"+ str(fileName)+"_matrix.p", "wb" ) )
		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.df = pickle.load(open( "./pkl/"+str(fileName)+"_df.p", "rb" ) )
		self.corpus = pickle.load( open( "./pkl/"+str(fileName)+"_corpus.p", "rb" ) )
		self.id2word = pickle.load( open( "./pkl/"+str(fileName)+"_id2word.p", "rb" ) )
		self.theta_final = pickle.load( open( "./pkl/"+str(fileName)+"_theta.p", "rb" ) )
		self.matrix_final = pickle.load( open( "./pkl/"+str(fileName)+"_matrix.p", "rb" ) )
		print(str(fileName) + " loaded")








		
		




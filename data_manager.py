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
		self.ID = 0
	
	def add(self,group,ID,text):#adds a new entry to the data frame
		newData = {"group":group,"id":ID,"text":text}
		self.df = self.df.append(newData,ignore_index = True)

	def reset(self):
		self.df = pd.DataFrame(columns = ["group","id", "text"]) 

	def parseGroup(self,group,toy_limit=None):#parses the read in of a directory full of text data
		counter = 0
		self.ID +=1
		for doc in os.listdir(group):
			self.add(group,self.ID,self.getText(doc,group))
			counter+=1
			if (counter == toy_limit):
				return

	def getText(self,doc,group):#helper function to get the text from a target file and strips bad chars
		docText = ""
		f = open(os.path.join(group, doc), "r")
		f.seek(0)
		for line in f: # goes through all lines
			for word in line.lower().split(): # goes throuugh all words per line
				# pureWord = word.strip("[]}{,.\\/!@#$%^&*()<>#;?''" '"')
				pureWord = word
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

Source: https://github.com/hammadshaikhha/Data-Science-and-Machine-Learning-from-Scratch/blob/master/Latent%20Dirichlet%20Allocation/Latent%20Dirichlet%20Allocation.ipynb

"""

class custom_manager(data_manager):
	def __init__(self,name):
		data_manager.__init__(self,name)
		self.matrix_final = None# 0 = prob_topics, 1 = word_topic_count, 2 = topic_doc_assign, 3 = doc_topic_count
		self.theta_final = None
		self.num_topics = None 
		self.phi = None 

	def LDA2(self,topics=5,alpha = 0.2, beta = 0.001, num_iter = 100, mode = 0):
		# Initialize hyperparameters in LDA
		vocab_total = self.id2word
		text_ID = self.corpus

		# Dirichlet parameters
		# Alpha is the parameter for the prior topic distribution within documents
		alpha = alpha

		# Beta is the parameter for the prior topic distribution within documents
		beta = beta

		# Text corpus itterations
		corpus_itter = num_iter

		# Number of topics
		K = topics
		
		# Vocabulary size
		V = len(vocab_total)

		# Number of Documents
		D = len(text_ID)

		# For practical implementation, we will generate the following three count matrices:
		# 1) Word-Topic count matrix, 2) Topic-Document assignment matrix, 3) Document-Topic count matrix

		# Initialize word-topic count matrix (size K x V, K = # topics, V = # vocabulary)
		word_topic_count = np.zeros((K,V))

		# Initialize topic-document assignment matrix
		topic_doc_assign = [np.zeros(len(sublist)) for sublist in text_ID] 

		# Initialize document-topic matrix
		doc_topic_count = np.zeros((D,K))
		# Generate word-topic count matrix with randomly assigned topics

		# Loop over documents
		for doc in range(D):
			
			# Loop over words in given document
			for word in range(len(text_ID[doc])):

				# Step 1: Randomly assign topics to each word in document
				# Note random.choice generates number {0,...,K-1}
				topic_doc_assign[doc][word] = np.random.choice(K,1)

				# Record word-topic and word-ID
				word_topic = int(topic_doc_assign[doc][word])
				word_doc_ID = text_ID[doc][word][0]
				
				# Increment word-topic count matrix
				word_topic_count[word_topic,word_doc_ID] += 1
		
		# Print word-topic matrix
		print('Word-topic count matrix with random topic assignment: \n%s' % word_topic_count)

		# Generate document-topic count matrix with randomly assigned topics

		# Loop over documents (D = numb. docs)
		for doc in range(D):
			
			# Loop over topics (K = numb. topics)
			for topic in range(K):
				
				# topic-document vector
				topic_doc_vector = topic_doc_assign[doc]
				
				# Update document-topic count
				doc_topic_count[doc][topic] = sum(topic_doc_vector == topic)

		# Print document-topic matrix
		print('Subset of document-topic count matrix with random topic assignment: \n%s' % doc_topic_count[0:5])

		# Main part of LDA algorithm (takes a few minutes to run)
		# Run through text corpus multiple times
		for itter in range(corpus_itter):
			
			# Loop over all documents
			for doc in range(D):
				
				# Loop over words in given document
				for word in range(len(text_ID[doc])):
					
					# Initial topic-word assignment
					init_topic_assign = int(topic_doc_assign[doc][word])
					
					# Initial word ID of word 
					word_id = text_ID[doc][word][0]
					
					# Before finiding posterior probabilities, remove current word from count matrixes
					doc_topic_count[doc][init_topic_assign] -= 1
					word_topic_count[init_topic_assign,word_id] -=1
					
					# Find probability used for reassigning topics to words within documents
					
					# Denominator in first term (Numb. of words in doc + numb. topics * alpha)
					denom1 = sum(doc_topic_count[doc]) + K*alpha
					
					# Denominator in second term (Numb. of words in topic + numb. words in vocab * beta)
					denom2 = np.sum(word_topic_count, axis = 1) + V*beta
					
					# Numerators, number of words assigned to a topic + prior dirichlet param
					numerator1 = [doc_topic_count[doc][col] for col in range(K)] 
					numerator1 = np.array(numerator1) + alpha
					numerator2 = [word_topic_count[row,word_id] for row in range(K)]
					numerator2 = np.array(numerator2) + beta
					
					# Compute conditional probability of assigning each topic
					# Recall that this is obtained from gibbs sampling
					prob_topics = (numerator1/denom1)*(numerator2/denom2)
					prob_topics = prob_topics/sum(prob_topics)
					  

					for index in range(len(prob_topics)):
						if prob_topics[index] > 1 or prob_topics[index] < 0:
							maxInd = list(prob_topics).index(max(prob_topics))
							prob_topics = prob_topics*0
							prob_topics[maxInd]=1
						

					# print(prob_topics) 
								  
					# Update topic assignment (topic can be drawn with prob. found above)
					update_topic_assign = np.random.choice(K,1,p=list(prob_topics))
					topic_doc_assign[doc][word] = update_topic_assign
					
					# Add in current word back into count matrixes
					doc_topic_count[doc][init_topic_assign] += 1
					word_topic_count[init_topic_assign,word_id] +=1
		theta = (doc_topic_count+alpha)
		theta_row_sum = np.sum(theta, axis = 1)
		theta = theta/theta_row_sum.reshape((D,1))

		# Compute posterior mean of word-topic distribution within documents
		phi = (word_topic_count + beta)
		phi_row_sum = np.sum(phi, axis = 1)
		self.phi = phi/phi_row_sum.reshape((K,1))

		# Print document-topic mixture
		if (mode == 1):
			print('Subset of document-topic mixture matrix: \n%s' % theta[0:30])
			print("Word Topics")
			print(word_topic_count)
			print("Prob Topics")
			print(prob_topics)
		self.theta_final = theta
		self.matrix_final = [prob_topics, word_topic_count, topic_doc_assign, doc_topic_count]
		self.num_topics = K




	def LDA(self,topics=5,alpha = 0.2, beta = 0.001, num_iter = 100, mode = 0):
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
					numerator1 = [doc_topic_count[doc_ind][col] for col in range(topics)] 
					numerator1 = np.array(numerator1) + alpha
					numerator2 = [word_topic_count[row][wordID] for row in range(topics)]
					numerator2 = np.array(numerator2) + beta
					
					# Compute conditional probability of assigning each topic
					# Recall that this is obtained from gibbs sampling
					prob_topics = (numerator1/denom1)*(numerator2/denom2)
					prob_topics = prob_topics/sum(prob_topics)
					
					# for index in range(len(prob_topics)):
					# 	if prob_topics[index] < 0:
					# 		prob_topics[index] = 0
					# 	if prob_topics[index] > 1:
					# 		prob_topics[index] = 1


					# Update topic assignment (topic can be drawn with prob. found above)
					# print(list(prob_topics))
					update_topic_assign = np.random.choice(topics,1,p=list(prob_topics))
					topic_doc_assign[doc_ind][word_ind] = update_topic_assign
					
					# Add in current word back into count matrixes
					doc_topic_count[doc_ind][init_topic] += 1
					word_topic_count[init_topic ,wordID] += 1
	#post process
		# Compute posterior mean of document-topic distribution
		theta = (doc_topic_count+alpha)
		theta_row_sum = np.sum(theta, axis = 1)
		theta = theta/theta_row_sum.reshape((D,1))

		# Compute posterior mean of word-topic distribution within documents
		phi = (word_topic_count + beta)
		phi_row_sum = np.sum(phi, axis = 1)
		self.phi = phi/phi_row_sum.reshape((topics,1))

		# Print document-topic mixture
		if (mode == 1):
			print('Subset of document-topic mixture matrix: \n%s' % theta[0:30])
			print("Word Topics")
			print(word_topic_count)
			print("Prob Topics")
			print(prob_topics)
		self.theta_final = theta
		self.matrix_final = [prob_topics, word_topic_count, topic_doc_assign, doc_topic_count]
		self.num_topics = topics

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

	def n_most_frequent_proportional(self,lst,topic,n):
		n_most = []
		n_most_scores = []
		for word_index in range(len(lst[topic])):
			#count of word in topic - sum of count of word in all topics
			# lst[topic][word_index]/sum([lst[other_topics][word_index] for other_topics in range(self.num_topics)])
			word_score = (lst[topic][word_index]/sum(lst[topic])*100)/(sum([lst[other_topics][word_index] for other_topics in range(self.num_topics)])/len(self.id2word)) #word_in/topic : word_out/all
			# word_score = lst[topic][word_index] - sum([lst[other_topics][word_index] for other_topics in range(self.num_topics)])
			if (len(n_most) == 0):
					n_most = [word_index]
					n_most_scores = [word_score]
			for rank_index in range(min((len(n_most),n))):#an index in the n_most ranking list
				if (word_score > n_most_scores[rank_index]):#compare target list value w/ current ranks value
					n_most.insert(rank_index,word_index)
					n_most_scores.insert(rank_index,word_score)
					break 
		return n_most[:n]


	def visualize(self,depth=20,mode=0):
		print("Most Common Words by Topics")
		word_depth = depth #number of most frequent words shown
		for topic in range(len(self.matrix_final[0])): #for each topic
			print("Topic #" + str(topic+1) + ":")
			if mode == 0:
				n_most = self.most_frequent_index(self.matrix_final[1][topic],word_depth)
			if mode == 1:
				n_most = self.n_most_frequent_proportional(self.matrix_final[1],topic,word_depth)
			for word in range(word_depth):
				# print("	" + str(word+1) + ") " + str(self.id2word[n_most[word]]) + " at " + str(round(self.matrix_final[1][topic][n_most[word]]/sum(self.matrix_final[1][topic])*100,4)) + "%")
				print("	" + str(word+1) + ") " + str(self.id2word[n_most[word]]) + " at " + str(round(self.phi[topic][n_most[word]]*100,4)) + "%")
		print("\n")
		print('Subset of document-topic mixture matrix: \n%s' % self.theta_final)#[0:30]) 

	def save(self, fileName = None):#helper function for pickle dump
		if fileName is None:
			fileName = self.name
		pickle.dump(self.df, open( "./pkl/"+ str(fileName)+"_df.p", "wb" ) )
		pickle.dump(self.corpus, open( "./pkl/"+ str(fileName)+"_corpus.p", "wb" ) )
		pickle.dump(self.id2word, open( "./pkl/"+ str(fileName)+"_id2word.p", "wb" ) )
		pickle.dump(self.theta_final, open( "./pkl/"+ str(fileName)+"_theta.p", "wb" ) )
		pickle.dump(self.matrix_final, open( "./pkl/"+ str(fileName)+"_matrix.p", "wb" ) )
		pickle.dump(self.phi, open( "./pkl/"+ str(fileName)+"_phi.p", "wb" ) )
		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.df = pickle.load(open( "./pkl/"+str(fileName)+"_df.p", "rb" ) )
		self.corpus = pickle.load( open( "./pkl/"+str(fileName)+"_corpus.p", "rb" ) )
		self.id2word = pickle.load( open( "./pkl/"+str(fileName)+"_id2word.p", "rb" ) )
		self.theta_final = pickle.load( open( "./pkl/"+str(fileName)+"_theta.p", "rb" ) )
		self.matrix_final = pickle.load( open( "./pkl/"+str(fileName)+"_matrix.p", "rb" ) )
		self.phi = pickle.load( open( "./pkl/"+str(fileName)+"_phi.p", "rb" ) )
		self.num_topics = self.matrix_final[0].shape[0]
		print(str(fileName) + " loaded")








		
		




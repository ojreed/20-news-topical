import numpy as np
import time
import codecs
import jieba
import re
import os
import pickle

"""
https://github.com/disadone/LDA-Gibbs-Sampling
"""


class LDAManager():

	def __init__(self,name,reset_on_init=False):
		self.name = name
		self.docs = None
		self.id2word = None
		self.N = None
		self.M = None
		self.word2id = None
		self.ndz= None
		self.nzw = None
		self.nz = None
		if reset_on_init:
			self.main(num_iter=100,toy_size=10)
			self.save()


	def save(self, fileName = None):#helper function for pickle dump
		if fileName is None:
			fileName = self.name
		pickle.dump(self.docs , open( "./pkl/"+str(fileName)+"_docs.p", "wb" ) )
		pickle.dump(self.id2word , open( "./pkl/"+str(fileName)+"_id2word.p", "wb" ) )
		pickle.dump(self.N , open( "./pkl/"+str(fileName)+"_N.p", "wb" ) )
		pickle.dump(self.M , open( "./pkl/"+str(fileName)+"_M.p", "wb" ) )
		pickle.dump(self.word2id , open( "./pkl/"+str(fileName)+"_word2id.p", "wb" ) )
		pickle.dump(self.ndz , open( "./pkl/"+str(fileName)+"_ndz.p", "wb" ) )
		pickle.dump(self.nzw , open( "./pkl/"+str(fileName)+"_nzw.p", "wb" ) )
		pickle.dump(self.nz , open( "./pkl/"+str(fileName)+"_nz.p", "wb" ) )

		params = [self.alpha, self.beta, self.iterationNum, self.Z, self.K]
		pickle.dump(params , open( "./pkl/"+str(fileName)+"_params.p", "wb" ) )

		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.docs     = pickle.load(open( "./pkl/"+str(fileName)+"_docs.p", "rb" ) )
		self.id2word  = pickle.load(open( "./pkl/"+str(fileName)+"_id2word.p", "rb" ) )
		self.N        = pickle.load(open( "./pkl/"+str(fileName)+"_N.p", "rb" ) )
		self.M        = pickle.load(open( "./pkl/"+str(fileName)+"_M.p", "rb" ) )
		self.word2id  = pickle.load(open( "./pkl/"+str(fileName)+"_word2id.p", "rb" ) )
		self.ndz      = pickle.load(open( "./pkl/"+str(fileName)+"_ndz.p", "rb" ) )
		self.nzw      = pickle.load(open( "./pkl/"+str(fileName)+"_nzw.p", "rb" ) )
		self.nz       = pickle.load(open( "./pkl/"+str(fileName)+"_nz.p", "rb" ) )
		params        = pickle.load(open( "./pkl/"+str(fileName)+"_params.p", "rb" ) )
		self.alpha        = params[0]
		self.beta         = params[1]
		self.iterationNum = params[2]
		self.Z            = params[3]
		self.K            = params[4]

		print(str(fileName) + " loaded")


	def main(self,alpha=7,beta=0.1,num_iter=300,K=10,toy_size=None):
		self.alpha = alpha
		self.beta = beta
		self.iterationNum = num_iter
		self.Z = []
		self.K = K
		group_lst = ["./20news-bydate/20news-bydate-train/alt.atheism",
					"./20news-bydate/20news-bydate-train/misc.forsale",
					"./20news-bydate/20news-bydate-train/rec.sport.hockey",
					"./20news-bydate/20news-bydate-train/talk.politics.guns",
					"./20news-bydate/20news-bydate-train/comp.sys.mac.hardware",
					"./20news-bydate/20news-bydate-train/sci.electronics"]
		self.docs, self.word2id, self.id2word = self.preprocessing(group_lst,toy_size)
		self.N = len(self.docs)
		self.M = len(self.word2id)
		self.ndz = np.zeros([self.N, self.K]) + self.alpha #Doc x Topic
		self.nzw = np.zeros([self.K, self.M]) + self.beta  #Topic x Word
		self.nz = np.zeros([self.K]) + self.M * self.beta  #Topic
		self.randomInitialize()
		#runs gibbs for each iteration
		for i in range(0, self.iterationNum):
			self.gibbsSampling()
			print(time.strftime('%X'), "Iteration: ", i, " Completed", " Perplexity: ", self.perplexity())
		#visualizes the results
		topicwords = []
		maxTopicWordsNum = 15
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(10, len(topicword))])
		for topic in range(self.K):
			print("Topic #" + str(topic) + " contains:")
			print("	" + str(topicwords[topic]))

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

	def parseGroup(self,group,toy_limit=None):#parses the read in of a directory full of text data
		documents = []
		counter = 0
		for doc in os.listdir(group):
			documents.append(self.getText(doc,group))
			counter+=1
			if (counter == toy_limit):
				break
		return documents

	def preprocessing(self,group_lst,toy_size=None):
		
		from stop_words import get_stop_words

		# Create English stop words list
		stopwords = [str(word) for word in get_stop_words('english')]
		stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'com', 'org'])
		
		
		documents = []
		for name in group_lst:
			documents += self.parseGroup(name,toy_size)
		word2id = {}
		id2word = {}
		self.docs = []
		currentDocument = []
		currentWordId = 0
		
		for document in documents:
			segList = jieba.cut(document)
			for word in segList: 
				word = word.lower().strip()
				if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
					if word in word2id:
						currentDocument.append(word2id[word])
					else:
						currentDocument.append(currentWordId)
						word2id[word] = currentWordId
						id2word[currentWordId] = word
						currentWordId += 1
			self.docs.append(currentDocument);
			currentDocument = []
		return self.docs, word2id, id2word
		
	def randomInitialize(self):
		for d, doc in enumerate(self.docs):
			zCurrentDoc = []
			for w in doc:
				pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
				z = np.random.multinomial(1, pz / pz.sum()).argmax()
				zCurrentDoc.append(z)
				self.ndz[d, z] += 1
				self.nzw[z, w] += 1
				self.nz[z] += 1
			self.Z.append(zCurrentDoc)

	def gibbsSampling(self):
		for d, doc in enumerate(self.docs):
			for index, w in enumerate(doc):
				#GET CURRENT
				z = self.Z[d][index]
				#REMOVE CURRENT WORD FROM THE INFO
				self.ndz[d, z] -= 1
				self.nzw[z, w] -= 1
				self.nz[z] -= 1
				#GIBBS MATH
				pz = np.divide(np.multiply(self.ndz[d, :], self.nzw[:, w]), self.nz)
				#CHOOSE NEW W/ NEW PROB
				z = np.random.multinomial(1, pz / pz.sum()).argmax()
				#CHANGE CURRENT
				self.Z[d][index] = z 
				#ADD BACK CURRENT WORD FROM THE INFO
				self.ndz[d, z] += 1
				self.nzw[z, w] += 1
				self.nz[z] += 1

	#perplexity as score for understanding progress
	def perplexity(self):
		nd = np.sum(self.ndz, 1)
		n = 0
		ll = 0.0
		for d, doc in enumerate(self.docs):
			for w in doc:
				ll = ll + np.log(((self.nzw[:, w] / self.nz) * (self.ndz[d, :] / nd[d])).sum())
				n = n + 1
		return np.exp(ll/(-n))

	def visualize_words(self):
		#print most common words
		topicwords = []
		maxTopicWordsNum = 15
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(10, len(topicword))])
		for topic in range(self.K):
			print("Topic #" + str(topic+1) + " contains:")
			print("	" + str(topicwords[topic]))
	def visualize_topics(self):
		#print Doc x Topic Distributions
		for num, doc in enumerate(self.ndz):
			print("Document #", num+1, ":")
			print(np.array([np.round(topic/np.sum(doc)*100,2) for topic in doc]))
			# for top_num, topic in enumerate(doc):
			# 	print("	Topic #",top_num," - ",np.round(topic/np.sum(doc)*100,2),"%")

		# print("Doc x Topic:\n",self.ndz)
		# print("Topic x Word:\n",self.nzw)
		# print("Topic:\n",self.nz)








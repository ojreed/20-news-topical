import numpy as np
import time
import codecs
import jieba
import re
import os
import pickle
import pandas as pd

"""
Implementation based on: https://github.com/disadone/LDA-Gibbs-Sampling
"""


class LDAManager():

	def __init__(self,name,saftey=True):
		self.name = name
		self.docs = None
		self.id2word = None
		self.N = None
		self.M = None
		self.word2id = None
		self.ndz = None
		self.nzw = None
		self.nz = None
		self.topicNames = []
		self.testSet = None
		self.predictions = None
		self.dataGroup = []
		self.shortDataGroup = []
		if saftey:#by default tries to load a previous version to prevent accidental overwrites
			self.load()

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
		pickle.dump(self.topicNames , open( "./pkl/"+str(fileName)+"_topicNames.p", "wb" ) )
		pickle.dump(self.testSet , open( "./pkl/"+str(fileName)+"_testSet.p", "wb" ) )
		pickle.dump(self.predictions, open( "./pkl/"+str(fileName)+"_predictions.p", "wb" ) )
		pickle.dump(self.dataGroup, open( "./pkl/"+str(fileName)+"_dataGroup.p", "wb" ) )
		pickle.dump(self.shortDataGroup, open( "./pkl/"+str(fileName)+"_shortDataGroup.p", "wb" ) )

		params = [self.alpha, self.beta, self.iterationNum, self.Z, self.K]
		pickle.dump(params , open( "./pkl/"+str(fileName)+"_params.p", "wb" ) )

		print(str(fileName) + " saved")

	def load(self, fileName = None):#helper function for pickle load
		if fileName is None:
			fileName = self.name
		self.docs           = pickle.load(open( "./pkl/"+str(fileName)+"_docs.p", "rb" ) )
		self.id2word        = pickle.load(open( "./pkl/"+str(fileName)+"_id2word.p", "rb" ) )
		self.N              = pickle.load(open( "./pkl/"+str(fileName)+"_N.p", "rb" ) )
		self.M              = pickle.load(open( "./pkl/"+str(fileName)+"_M.p", "rb" ) )
		self.word2id        = pickle.load(open( "./pkl/"+str(fileName)+"_word2id.p", "rb" ) )
		self.ndz            = pickle.load(open( "./pkl/"+str(fileName)+"_ndz.p", "rb" ) )
		self.nzw            = pickle.load(open( "./pkl/"+str(fileName)+"_nzw.p", "rb" ) )
		self.nz             = pickle.load(open( "./pkl/"+str(fileName)+"_nz.p", "rb" ) )
		self.topicNames     = pickle.load(open( "./pkl/"+str(fileName)+"_topicNames.p", "rb" ) )
		self.testSet        = pickle.load(open( "./pkl/"+str(fileName)+"_testSet.p", "rb" ) )
		self.predictions    = pickle.load(open( "./pkl/"+str(fileName)+"_predictions.p", "rb" ) )
		self.dataGroup      = pickle.load(open( "./pkl/"+str(fileName)+"_dataGroup.p", "rb" ) )
		self.shortDataGroup = pickle.load(open( "./pkl/"+str(fileName)+"_shortDataGroup.p", "rb" ) )
		params              = pickle.load(open( "./pkl/"+str(fileName)+"_params.p", "rb" ) )
		self.alpha          = params[0]
		self.beta           = params[1]
		self.iterationNum   = params[2]
		self.Z              = params[3]
		self.K              = params[4]

		print(str(fileName) + " loaded")


	def main(self,alpha=7,beta=0.1,num_iter=300,K=10,toy_size=None,skew=False):
		#INIT PARAMS
		self.alpha = alpha
		self.beta = beta
		self.iterationNum = num_iter
		self.Z = []
		self.K = K
		#LIST OF INPUT TEXT
		group_lst = self.dataGroup
		self.docs, self.word2id, self.id2word = self.preprocessing(group_lst,toy_size,skew)
		self.N = len(self.docs) #Number of Documents
		self.M = len(self.word2id) #Number of Words
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
			topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
		for topic in range(self.K):
			print("Topic #" + str(topic) + " contains:")
			print("	" + str(topicwords[topic]))

	def defaultSetDataGroup(self): #used to set topics to the main set I have been using for liu
		self.dataGroup = ["./20news-bydate/20news-bydate-train/alt.atheism",
					"./20news-bydate/20news-bydate-train/misc.forsale",
					"./20news-bydate/20news-bydate-train/rec.sport.hockey",
					"./20news-bydate/20news-bydate-train/talk.politics.guns",
					"./20news-bydate/20news-bydate-train/comp.sys.mac.hardware",
					"./20news-bydate/20news-bydate-train/sci.electronics"]
		self.shortDataGroup = ["alt.atheism","misc.forsale","rec.sport.hockey","talk.politics.guns","comp.sys.mac.hardware","sci.electronics"]

	def setDataGroup(self,datagroup): #takes an input of a list of file paths
		self.dataGroup = datagroup
		self.shortDataGroup = [shorty.split("/")[-1] for shorty in datagroup]

	def appendDataGroup(self,newGroup): #takes an input of a single file path
		self.dataGroup.append(newGroup)
		self.shortDataGroup.append(newGroup.split("/")[-1])

	def initTestSet(self,toy_size=None): #initialzies a test set for our model
		group_lst = self.dataGroup
		from stop_words import get_stop_words

		# Create English stop words list
		stopwords = [str(word) for word in get_stop_words('english')]
		stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'com', 'org'])
		
		#read in each group of text
		documents = []
		for name in group_lst:
			documents += self.parseGroupTest(name,toy_size)
		
		#init containers 
		docs = []
		currentDocument = []

		#tokenizes the data
		for document in documents:
			segList = jieba.cut(document[0])
			for word in segList: 
				#simplify to just words
				word = word.lower().strip()
				word = re.sub(r'[^a-zA-Z]','', word)
				if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
					if word in self.word2id:
						currentDocument.append(self.word2id[word])
			docs.append((currentDocument,document[1]));
			currentDocument = []
		self.testSet = docs

	def labelTestCorpus(self,treshhold=1):
		docTopicPredictions = []
		correct = 0
		total = 0
		for doc in self.testSet: #loop through all docs
			docTopicCount = [0 for x in range(self.K)]
			for word in doc[0]: #check each word in the doc
				for topic in range(self.K): #see if it is popular in each of the topics
					if self.nzw[topic,word] >= treshhold:
						docTopicCount[topic] +=1 #if it is above the hyper param threshold then we increment topic distribution
			docTopicPredictions.append((docTopicCount,doc[1]))
		for num, doc in enumerate(docTopicPredictions):
			print("Document",str(num+1)+":\n   Actual Topic:",doc[1].split(".")[-1],"\n   Predicted Topic:",self.topicNames[doc[0].index(max(doc[0]))])
			total +=1 
			if (doc[1].split(".")[-1] == self.topicNames[doc[0].index(max(doc[0]))]):
				correct+=1
			#hard coded topic name fixes
			elif ((doc[1].split(".")[-1] == "hardware" and self.topicNames[doc[0].index(max(doc[0]))]) == "apple hardware"):
				correct+=1
			elif ((doc[1].split(".")[-1] == "forsale" and self.topicNames[doc[0].index(max(doc[0]))]) == "for sale"):
				correct+=1
		self.predictions = [docTopicPredictions,correct/total]

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

	def parseGroupTest(self,group,toy_limit=None):#parses the read in of a directory w/ labeled actual topics
		documents = []
		counter = 0
		for doc in os.listdir(group):
			documents.append((self.getText(doc,group),group))
			counter+=1
			if (counter == toy_limit):
				break
		return documents

	def preprocessing(self,group_lst,toy_size=None,skew=False):
		
		from stop_words import get_stop_words

		# Create English stop words list
		stopwords = [str(word) for word in get_stop_words('english')]
		stopwords.extend(['from', 'subject', 're', 'edu', 'use', 'com', 'org'])
		
		#read in each group of text
		documents = []
		if skew == True:
			skewCount=.1
			for name in group_lst:
				skewCount*=2
				documents += self.parseGroup(name,toy_size*skewCount)
		else:
			for name in group_lst:
				documents += self.parseGroup(name,toy_size)

		#init containers 
		word2id = {}
		id2word = {}
		self.docs = []
		currentDocument = []
		currentWordId = 0
		
		#tokenizes the data
		for document in documents:
			segList = jieba.cut(document)
			for word in segList: 
				#simplify to just words
				word = word.lower().strip()
				word = re.sub(r'[^a-zA-Z]','', word)
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
			
	def randomInitialize(self): #creates an initial state for each of the matricies we are working with
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

	def perplexity(self): #perplexity as score for understanding progress --> the lower the better
		nd = np.sum(self.ndz, 1)
		n = 0
		ll = 0.0
		for d, doc in enumerate(self.docs):
			for w in doc:
				ll = ll + np.log(((self.nzw[:, w] / self.nz) * (self.ndz[d, :] / nd[d])).sum())
				n = n + 1
		return np.exp(ll/(-n))

	def visualize_words(self,maxTopicWordsNum = 15):
		#print most common words
		topicwords = []
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
		for topic in range(self.K):
			if len(self.topicNames) == self.K:
				print("Topic '" + self.topicNames[topic] + "' contains:")
				print("	" + str(topicwords[topic]))
				print("\n")
			else:
				print("Topic #" + str(topic+1) + " contains:")
				print("	" + str(topicwords[topic]))
				print("\n")

	def visualize_words_print_out(self,maxTopicWordsNum = 15):
		#print most common words
		ss = ""
		topicwords = []
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
		for topic in range(self.K):
			if len(self.topicNames) == self.K:
				ss += str("Topic '" + self.topicNames[topic] + "' contains:")
				ss += str("	" + str(topicwords[topic]))
				ss += str("\n\n")
			else:
				ss += str("Topic #" + str(topic+1) + " contains:")
				ss += str("	" + str(topicwords[topic]))
				ss += str("\n\n")
		return ss


	def visualize_topics(self):
		#print Doc x Topic Distributions
		for num, doc in enumerate(self.ndz):
			print("Document #", num+1, ":")
			doc_topics = np.array([np.round(topic/np.sum(doc)*100,2) for topic in doc])
			print(doc_topics)
			if (len(self.topicNames)==self.K):
				print("	Most Common Topic: ",self.topicNames[np.argmax(doc_topics)])
			else:
				print("	Most Common Topic: #",np.argmax(doc_topics))
		print("\n")

	def name_topics(self):
		self.topicNames = []
		topicwords = []
		maxTopicWordsNum = 50
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
		for topic in range(self.K):
			print("Generate a name for:")
			print(topicwords[topic])
			self.topicNames.append(input())

	def produce_csv(self):
		self.train_document_topics_csv() #train: doc | topic | distributions
		self.document_topics_csv() #test: doc | topic | distributions
		self.topic_words_csv() #topic: words
		self.real_pred_topics_csv() #real to pred stats

	def train_document_topics_csv(self):
		fileName = self.name 
		col_names = ["document","actual"]
		col_names += self.topicNames
		df = pd.DataFrame(columns = col_names) 
		for num, doc in enumerate(self.ndz):
			doc_topics =[np.round(topic/np.sum(doc)*100,2) for topic in doc]
			newRow = [num, self.topicNames[np.argmax(doc_topics)]]
			newRow += doc_topics
			df.loc[len(df.index)] = newRow
		print(df)
		df.to_csv('data/train_documents'+str(fileName)+'.csv', index=False)


	def document_topics_df(self): 
		col_names = ["document","actual","labeled"]
		col_names += self.topicNames
		df = pd.DataFrame(columns = col_names) 
		for num, doc in enumerate(self.predictions[0]):
			newRow = [num,doc[1].split("/")[-1], self.topicNames[doc[0].index(max(doc[0]))]]
			newRow += doc[0]
			df.loc[len(df.index)] = newRow
		return df

	def document_topics_csv(self):
		fileName = self.name 
		df = self.document_topics_df()
		print(df)
		df.to_csv('data/labeled_documents'+str(fileName)+'.csv', index=False)

	def topic_words_csv(self):
		fileName = self.name 
		maxTopicWordsNum = 50
		topicwords = []
		for z in range(0, self.K):
			ids = self.nzw[z, :].argsort()
			topicword = []
			for j in ids:
				topicword.insert(0, self.id2word[j])
			topicwords.append(topicword[0 : min(maxTopicWordsNum, len(topicword))])
		df = pd.DataFrame(topicwords,self.topicNames)
		print(df)
		df.to_csv('data/topics'+str(fileName)+'.csv')

	def real_pred_topics_csv(self):
		#init dataframes
		fileName = self.name 
		col_names = ["actual"]
		col_names += self.topicNames
		col_names += ["total words", "total docs","accuracy"]
		doc_topics = self.document_topics_df()
		df = pd.DataFrame(columns = col_names)
		group_lst = self.shortDataGroup
		#calculate accuracy
		correct_pred = {real_topic:0 for real_topic in group_lst}
		total_pred = {real_topic:0 for real_topic in group_lst}
		for num, doc in enumerate(self.predictions[0]):
			total_pred[doc[1].split("/")[-1]]+=1
			if (doc[1].split(".")[-1] == self.topicNames[doc[0].index(max(doc[0]))]):
				correct_pred[doc[1].split("/")[-1]]+=1
			#hard coded topic name fixes
			elif ((doc[1].split(".")[-1] == "hardware" and self.topicNames[doc[0].index(max(doc[0]))]) == "apple hardware"):
				correct_pred[doc[1].split("/")[-1]]+=1
			elif ((doc[1].split(".")[-1] == "forsale" and self.topicNames[doc[0].index(max(doc[0]))]) == "for sale"):
				correct_pred[doc[1].split("/")[-1]]+=1
		#build dataframe
		for real_topic in group_lst:
			row = {}
			row["actual"] = real_topic
			total = 0
			for pred_topic in self.topicNames:
				row[pred_topic] = sum(doc_topics[pred_topic].where(doc_topics["actual"] == real_topic, 0))
				total += row[pred_topic]
			row["total words"] = total
			row["total docs"] = sum(doc_topics["actual"] == real_topic)
			row["accuracy"] = correct_pred[real_topic]/total_pred[real_topic]
			# print(df)
			# print(row)
			df = df.append(row,ignore_index=True)
		print(df)
		df.to_csv('data/real_pred_topics'+str(fileName)+'.csv')
		











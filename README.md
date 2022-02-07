# 20-news-topical
As of 2/6/2022:
The code in this repository is focused mostly on the data manager class within data_manager.py. This class is used to create a Pandas dataframe of the training data in a target directory. This process is easily managed by the member functions of the data manager class.
The Topical Modeling is currently running on genSimLDA and uses pyLDAvis to display this data in a meaningful way. 
The main goals now are to implement an easy UI for labeling topics, refine the list of stopwords not included in topic modeling, and implement a cusom versio of the LDA method. 

The Workbench file serves as flexible mainfile to run LDA experiments. 

Exploratory displays word frequency data as a word cloud to provide qualitative understanding of the most frequent words in a topic group.

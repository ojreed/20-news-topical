# 20-news-topical
As of 3/30/2022:
The code in this repository is focused mostly on the LDA manager class within data_manager_2.py. This class is able to read in text from a list of directories, tokenize this data, and then run an LDA model w/ Gibbs sampleing on the data. It is then capable of saving this data and results for easy visualization, manipulation, and usage later on.

In data_manager.py there is an original implemented method of topical modeling currently running on genSimLDA and uses pyLDAvis to display this data in a meaningful way. The custom version in data manager_2.py is designed for increased flexibility and transparency. 

There is now an implemented text label process that is able to predict topic distributions for inputed text as well as datasets of labeled test text.

The Workbench file serves as flexible mainfile to run LDA experiments. 

Exploratory.py file displays word frequency data as a word cloud to provide qualitative understanding of the most frequent words in a topic group.

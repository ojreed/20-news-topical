import data_manager_2 as dm

# LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization")
LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization_2")
#MAKE NEW
# LDA.main(alpha=10,beta=0.05,num_iter=500,K=8,toy_size=350)
# LDA.save()
#LOAD AND VISUALIZE
LDA.load()
# LDA.visualize_topics()
# LDA.visualize_words(20)
#LABEL
# LDA.name_topics()
# LDA.save()
#TEST SET
LDA.initTestSet(50)
# print(LDA.testSet)
LDA.labelTestCorpus()
LDA.save()


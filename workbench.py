import data_manager_2 as dm

# LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization")
# LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization_2")
# LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization_2_Skew")
LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization_3_Skew")
#MAKE NEW
# LDA.defaultSetDataGroup()
# LDA.main(alpha=10,beta=0.05,num_iter=500,K=8,toy_size=350,skew=True)
# LDA.main(alpha=8,beta=0.01,num_iter=250,K=6,toy_size=200,skew=True)
# LDA.save()
#LOAD AND VISUALIZE
LDA.load()
# LDA.visualize_topics()
# LDA.visualize_words(40)
#LABEL
LDA.name_topics()
LDA.save()
#TEST SET
LDA.load()
LDA.initTestSet(100)
# print(LDA.testSet)
LDA.labelTestCorpus()
LDA.produce_csv()
# LDA.real_pred_topics_csv()
LDA.save()


import data_manager_2 as dm

LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization")
#MAKE NEW
# LDA.main(alpha=10,beta=0.1,num_iter=350,K=6,toy_size=200)
# LDA.save()
#LOAD AND VISUALIZE
LDA.load()
LDA.visualize_words()
# LDA.visualize_topics()
#LABEL
# LDA.name_topics()
LDA.save()
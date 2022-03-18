import data_manager_2 as dm

# LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization")
LDA = dm.LDAManager("All_Broad_Topics_Improved_Tokenization_2")
#MAKE NEW
# LDA.main(alpha=10,beta=0.1,num_iter=500,K=8,toy_size=350)
# LDA.save()
#LOAD AND VISUALIZE
LDA.load()
LDA.visualize_words(20)
# LDA.visualize_topics()
#LABEL
# LDA.name_topics()
LDA.save()


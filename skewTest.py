import data_manager_2 as dm

#init
LDA_S = dm.LDAManager("Skew_Steep_All_Topics")
LDA_R = dm.LDAManager("Skew_Steep_All_Topics")

#train models
LDA_S.main(alpha=10,beta=0.05,num_iter=500,K=6,toy_size=100,skew=True)
LDA_R.main(alpha=10,beta=0.05,num_iter=500,K=6,toy_size=100,skew=False)

LDA_S.save()
LDA_R.save()

#name topics
LDA_S.load()
LDA_R.load()

LDA_S.visualize_words(40)
LDA_R.visualize_words(40)

LDA_S.name_topics()
LDA_R.name_topics()

LDA_S.save()
LDA_R.save()

#test models
LDA_S.initTestSet(100)
LDA_R.initTestSet(100)

LDA_S.labelTestCorpus()
LDA_R.labelTestCorpus()

#format results
LDA_S.produce_csv()
LDA_R.produce_csv()

LDA_S.save()
LDA_R.save()

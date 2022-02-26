if __name__ == '__main__':
	import data_manager as dm 
	
	experiment_count = 0
	topics = 4
	alpha = .1
	beta = .002
	num_iter = 2000


	# test_med_misc_df = dm.data_manager("sci_med_2")
	# test_med_misc_df.load()
	# test_med_misc_df.parseGroup("./20news-bydate/20news-bydate-train/sci.med",30)
	# test_med_misc_df.parseGroup("./20news-bydate/20news-bydate-train/misc.forsale",30)
	# test_med_misc_df.tokenize()
	# test_med_misc_df.save()
	# test_med_misc_df.LDA(2)

	Test = dm.custom_manager("Test_Alt_Misc")
	Test.load()
	Test.reset()
	Test.parseGroup("./20news-bydate/20news-bydate-train/alt.atheism",200)
	Test.parseGroup("./20news-bydate/20news-bydate-train/misc.forsale",200)
	Test.parseGroup("./20news-bydate/20news-bydate-train/rec.sport.hockey",200)
	Test.parseGroup("./20news-bydate/20news-bydate-train/talk.politics.guns",200)
	Test.tokenize()
	Test.save()
	Test.LDA2(topics,alpha,beta,num_iter)
	Test.save()
	Test.visualize(35)
	Test.save()





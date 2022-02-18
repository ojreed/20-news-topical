if __name__ == '__main__':
	import data_manager as dm 
	
	experiment_count = 0
	topics = 5
	alpha = 10
	beta = 5
	num_iter = 100

	sci_med_df = dm.custom_manager("sci_med")
	sci_med_df.load()
	# sci_med_df.parseGroup("./20news-bydate/20news-bydate-train/sci.med")
	# sci_med_df.tokenize()
	# sci_med_df.save()
	sci_med_df.LDA(topics,alpha,beta,num_iter)
	sci_med_df.save()
	sci_med_df.visualize(30)
	sci_med_df.save()





if __name__ == '__main__':
	import data_manager as dm 
	
	experiment_count = 0
	topics = 5
	alpha = 10
	beta = 0.001
	num_iter = 100

	experiment_count+=1
	msc_sale_df = dm.custom_manager("msc_sale")
	msc_sale_df.load()
	msc_sale_df.LDA(topics,alpha,beta,num_iter)
	msc_sale_df.visualize()
	msc_sale_df.save()


	"""
	sci_med_df = dm.custom_manager("sci_med")
	sci_med_df.load()
	experiment_count = 0
	num_iter = 7
	topics = 7
	alpha = 10
	beta = 5
	num_iter = 100

	experiment_count+=1
	print("Experiment " + str(experiment_count) + ":")
	print("Topics: " + str(topics) + "; Alpha: " + str(round(alpha,2)) + "; Beta: " + str(round(beta,4)) + "; Iterations: " + str(num_iter))
	sci_med_df.LDA(topics,alpha,beta,num_iter)
	sci_med_df.visualize()
	"""
	"""
	for x in range(1,10):
		alpha = x*0.10
		
		print("Experiment " + str(experiment_count) + ":")
		print("Topics: " + str(topics) + "; Alpha: " + str(round(alpha,2)) + "; Beta: " + str(round(beta,4)) + "; Iterations: " + str(num_iter))
		sci_med_df.LDA(topics,alpha,beta,num_iter)
		sci_med_df.visualize()
	"""
	"""
	for topics in range(2,15,3):
		alpha = 0 
		for x in range(5):
			alpha += 0.2
			beta = 0
			for y in range(10):
				beta += 0.01
				experiment_count+=1
				print("Experiment " + str(experiment_count) + ":")
				print("Topics: " + str(topics) + "; Alpha: " + str(round(alpha,2)) + "; Beta: " + str(round(beta,2)) + "; Iterations: " + str(num_iter))
				sci_med_df.LDA(topics,alpha,beta,num_iter = 25)
				sci_med_df.visualize()
				print("\n\n\n")
	"""		
			
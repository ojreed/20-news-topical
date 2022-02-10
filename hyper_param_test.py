if __name__ == '__main__':
	import data_manager as dm 
	sci_med_df = dm.custom_manager("sci_med")
	sci_med_df.load()
	experiment_count = 0
	for topics in range(1,5,3):
		alpha = 0.01 
		while (alpha <= 1):
			beta = 0.001
			while (beta <= .005):
				experiment_count+=1
				print("Experiment " + str(experiment_count) + ":")
				print("Topics: " + str(topics) + "; Alpha: " + str(alpha) + "; Beta: " + str(beta))
				sci_med_df.LDA(topics,alpha,beta)
				sci_med_df.visualize()
				print("\n\n\n")
				beta += 0.01
			alpha += .2
if __name__ == '__main__':
	import data_manager as dm 


	religion_misc_df = dm.data_manager("talk_religion_misc")
	religion_misc_df.load()
	for x in range(2,10):
		religion_misc_df.genSimLDA(topics = x)
	sci_med_df = dm.data_manager("sci_med")
	sci_med_df.parseGroup("./20news-bydate/20news-bydate-train/sci.med")
	sci_med_df.tokenize()
	for x in range(2,10):
		sci_med_df.genSimLDA(topics = x)


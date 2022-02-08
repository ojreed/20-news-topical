if __name__ == '__main__':
	import data_manager as dm 
	sci_med_df = dm.custom_manager("sci_med")
	sci_med_df.load()
	sci_med_df.LDA()



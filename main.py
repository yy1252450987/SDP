import os, sys
import pandas as pd
#from utils import *
from train import *



def main():
	MainDir = '/export/home/yangsheng/project/SDP/'
	DataDir = MainDir + 'data/'
	ResultDir = MainDir + 'result/'
	SoftwareDir = MainDir + 'software/'

	integrate_file = DataDir + 'Integrated_Dataset.txt'
	disprot_file = DataDir + 'search_in_disprot.tsv'
	id_mapping_file = DataDir + 'HUMAN_9606_idmapping_selected.tab'
	uniprot_seq_file = DataDir + 'uniprot_filtered_reviewed_2019_11_11.tab'
	seqvec_model_dir = SoftwareDir+'seqvec/uniref50_v2/'

	humvar_file = DataDir + 'humvar_tool_scores.csv'
	exovar_file = DataDir + 'exovar_tool_scores.csv'
	swissvar_file = DataDir + 'swissvar_selected_tool_scores.csv'
	varibench_file = DataDir + 'varibench_selected_tool_scores.csv'
	predictSNP_file = DataDir + 'predictSNP_selected_tool_scores.csv'
	
	dataset_names = ['humvar', 'exovar', 'swissvar', 'varibench', 'predictSNP']
	dataset_files = [humvar_file, exovar_file, swissvar_file, varibench_file, predictSNP_file]
	
	encode_dict = {'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'C':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'D':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'E':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'F':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'G':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'H':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'I':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
				   'K':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
				   'L':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
				   'M':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
				   'N':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
				   'O':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
				   'P':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
				   'Q':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
				   'R':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
				   'S':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
				   'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
				   'U':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
				   'V':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
				   'W':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
				   'Y':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
				   'X':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
	# mapping uniprot id for swissvar and predictSNP datasets
	# for i, dataset_file in enumerate(dataset_files):
	# 	df = pd.read_csv(dataset_file)
	# 	name = dataset_names[i]
	# 	if name == 'swissvar' or name == 'predictSNP':
	# 		mapped_df = UniprotIDMapping(df, id_mapping_file)
	# 	else:
	# 		mapped_df = df
	# 	mapped_df.to_csv(DataDir + '%s_scores_filter.csv' % name)

	# extarct sequence via uniprot id
	# i = int(sys.argv[1])
	# df = pd.read_csv(DataDir + '%s_scores_filter.csv' % dataset_names[i])
	# GetProteinPredIDR(df, DataDir + 'd2p2/%s/' % dataset_names[i])

	# extarct Prediction IDR vis uniprotid
	# for i in range(5):
	# 	df = pd.read_csv(DataDir + '%s_scores_filter.csv' % dataset_names[i])
	# 	df = GetVarLocatedPredIDR(df, DataDir + 'd2p2/%s/' % dataset_names[i], uniprot_seq_file)
	# 	df.to_csv(DataDir + '%s_PredIDR_SEQ.csv' % dataset_names[i], index=None)
		
	# for i in range(len(dataset_names)):
	# 	dataset_name = dataset_names[i]
	# 	score_df = pd.read_csv(DataDir + '%s_scores_filter.csv' % dataset_name)
	# 	append_df = pd.read_csv(DataDir + '%s_PredIDR_SEQ.csv' % dataset_name)
	# 	for j in range(1, 8):
	# 		threshold = j
	# 		predictor_performance_df = CompDiffPerf(score_df, append_df, threshold, dataset_name)
	# 		predictor_performance_df.to_csv(ResultDir+'%s_predictor_performance_trd%s.csv' % (dataset_name, threshold), index=None, float_format='%.4f')
	# dataset_name = 'humvar'
	# append_df = pd.read_csv(DataDir + '%s_PredIDR_SEQ.csv' % dataset_name, keep_default_na=False)
	# for lenth in [100, 200, 500]:
	# 	data_df = GetTrainData(append_df, IDR=True, maxlength=lenth)
	# 	data_df.to_csv(DataDir+'humvar_disorder_len%s.data' % lenth, index=None)

	#batch_size = 1000

	data_df = pd.read_csv(DataDir+'humvar_disorder_len1000.data')
	#part1_df = data_df.iloc[:int(data_df.shape[0]/2), ]
	#part2_df = data_df.iloc[int(data_df.shape[0]/2):, ]
	#SeqEncode_seqvec(part1_df, seqvec_model_dir, DataDir, cuda_device=0)
	#SeqEncode_seqvec(part2_df, seqvec_model_dir, DataDir, cuda_device=1)
	# np.save(DataDir+'humvar_disorder_x_wild_len1000_seqvec.npy', x_wild_array)
	# np.save(DataDir+'humvar_disorder_x_mut_len1000_seqvec.npy', x_mut_array)
	# np.save(DataDir+'humvar_disorder_y_len1000_seqvec.npy', y_array)
	
	# for i in range(0, data_df.shape[0], batch_size):
	# 	start_idx, end_idx = i, i+batch_size
	# 	if i+batch_size > data_df.shape[0]:
	# 		end_idx = data_df.shape[0]
	# 	batch_df = data_df.iloc[i:i+batch_size,]
	# 	x_wild_array, x_mut_array, y_array = SeqEncode_seqvec(batch_df, seqvec_model_dir)
	# 	np.save(DataDir+'humvar/humvar_disorder_x_wild_len1000_seqvec.bs%s_%s.npy' % (batch_size, int(i/batch_size)), x_wild_array)
	# 	np.save(DataDir+'seeds/humvar_disorder_x_mut_len1000_seqvec.bs%s_i%s.npy' % (batch_size, int(i/batch_size)), x_mut_array)
	# 	np.save(DataDir+'seeds/humvar_disorder_y_len1000_seqvec.bs%s_i%s.npy' % (batch_size, int(i/batch_size)), y_array)

	model_train(DataDir+'humvar/', 'humvar_disorder_x_*_len1000_seqvec', data_df, base_network='cnn1d_bilstm')

main()

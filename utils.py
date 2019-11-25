import pandas as pd
import numpy as np

import keras

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import scipy

import json
#import urllib2

import os, random

from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path

def DownloadSeq(method=''):
	'''
	function: using different method (uniprot, ensembl) to download sequence for all variants
	return: fasta sequence
	'''
	return 0

def UniprotIDMapping(benchmark_df, id_mapping_file):
	id_mapping_df = pd.read_csv(id_mapping_file, header=None,sep='\t')
	ensp2uniprot_mapping_dict = {}
	for idx, row in id_mapping_df.iterrows():
		uniprot_id = row[0]
		np_ids = str(row[3]).split(';')
		ensp_ids = str(row[20]).split(';')
		for ensp_id in ensp_ids:
			ensp2uniprot_mapping_dict[ensp_id.strip()] = uniprot_id

		for np_id in np_ids:
			ensp2uniprot_mapping_dict[np_id.strip()] = uniprot_id

	uniprot_ids = []
	not_c = 0
	for idx, row in benchmark_df.iterrows():
		ensp_id = row['Ensembl-Protein-ID']
		uniprot_id = row['Uniprot-Accession']
		if '?' in uniprot_id or 'HUMAN' in uniprot_id:
			if ensp_id in ensp2uniprot_mapping_dict:
				uniprot_ids.append(ensp2uniprot_mapping_dict[ensp_id])
			else:
				uniprot_ids.append('')
		else:
			uniprot_ids.append(uniprot_id)
	benchmark_df['Uniprot-Accession'] = uniprot_ids
	benchmark_df = benchmark_df[benchmark_df['Uniprot-Accession']!='']
	return benchmark_df


def GetProteinPredIDR(benchmark_df, out_dir, d2p2_url = 'http://d2p2.pro/api/seqid'):
	'''
	'''
	uniprot_ids = benchmark_df['Uniprot-Accession'].unique()
	#predictors = {'VLXT','VSL2b','PrDOS','PV2','IUPred-S','IUPred-L','Espritz-N','Espritz-X','Espritz-D','','',''}

	for uniprot_id in uniprot_ids:
		data = 'seqids=["%s"]' % uniprot_id
		try:
			rqs = urllib2.Request('http://d2p2.pro/api/seqid', data)
			response = json.loads(urllib2.urlopen(rqs).read())
			records = response[uniprot_id][0][2]['disorder']['disranges']
			outfile = open(out_dir + '%s.d2p2' % uniprot_id, 'w')
			outfile.write('predictor,start,end\n')
			for record in records:
				outfile.write(','.join(record))
				outfile.write('\n')
			outfile.close()
			print('Y:'+uniprot_id)
		except:
			print('N:'+uniprot_id)


def GetVarLocatedPredIDR(benchmark_df, dp2p_dir, uniprot_seq_file):
	cols = ['label','uniprot_id','pos','d2p2','VLXT','VSL2b','PrDOS','PV2','IUPred-S','IUPred-L','Espritz-N','Espritz-X','Espritz-D','IDR','wild_seq', 'mut_seq']
	append_df = pd.DataFrame(np.zeros((benchmark_df.shape[0], len(cols))), dtype='int')
	append_df.columns = cols
	uniprot_seq_df = pd.read_csv(uniprot_seq_file, sep='\t')

	for idx, row in benchmark_df.iterrows():
		label = row['True Label']
		uniprot_id = row['Uniprot-Accession']
		pos = row['AA-Pos']
		d2p2_file = dp2p_dir + '%s.d2p2' % uniprot_id
		ref_aa = row['REF-AA']
		alt_aa = row['ALT-AA']
		append_df.loc[idx, 'label'] = 1 if row['True Label'] == 1 else 0
		append_df.loc[idx, 'pos'] = pos
		append_df.loc[idx, 'uniprot_id'] = uniprot_id
		if os.path.exists(d2p2_file):
			append_df.loc[idx, 'd2p2'] = 1
			pred_idr_df = pd.read_csv(d2p2_file, skiprows=1)
			pred_idr_df.columns = ['predictor','start','end']
			for x_idx, x_row in pred_idr_df.iterrows():
				predictor = x_row['predictor']
				start = int(x_row['start'])
				end = int(x_row['end'])
				if pos >= start and pos <= end:
					append_df.loc[idx, predictor] = 1
		
		seq_ = uniprot_seq_df[uniprot_seq_df['Entry'] ==uniprot_id]['Sequence'].values
		if len(seq_) >= 1 and len(seq_[0]) >= pos:
			seq = seq_[0]
			append_df.loc[idx, 'wild_seq'] = seq
			append_df.loc[idx, 'mut_seq'] = seq[:pos-1]+alt_aa+seq[pos:]
		else:
			append_df.loc[idx, 'wild_seq'] = ''
			append_df.loc[idx, 'mut_seq'] = ''
	for col in cols[4:12]:
		append_df['IDR'] += append_df[col]
	return append_df


def CompDiffPerf(score_df, append_df, threshold=1,dataset='humvar'):
	score_df.columns = ['null_index','True Label','#RS-ID','CHR','Nuc-Pos','REF-Nuc','ALT-Nuc','MAF',\
	'Ensembl-Gene-ID','Ensembl-Protein-ID','Ensembl-Transcript-ID','Uniprot-Accession',\
	'AA-Pos','REF-AA','ALT-AA','MutationTaster','MutationTaster-Label','MutationAssessor',\
	'MutationAssessor-Label','PolyPhen2','PolyPhen2-Label','CADD','SIFT','SIFT-Label','LRT',\
	'LRT-Label','FatHMM-U','FatHMM-U-Label','FatHMM-W','log(FatHMM-W Wd)','log(FatHMM-W Wn)',\
	'log(FatHMM-W Wn)-Label','GERP++','PhyloP','Condel','Condel-Label','Condel+','Condel+-Label',\
	'Logit','Logit-Label','Logit+','Logit+-Label']

	predictors = ["MutationTaster", "MutationAssessor", "PolyPhen2", "CADD", "SIFT", "LRT",\
	"FatHMM-U", "FatHMM-W", "GERP++","PhyloP"]

	results = []
	all_df = pd.concat([score_df, append_df], axis=1)
	all_df = all_df[(all_df.d2p2==1) & (all_df.wild_seq!='')]
	all_df.fillna(0, inplace=True)
	all_df['SIFT'] = -all_df['SIFT']
	all_df['LRT'] = -all_df['LRT']
	all_df['FatHMM-U'] = -all_df['FatHMM-U']
	all_df['FatHMM-W'] = -all_df['FatHMM-W']

	IDR_df = all_df[all_df.IDR>=threshold]
	FOLD_df = all_df[all_df.IDR<threshold]
	out = [dataset, threshold, IDR_df.shape[0], FOLD_df.shape[0], all_df.shape[0]]
	out = map(str, out)
	print(','.join(out))
	for predictor in predictors:
		#print(IDR_df.shape[0], FOLD_df.shape[0], all_df.shape[0])
		
		idr_rocauc = roc_auc_score(IDR_df['label'], IDR_df[predictor])
		fold_rocauc = roc_auc_score(FOLD_df['label'], FOLD_df[predictor])
		diff_rocauc = abs(idr_rocauc-fold_rocauc)

		random.seed(1)

		right_diff_case = 0
		n = 1000

		for i in range(n):
			idr_index = random.sample(range(all_df.shape[0]), IDR_df.shape[0])
			fold_index = list(set(range(all_df.shape[0])) - set(idr_index))
			#print(idr_index, fold_index)
			rand_IDR_df = all_df.iloc[idr_index, :]
			rand_FOLD_df = all_df.iloc[fold_index, :]
			rand_idr_rocauc = roc_auc_score(rand_IDR_df['label'], rand_IDR_df[predictor])
			rand_fold_rocauc = roc_auc_score(rand_FOLD_df['label'], rand_FOLD_df[predictor])
			rand_diff_rocauc = abs(rand_idr_rocauc-rand_fold_rocauc)
			if rand_diff_rocauc >= diff_rocauc:
				right_diff_case += 1

		pvalue = right_diff_case*1.0/n
		results.append([predictor, idr_rocauc, fold_rocauc, abs(idr_rocauc-fold_rocauc), pvalue])
	results = pd.DataFrame(results, columns=['predictor','idr_rocauc','fold_rocauc','diff_rocauc','pvalue'])
	return results

def CalMetricScore(pred_prob, pred_label, true, fmt='dict'):
	
	if len(pred_label) != 0:
		accuracy = accuracy_score(true, pred_label)
		precision = precision_score(true, pred_label)
		recall = recall_score(true, pred_label)
		f1 = f1_score(true, pred_label)
		mcc = matthews_corrcoef(true, pred_label)
		roc_auc = roc_auc_score(true, pred_prob)
		scores = {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1_score':f1, 'mcc':mcc, 'roc_auc':roc_auc}
		if fmt == 'list':
			scores = [accuracy, precision, recall, f1, mcc, roc_auc]
	else:
		roc_auc = roc_auc_score(true, pred_prob)
		scores = {'accuracy':None, 'precision':None, 'recall':None, 'f1_score':None, 'mcc':None, 'roc_auc':roc_auc}
		if fmt == 'list':
			scores = [None, None, None, None, None, roc_auc]
	return scores

def GetTrainData(benchmark_df, IDR=True, maxlength=1000):
	if IDR:
		benchmark_df = benchmark_df[(benchmark_df.d2p2==1) & (benchmark_df.IDR==1) & (benchmark_df.wild_seq!='')]
	else:
		benchmark_df = benchmark_df[(benchmark_df.d2p2==1) & (benchmark_df.IDR==0) & (benchmark_df.wild_seq!='')]

	left_len = int(maxlength/2)
	right_len = int(maxlength/2)
	for idx, row in benchmark_df.iterrows():
		print(idx)
		pos = row['pos']-1
		wild_seq = row['wild_seq']
		mut_seq = row['mut_seq']
		#seq_left = '^'*left_len + wild_seq[:pos]
		#seq_right = wild_seq[pos+1:] + '$'*right_len
		start_index = pos-left_len if pos-left_len>=0 else 0
		end_index = pos+right_len if pos+right_len<len(wild_seq) else len(wild_seq)-1
		benchmark_df.loc[idx, 'wild_seq'] = wild_seq[start_index:end_index+1]
		benchmark_df.loc[idx, 'mut_seq'] = mut_seq[start_index:end_index+1]
	benchmark_df = benchmark_df[['uniprot_id', 'pos', 'wild_seq','mut_seq','label']]

	return benchmark_df


def SeqEncode_onehot(train_df, onehot_encode_dict):
	x = []
	for idx, row in train_df.iterrows():
		wild_seq = row['wild_seq']
		mut_seq = row['mut_seq']
		wild_embedding = np.asarray([onehot_encode_dict[c] for c in wild_seq])
		mut_embedding = np.asarray([onehot_encode_dict[c] for c in mut_seq])
		x.append([wild_embedding, mut_embedding])
	y = train_df['label']
	
	return x, y

def SeqEncode_seqvec(data_df, model_dir, DataDir, save=True, cuda_device=0):
	x = []
	model_dir = Path(model_dir)
	weights = model_dir / 'weights.hdf5'
	options = model_dir / 'options.json'
	seqvec  = ElmoEmbedder(options, weights, cuda_device=cuda_device)

	if save:
		for idx, row in data_df.iterrows():
			print(idx)
			wild_seq = row['wild_seq']
			mut_seq = row['mut_seq']
			wild_embedding = seqvec.embed_sentence(list(wild_seq))
			mut_embedding = seqvec.embed_sentence(list(mut_seq))
			np.save(DataDir+'humvar/humvar_disorder_x_wild_len1000_seqvec.%s.npy' % idx, wild_embedding)
			np.save(DataDir+'humvar/humvar_disorder_x_mut_len1000_seqvec.%s.npy' % idx, mut_embedding)


# def LoadDataGenerator(data_df, seqvec_model_dir, batch_size):
# 	while True:
# 		number = np.random.choice(data_df.index.values, batch_size, replace=False)
# 		batch_df = data_df.ix[number,]
# 		x_wild_array, x_mut_array, y_array = SeqEncode_seqvec(batch_df, seqvec_model_dir)
# 		yield ([x_wild_array, x_mut_array], y_array)


# def LoadData(indexs = [], data_df, DataDir='', prefix1='humvar_disorder_x_wild_len1000_seqvec', prefix2='humvar_disorder_x_mut_len1000_seqvec'):
# 	x_wild = []
# 	x_mut = []
	
# 	for idx in indexs:
# 		x_wild_ = np.load(DataDir+'/'+prefix1+'.%s.npy' % idx)
# 		x_mut_ = np.load(DataDir+'/'+prefix2+'.%s.npy' % idx)
# 		x_wild.append(x_wild_)
# 		x_mut.append(x_mut_)
# 	x_wild = np.asarray(x_wild_)
# 	x_mut = np.asarray(x_mut_)
# 	y = np.asarray(data_df['label']).reshape(-1, 1)
# 	return x, y


	# x_mut_seqs = list(train_df['mut_seq'].apply(lambda x:list(x)))
	# x_mut_elmo_embedding, x_mut_elmo_mask = seqvec.batch_to_embeddings(x_mut_seqs)

	# x_wild_array = x_wild_elmo_embedding.cpu().numpy()
	# x_mut_array = x_mut_elmo_embedding.cpu().numpy()

	#y = np.asarray(train_df['label']).reshape(-1, 1)
	#np.save(DataDir+'humvar/humvar_disorder_y_mut_len1000_seqvec.npy', mut_embedding)


# def SeqEncode_seqvec1(train_df, model_dir):
# 	x = []
# 	model_dir = Path(model_dir)
# 	weights = model_dir / 'weights.hdf5'
# 	options = model_dir / 'options.json'
# 	seqvec  = ElmoEmbedder(options, weights,cuda_device=0)
# 	wild_seqs = []
# 	mut_seqs = []

# 	for idx, row in train_df.iterrows():
# 		print(idx)
# 		wild_seq = row['wild_seq']
# 		mut_seq = row['mut_seq']
# 		wild_embedding = seqvec.embed_sentence(list(wild_seq))
# 		mut_embedding = seqvec.embed_sentence(list(mut_seq))
# 		x.append([wild_embedding, mut_embedding])

# 	y = train_df['label']

# 	return x, y

# def DataReshape(x):
# 	reshape_x = []
# 	for i in range(len(x)):
# 		wild_seq, mut_seq = x[i]
# 		wild_seq1 = wild_seq.reshape((wild_seq.shape[0], wild_seq.shape[1], 1))
# 		mut_seq1 = mut_seq.reshape((mut_seq.shape[0], mut_seq.shape[1], 1))
# 		reshape_x.append([wild_seq1, mut_seq1])
# 	return reshape_x


# def DataCheck(x_data, y_data):
# 	for i in range(len(x_data)):
# 		x1 = x_data[i][0]
# 		x2 = x_data[i][1]
# 		#print(x1, x2)
# 		s1 = np.isnan(x1).sum()
# 		s2 = np.isnan(x1).sum()
# 		if s1>0 or s2>0:
# 			print('%s: NA value(%s, %s)' % (i, s1, s2))



#!/usr/bin/python

from allennlp.commands.elmo import ElmoEmbedder
from pathlib import Path

import numpy as np

def SeqEncode_seqvec(train_df, model_dir):
	x = []
	model_dir = Path(model_dir)
	weights = model_dir / 'weights.hdf5'
	options = model_dir / 'options.json'
	seqvec  = ElmoEmbedder(options, weights,cuda_device=0)
	wild_seqs = []
	mut_seqs = []

	for idx, row in train_df.iterrows():
		print(idx)
		wild_seq = row['wild_seq']
		mut_seq = row['mut_seq']
		wild_embedding = seqvec.embed_sentence(list(wild_seq))
		mut_embedding = seqvec.embed_sentence(list(mut_seq))
		x.append([wild_embedding, mut_embedding])

	y = train_df['label']

	return x, y

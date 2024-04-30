
import numpy as np
from sklearn.decomposition import PCA

from auxiliary_scripts.constants import MAX_ACTIONS_PER_FIELD, MAX_SEQUENCE_LEN
from auxiliary_scripts.featurization import FIELD_ORDER, get_len, get_bounds


def ngram_sequence_to_feature_counts(sequences, fvec_size, keep_fields=None):

    keep = list(FIELD_ORDER.keys()) if keep_fields is None else keep_fields
    keep_idx = [FIELD_ORDER[k] for k in keep]

    num_feats = 2**fvec_size

    fmat = np.zeros((len(sequences), num_feats))

    for ii in range(len(sequences)):

        for field_idx in keep_idx:

            field_acts = sequences[ii][field_idx]

            for act in field_acts:
                for tidx in act:
                    fmat[ii][tidx%num_feats] += 1

    return fmat


def get_remapper(sequences, feat_n=2, keep_ns=[1,2], start_idx=1):

    list_lens = {ii:get_len(ii, feat_n) for ii in np.arange(100)}
    keep_indices = {ll:get_bounds(ll, keep_ns) for ll in list_lens.keys()}

    remap = {}
    ii = start_idx

    for seq in sequences:
        
        for field_seq in seq:

            for act in field_seq:
                cur_list_len = list_lens[len(act)]
                cur_keep_indices = keep_indices[cur_list_len]
                keep_act = act[cur_keep_indices]

                for tidx in keep_act:

                    if tidx not in remap:
                        remap[tidx] = ii
                        ii += 1

    return remap, set(remap.keys())


def remap_sequences(sequences, remap, feat_n=2, keep_ns=[1,2], max_actions_field=100):

    list_lens = {ii:get_len(ii, feat_n) for ii in np.arange(100)}
    keep_indices = {ll:get_bounds(ll, keep_ns) for ll in list_lens.keys()}

    unknown = max(remap.values()) + 1
    new_seqs = []
    max_len = 0

    for seq in sequences:
        cur_seq = []
        cur_len = 0

        for field in seq:

            for ii, act in enumerate(field):

                if ii == max_actions_field:
                    break

                cur_list_len = list_lens[len(act)]
                cur_keep_indices = keep_indices[cur_list_len]
                keep_act = act[cur_keep_indices]

                for tidx in keep_act:
                    cur_seq.append(remap.get(tidx, unknown))
                    cur_len += 1

        cur_seq = np.asarray(cur_seq)
        new_seqs.append(cur_seq)

        if cur_len > max_len:
            max_len = cur_len
    
    new_seqs = np.asarray(new_seqs, dtype=object)

    return new_seqs, max_len


def pad_sequences(sequences, max_len, padding_idx=0):

    padded_sequences = np.ones((len(sequences), max_len)) * padding_idx

    for ii, seq in enumerate(sequences):        
        for jj, tidx in enumerate(seq):

            if jj == max_len:
                break

            padded_sequences[ii][jj] = tidx

    return padded_sequences

def flatten_sequences(sequences):
    flat_sequences = []

    for seq in sequences:
        cur_seq = []        
        for field_seq in seq:
            for act in field_seq:
                for tidx in act:
                    cur_seq.append(tidx)

        cur_seq = np.asarray(cur_seq)
        flat_sequences.append(cur_seq)
    return np.asarray(flat_sequences, dtype=object)


def pooling(X):
    min_idx = np.argmin(np.mean(X, axis=1))
    return X[:, min_idx]


def remapped_sequence_to_feature_counts(sequences, fvec_size):

    fmat = np.zeros((len(sequences), fvec_size))
    for ii in range(len(sequences)):
        for tidx in sequences[ii]:
            fmat[ii, tidx] += 1
    
    return fmat

def get_feature_processor(raw_data, train_dataset, feat_n, processing_type):

    if 'bag_of_n_grams' in processing_type:
        red_type = processing_type.split('_')[-1]
        return get_feature_processor_ngram(raw_data, train_dataset, feat_n=feat_n, red_type=red_type)
    
    elif 'sequence' in processing_type:
        return get_feature_processor_sequence(raw_data, train_dataset, feat_n=feat_n) 


class EmptyTransformer:
    def __init__(self):
        pass
    def transform(self, X):
        return X


def get_processor(processing_type, n_components, inputs):

    if processing_type == 'none':
        return EmptyTransformer(), input.shape[1]
    elif processing_type == 'pca':
        print(f'PCA: {n_components}')
        processor = PCA(n_components=n_components, random_state=0)
        processor.fit(inputs)
        print(sum(processor.explained_variance_ratio_))
        return processor, n_components

def get_feature_processor_ngram(raw_data, train_dataset, feat_n=2, red_type='pca'):
    fvec_size = 14 # hashing trick, project the n-grams to a 2^14 dimensional vector
    param = 2**10 # number of principal components for PCA or random projection directions
    keep_ns = np.arange(1, feat_n+1)

    if train_dataset[0] == 'ep':
        TRACE_PER_SAMPLE = 5
        pca_seqs = np.vstack([v[:TRACE_PER_SAMPLE] for v in raw_data['feats']])       

    else:
        pca_seqs = np.concatenate([raw_data[sbn][0] for sbn in train_dataset])

    print(f'red type = {red_type}')

    sequences_to_fmat = lambda seqs: ngram_sequence_to_feature_counts(seqs, fvec_size)

    scaler = lambda feats: np.log2(feats+1) # log scaler
    
    transformer, nfeats = get_processor(red_type, n_components=param, inputs=scaler(sequences_to_fmat(pca_seqs)))

    transform_pipeline = lambda sequences: transformer.transform(scaler(sequences_to_fmat(sequences)))

    return transform_pipeline

def get_feature_processor_sequence(raw_data, train_dataset, feat_n=2):

    if train_dataset[0] == 'ep':
        TRACE_PER_SAMPLE = 5
        process_seqs = np.vstack([v[:TRACE_PER_SAMPLE] for v in raw_data['feats']])       

    else:
        process_seqs = np.concatenate([raw_data[sbn][0] for sbn in train_dataset])

    remap, _ = get_remapper(process_seqs, feat_n=feat_n, keep_ns=[1], start_idx=1) # 0 is padding
    
    transform_pipeline = lambda x: pad_sequences(remap_sequences(x, remap, feat_n=feat_n, keep_ns=[1], max_actions_field=MAX_ACTIONS_PER_FIELD)[0], max_len=MAX_SEQUENCE_LEN) # truncate after 1500 tokens

    return transform_pipeline
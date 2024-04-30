import numpy as np
import hashlib

import auxiliary_scripts.tokenization as tokenization


# the order of fields in the featurized report
FIELD_ORDER = {'regs_created':0, 'regs_deleted':1, 'mutexes_created':2, 'processes_created':3, 'files_created':4, 'processes_injected':5}
ORDER_FIELD = {v:k for k,v in FIELD_ORDER.items()}


def filter_report_w_vocabulary(trep, kept_tokens_info):

    tokenized_report = {}

    rep_rares = []

    for field, acts in trep.items():
        
        tokenized_report[field] = []

        for ii, act in enumerate(acts):

            cur_act = []

            for jj, (tok, type) in enumerate(act):

                s0, s1 = ('proc_f', 'proc_p')

                if type == 'proc_m':
                    if tokenization.get_tok_str(s0, tok) in kept_tokens_info:
                        type = s0
                    elif tokenization.get_tok_str(s1, tok) in kept_tokens_info:
                        type = s1
                    
                tok_str = tokenization.get_tok_str(type, tok)

                if tok_str not in kept_tokens_info:
                    rep_rares.append((tok, field, ii, jj, type))
                    tok_str = tokenization.get_tok_str(type, '<rare_singleton>')

                cur_act.append(tok_str)


            tokenized_report[field].append(cur_act)

    # replace the rare tokens that occur multiple times in the report with special tokens (preserve non-random rare tokens)
    tok_fields = {}

    for r in rep_rares:
        if r[0] not in tok_fields:
            tok_fields[r[0]] = []
        tok_fields[r[0]].append(r[1])

    tok_fields = [(k, v) for k, v in tok_fields.items() if len(v) > 1]
    tok_fields = sorted(tok_fields, key=lambda r: (len(np.unique(r[1])), len(r[1])), reverse=True)[:25]
    multiples_idx = {r[0]:(idx+1) for idx, r in enumerate(tok_fields)}

    for tok, field, ii, jj, type in rep_rares:
        if tok in multiples_idx:
            tokenized_report[field][ii][jj] = tokenization.get_tok_str(type, f'<rare_{multiples_idx[tok]}>')

    return tokenized_report

def get_bounds(len_list, ns):

    helper = lambda n: n*len_list - (n*(n-1))/2

    all_indices = []

    for n in ns:
        if n > len_list:
            cur_indices = np.arange(helper(len_list-1), helper(len_list)).astype(int)
        else:
            cur_indices = np.arange(helper(n-1), helper(n)).astype(int)

        all_indices.extend(cur_indices)

    return sorted(np.unique(all_indices))

def get_len(num_ngrams, n):
    a = (2*num_ngrams)/n
    b = n - 1
    return int((a + b)/2)


def filtered_report_to_sequence(trep):

    sequence = []

    for field_idx in [0,1,2,3,4,5]:
        field_acts = trep[ORDER_FIELD[field_idx]]
        field_seq = []

        for act in field_acts:
            field_seq.append(np.asarray(act, dtype=str))
        
        sequence.append(np.asarray(field_seq, dtype=object))
        
    return np.asarray(sequence, dtype=object)


def find_ngrams(input_list, n):

    all_grams = []
    split_indices = {}
    for ii in range(1, n+1):
        n_grams =  list(zip(*[input_list[i:] for i in range(ii)]))
        n_grams = ['::'.join([str(t) for t in n_gram]) for n_gram in n_grams]
        all_grams.extend(n_grams)
        split_indices[ii] = len(all_grams)

    return all_grams, split_indices


### HASHING TRICK

# token_to_md5 is a lookup
def get_token_md5(cur_str, n_buckets, start_idx=0, token_to_md5={}):

    if cur_str in token_to_md5:
        md5 = token_to_md5[cur_str]['md5']
        token_to_md5[cur_str]['count'] += 1
    else:
        md5 = int(hashlib.md5(cur_str.encode('utf-8')).hexdigest(), 16)
        token_to_md5[cur_str] = {'md5':md5, 'count':1}

    return start_idx + (md5 % 2**n_buckets)

# avoid hashing same ngram multiple times, use the lookup
def sequences_to_ngrams(sequences, n, token_to_md5={}):   

    n = int(n)

    fvec_size = 32

    all_ngram_seq = []

    for seq in sequences:
        cur_ngram_seq = []

        for field_seq in seq:
            cur_ngram_field_seq = []
            # take the n-gram of each action seperately - action order is not consistent between sandboxes so it doesn't matter
            for act in field_seq:
                ngrams, split_indices = find_ngrams(act, n)
                hash_indices = [get_token_md5(ngram, fvec_size, start_idx=0, token_to_md5=token_to_md5) for ngram in ngrams]
                cur_ngram_field_seq.append(np.asarray(hash_indices[:split_indices[n]], dtype=object))

            cur_ngram_field_seq = np.asarray(cur_ngram_field_seq, dtype=object)
            cur_ngram_seq.append(cur_ngram_field_seq)

                
        cur_ngram_seq = np.asarray(cur_ngram_seq, dtype=object)
        all_ngram_seq.append(cur_ngram_seq)

    return np.asarray(all_ngram_seq, dtype=object)
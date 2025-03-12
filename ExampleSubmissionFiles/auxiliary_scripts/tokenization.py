import re

from auxiliary_scripts.constants import MAX_LENGTH_PER_ACTION, MAX_LENGTH_PER_FIELD, DEPTH_SEP
import auxiliary_scripts.normalization_string_cleaner as sc


# token cleaner
def TOKEN_CLEANER(tok):
    return sc.clean_up_token(tok, max_token_length=15)

def TOKEN_FILTER(tok):
    return False if sc.get_token_length(tok) < 1 else True


# assign unique indices to the kept tokens - also assign indices to enumerated rare tokens
def get_tok_str(tok_type, tok):
    return f'{tok_type}::{tok}'


def tokenize_report(rep, rep_fields=None, add_depth=True, concat=False, return_token_types=True):
    
    tokenized_report = {}

    if rep_fields is None: # all fields
        collect_fields = ['regs_created', 'regs_deleted', 'mutexes_created', 'processes_created', 'files_created', 'processes_injected'] 
        
    else: # collect specified fields
        collect_fields = rep_fields
    

    for field in rep:
        if field not in collect_fields:
            continue
        tokenized_report[field] = []

        for entry in rep[field]:
            tokens = tokenize_entry(entry, field, add_depth)
            filtered_tokens, filtered_tok_types = [], []

            # clean up and filter the tokens
            for ii in range(len(tokens)):
                ct = TOKEN_CLEANER(tokens[ii][0])
                if TOKEN_FILTER(ct): # returns true for the tokens that will be kept
                    filtered_tokens.append(ct)
                    filtered_tok_types.append(tokens[ii][1])
            

            if return_token_types:
                tokens = list(zip(filtered_tokens, filtered_tok_types))
            else:
                tokens = filtered_tokens

            if len(tokens) == 0:
                continue

            tokens = tokens[:MAX_LENGTH_PER_ACTION[field]]

            if concat:
                tokenized_report[field].extend(tokens) # create a single list of tokens from all actions

            else:
                tokenized_report[field].append(tokens) # each action will be a different list of tokens

    tokenized_report = {field:entries[:MAX_LENGTH_PER_FIELD[field]] for field, entries in tokenized_report.items()}

    return tokenized_report

def add_path_depth(path_toks, enabled=False):
    return [f'{t}{DEPTH_SEP}{ii+1}' for ii,t in enumerate(path_toks)] if enabled else path_toks

def tokenize_entry(entry, field, add_depth):
    if field == 'regs_created':
        # only take the key for now
        tokens = add_path_depth(entry.split('\\'), add_depth)
        tok_types = ['reg_c' for _ in tokens]

    elif field == 'regs_deleted':
        tokens = add_path_depth(entry.split('\\'), add_depth)
        tok_types = ['reg_d' for _ in tokens]

    elif field == 'mutexes_created':
        tokens = [entry] # no need to split
        tok_types = ['mtx_c']

    elif field == 'processes_created':
        tokens = []
        tok_types = []

        # first split with space
        ptokens = entry.split(' ')
        for t in ptokens:
            if any(re.findall(r'\b[a-z]:\\', t)) or '.exe' in t or '.dll' in t or '.bat' in t or '.ocx' in t or '.sys' in t: #path or executable
                path_toks = t.split('\\')
                path, fname = path_toks[:-1], path_toks[-1]
                tokens.extend(add_path_depth(path, add_depth))
                tok_types.extend(['proc_p' for _ in path])

                ftoks, fttypes = tokenize_filename(fname, 'proc')
                tokens.extend(ftoks)
                tok_types.extend(fttypes)
            
            # urls
            elif any(re.findall(r'(https?:\\)|(www.)|(ftps?:\\)', t)):
                matches = list(re.finditer(r'(https?:\\)|(www.)|(ftps?:\\)', t))
                url_type = re.split('\:|\.', matches[-1].group())[0]
                tokens.append(f'<{url_type}url>')
                tok_types.append('proc_m')

            # ipaddress:port
            elif any(re.findall(r'((localhost)|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(:[0-9]{1,5})?', t)):
                tname = '<'
                tname = tname + 'lochost' if 'localhost' in t else tname + 'ipaddr'
                tname = tname + 'port' if ':' in t else tname     
                tname = tname + '>'
                tokens.append(tname)
                tok_types.append('proc_m')

            else: # no path
                tokens.append(t)
                tok_types.append('proc_m') # miscellaneous

    elif field == 'files_created':
        tokens = []
        tok_types = []

        ptokens = entry.split('\\')
        path, fname = ptokens[:-1], ptokens[-1]
        tokens.extend(add_path_depth(path, add_depth))
        tok_types.extend(['file_p' for _ in path])

        ftoks, fttypes = tokenize_filename(fname, 'file')
        tokens.extend(ftoks)
        tok_types.extend(fttypes)  

    elif field == 'processes_injected':
        tokens = []
        tok_types = []
        ftoks, fttypes = tokenize_filename(entry, 'inj')
        tokens.extend(ftoks)
        tok_types.extend(fttypes)    

    assert len(tokens) == len(tok_types), 'entry: num tokens != num token types'

    return [(t,tt) for t, tt in zip(tokens, tok_types)]


def tokenize_filename(entry, act_type):
    fname, exts = sc.split_filename_extension(entry)

    # handle the extension
    if len(exts) > 1: # multiple extensions
        merged_exts = ''.join(exts)
        ext_type = f'{act_type}_me'
    else:
        merged_exts = exts[0]
        ext_type = f'{act_type}_e'

    return [fname, merged_exts], [f'{act_type}_f', ext_type]
import re

from auxiliary_scripts.constants import DEPTH_SEP


def token_shortener(token, max_token_length=15):

    tok, depth = get_token_characters(token)

    short_tok = ''.join(tok[:max_token_length]) 

    short_tok = short_tok + f'{DEPTH_SEP}{depth}' if depth else short_tok

    return short_tok

def get_token_length(token):
    tok, _ = get_token_characters(token)
    return len(tok)

def get_token_characters(token):
    rem_characters = []
    cur_index = 0

    token_depth = token.split(DEPTH_SEP)
    if len(token_depth) > 1 and is_int(token_depth[-1]):
        tok, depth = DEPTH_SEP.join(token_depth[:-1]), token_depth[-1]
    else:
        tok, depth = token, None

    for match in re.finditer('<\w+>', tok):
        
        rem_characters.extend(tok[cur_index: match.span()[0]])
        rem_characters.append(match.group())
        cur_index = match.span()[1]

    rem_characters.extend(tok[cur_index:])

    return rem_characters, depth


def clean_up_token(token, max_token_length=10):

    # remove non-ascii
    new_token = re.sub(r'[^\x00-\x7F]', 'x', token)
    
    # remove all non alphanumeric characters
    # new_token = re.sub('[^a-z0-9]', '', new_token.lower())
    new_token = re.sub(r'[_]|[\s+]|,|\$|\?|!|\.|-|{|}|:|\[|\]|~|=|/|//|\"|\(|\)|\'|\\|\-', '', new_token.lower())

    return token_shortener(new_token, max_token_length)

def split_filename_extension(fname):

    # filename and extension
    tokens = fname.split('.')
    fname_toks = tokens[:1]

    exts = tokens[1:]
    exts = process_extensions(exts)

    fname_toks.extend([t for t in exts if len(t)>=5]) # long token, unlikely to be an extension
    exts = [t for t in exts if len(t) < 5 and len(t) > 0]

    if len(exts) == 0:
        exts = ['noextension']

    return '_'.join(fname_toks), exts

# ignore after any non alpha numeric character
def process_extensions(exts):
    return [re.split('[^a-z0-9]', ext)[0] for ext in exts]


def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
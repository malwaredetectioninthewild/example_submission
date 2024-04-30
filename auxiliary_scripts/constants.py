ALL_TOKEN_TYPES = ['reg_c', 'reg_d', 'mtx_c', 'proc_p', 'proc_f', 'proc_me', 
                   'proc_e', 'proc_m', 'inj_f', 'inj_me', 'inj_e', 
                   'file_p', 'file_f', 'file_me', 'file_e']



FIELD_ORDER = {'regs_created':0, 'regs_deleted':1, 'mutexes_created':2, 'processes_created':3, 'files_created':4, 'processes_injected':5}

ORDER_FIELD = {0:'regs_created', 1:'regs_deleted', 2:'mutexes_created', 3:'processes_created', 4:'files_created', 5:'processes_injected'}


MAX_LENGTH_PER_FIELD  = {'regs_created': 100, 'regs_deleted': 100, 
                         'mutexes_created': 60, 'processes_created': 15, 
                         'files_created': 250, 'processes_injected': 25}

MAX_LENGTH_PER_ACTION = {'regs_created': 14, 'regs_deleted': 9, 
                         'mutexes_created': 1, 'processes_created': 25, 
                         'files_created': 12, 'processes_injected': 2}


NUM_OUT = 2

MAX_ACTIONS_PER_FIELD = 75

MAX_SEQUENCE_LEN = 1500

SPLIT_TS = 1522540800 # APRIL 1st 2018 - training data comes before this date

ALL_SB_NAMES = ['Habo', 'Cuckoo']

VALID_RATIO = 0.1

VALID_EP_NUM_HASHES = (500, 100)

SPLIT_RANDOM_SEED = 17

DEPTH_SEP = '@@'

LABEL = 1 # 1: malware detection, 2: PUP detection

# pup samples are labeled as class '2' by default, this flag maps them to flag 1 for binary classification.
PUP_LABEL_1 = True

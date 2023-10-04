import numpy as np
import utils
import pickle as pk
from os.path import join as oj


def load_pk(path):
        with open(path, 'rb') as f:
            x = pk.load(f)
        return x


def load_dummy_data(num_data = 500, num_input = 4000, num_output = 8, num_channels = 2):
    x = np.random.uniform(size = (num_data,num_channels, num_input))
    y = np.random.uniform(size = (num_data, num_output))
    y[:,1] +=.1 
    y = (y > .2).astype(np.int32)
    
    return x, y

def load_data(data_path):
    
    ir_spectra_data = load_pk(oj(data_path,'ir_spectra_data.pkl'))
    ir_spectra_index = load_pk(oj(data_path,'ir_spectra_index.pkl'))
    x,y  = utils.basic_filter(ir_spectra_data, ir_spectra_index)
    return x,y


def get_class_weights(y):
    props = y.mean(axis=0)
    anti_props = 1- props
    return props,anti_props

    
def get_split(my_len, seed=42):
    

    split_idxs = np.arange(my_len)
    np.random.seed(seed)
    np.random.shuffle(split_idxs)

    num_train, num_val, = (
        int(my_len * 0.7),
        int(my_len * 0.15),
    )
    num_test = my_len - num_train - num_val
    train_idxs = split_idxs[:num_train]
    val_idxs = split_idxs[num_train : num_train + num_val]
    test_idxs = split_idxs[-num_test:]
    return train_idxs, val_idxs, test_idxs

if __name__ == "__main__":
    x, y = load_dummy_data(num_data = 1000)
    print(get_class_weights((y)))
        
    print(get_class_weights((y)).sum())
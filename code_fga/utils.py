import numpy as np
import pandas as pd
import pickle as pk
import os

import copy
import torch


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x
    

def extract_ranges(index, firstx, lastx, n_bins=50):
    ## bin the initials
    firstx = index['firstx'].astype(float)
    lastx = index['lastx'].astype(float)
    firstx = pd.cut(firstx, bins=20) # returns Interval object
    lastx = pd.cut(lastx, bins=20)

    ranges = {}
    for fx, lx in zip(firstx, lastx):
        if fx is not np.nan and lx is not np.nan:
            key = (fx.left, lx.left)
            if key in ranges:
                ranges[key] += 1.
            else:
                ranges[key] = 1.
    ranges = np.array([[k[0], k[1], v] for k, v in ranges.items()])
    ranges = np.sort(ranges, axis=0)
    ranges = np.array([[str((r[0], r[1])), float(r[2])] for r in ranges])
    ranges = pd.DataFrame(ranges, columns=['ranges', 'frequency'])
    ranges['frequency'] = ranges['frequency'].astype(float)
    return ranges


def get_freqs(data, f_groups):
    f_groups_freqs = data[f_groups].sum(0)
    return f_groups_freqs


def cudify(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def transm_to_absorbance(y_transm: np.array):
    #y in absorbance
    return -np.log10(y_transm)


def micro_to_inversecm(x_micro: np.array):
    # x in inverse cm
    return  10000*np.power(x_micro,-1)


def filter_info(index, filters=None):
    if filters is None:
        filters = (('state', 'gas'), ('yunits', 'ABSORBANCE'), ('xunits', '1/CM'))
    for property, filter in filters:  # index is a pandas dataframe
        index = index[index[property] == filter]
    index.dropna(inplace=True)
    idxs = list(index.index)
    return idxs, index


def basic_filter(data, index, filters=None, npoints=100):
    idxs, _ = filter_info(index, filters)

    inputs = []
    targets = []
    for cas, row in data.items():
        if cas in idxs:
            x, y = row['x'], row['y']

            wave_numbers = np.linspace(min(x), max(x), npoints)
            spectra = np.interp(wave_numbers, x, y)

            inputs.append([wave_numbers, spectra])
            targets.append(row['target'])

    return np.array(inputs, dtype=float), np.array(targets, dtype=float)


def max_domain_filter(data, index, filters=None, npoints=100):
    idxs, index = filter_info(index, filters)
    index['domain_size'] = index['lastx'] - index['firstx']
    max_domain = index['domain_size'].max()
    dt = max_domain / npoints
    print('target dt is %.2f' % dt)
 
    inputs = []
    targets = []
    for cas, row in data.items():
        if cas in idxs:
            x, y = row['x'], row['y']
 
            # get the start value
            nsamples = len(x)
            start, end = min(x), max(x)
            domain = end - start
           
            side_difference = (max_domain - domain) /  2.  # each side results in 2
            new_start = start - side_difference
            new_end = end + side_difference
            assert np.isclose((new_end - new_start), max_domain)
           
            dt = domain / nsamples  # dt of this data point
            n_pad_domain_side = int(np.floor(side_difference / dt))
            zero_pad = np.zeros((n_pad_domain_side,))
           
            y = np.concatenate([zero_pad, y, zero_pad])
            x = np.linspace(new_start, new_end, len(y))
 
            wave_numbers = np.linspace(min(x), max(x), npoints)
            spectra = np.interp(wave_numbers, x, y)
 
            inputs.append([wave_numbers, spectra])
            targets.append(row['target'])
 
    return np.array(inputs, dtype=float), np.array(targets, dtype=float)


def filter_by_state(ir_dataset: dict, ir_index: pd.DataFrame, keep: str = 'gas'):

    new_dataset = {}
    cas_gas_state = ir_index[ir_index['state']== keep].index
    for key, value in ir_dataset.items():
        if key in cas_gas_state:
            new_dataset.update({key:value})

    return new_dataset


def filter_datapoints(data, index, filters):
    filtered_cas, _ = filter_info(index, filters)

    xs = []
    ys = []
    targets = []

    #Fixed wavenumber sequence, query points for interpolation
    for cas, row in data.items():
        #take only spectra in the filtered cas

        some_weird_unit = False

        if cas in filtered_cas: 
        
            if index.loc[cas,'xunits']=='MICROMETERS':
                x = micro_to_inversecm(row['x'])
            elif index.loc[cas,'xunits']=='1/CM':
                x = row['x']
            else:
                some_weird_unit = True


            if index.loc[cas,'yunits']=='TRANSMITTANCE':
                y = transm_to_absorbance(row['y'])
            elif index.loc[cas,'yunits']=='ABSORBANCE':
                y = row['y']
            else:
                some_weird_unit = True

            if some_weird_unit:
                continue
            else:
                #interpolate by quering the fixed wavenumbers. Pad with zeros at left or right for x outside the quering vector    

                xs.append(x)
                ys.append(y)
                targets.append(row['target'])

    return xs, ys, np.array(targets, dtype=float)




def fixed_domain_filter(data, index, filters, delta=4, min_wav=500, max_wav=3900):
    filtered_cas, _ = filter_info(index, filters)

    inputs = []
    targets = []

    #Fixed wavenumber sequence, query points for interpolation
    fixed_wavenumbers =  np.linspace(min_wav, max_wav, int(np.ceil((max_wav-min_wav)/delta)))

    for cas, row in data.items():

        some_weird_unit = False

        if cas in filtered_cas: 
            

            if index.loc[cas,'xunits']=='MICROMETERS':
                x = micro_to_inversecm(row['x'])
            elif index.loc[cas,'xunits']=='1/CM':
                x = row['x']
            else:
                some_weird_unit = True

            if index.loc[cas,'yunits']=='TRANSMITTANCE':
                y = transm_to_absorbance(row['y'])
            elif index.loc[cas,'yunits']=='ABSORBANCE':
                y = row['y']
            else:
                some_weird_unit = True

            if some_weird_unit:
                continue
            else:
                #interpolate by quering the fixed wavenumbers. Pad with zeros at left or right for x outside the quering vector    
                counts = np.interp(fixed_wavenumbers, x, y, left = 0.0, right = 0.0)

                inputs.append(counts)
                targets.append(row['target'])

    return np.array(inputs, dtype=float), np.array(targets, dtype=float)


def dump_to_database(data, key, path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            database = pk.load(f)
        database[key] = data
    else:
        database = {key:data}
    
    with open(path, 'wb') as f:
        pk.dump(database, f)
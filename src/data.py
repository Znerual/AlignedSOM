from typing import Tuple, Dict
import numpy as np
import gzip

from src.config import config

# This code was used from https://github.com/smnishko/PySOMVis from the module SOMToolBox_Parse.py
# it loads data provided from the SOMToolBox


def load_dataset(name: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load input_data, components, weights and classinfo for a dataset from the datasets folder.

    Args:
        name (str):
            Name of the dataset
            Possible values: 'animals', 'iris', 'chainlink', '10clusters', 'BostonHousing', 'climate'

    Returns:
        Tuple[Dict, Dict, Dict, Dict]:
            input_data, components, weights and classinfo
            each dict has an attribute ['arr'] which holds the actual data
    """

    input_data = _read_weight_file(_db_file_string(name, 'vec'))
    components = _read_weight_file(_db_file_string(name, 'tv'))
    weights = _read_weight_file(_db_file_string(name, 'wgt.gz'))
    classinfo = _read_weight_file(_db_file_string(name, 'cls'))
    return input_data, components, weights, classinfo


def _db_file_string(dataset_name: str, extension: str) -> str:
    return str((config.DATASET_PATH / dataset_name / f'{dataset_name}.{extension}').resolve())


def _read_weight_file(file_path):
    df = {}
    if file_path[-3:len(file_path)] == '.gz':
        with gzip.open(file_path, 'rb') as file:
            df = _read_vector_file_to_df(df, file)
    else:
        with open(file_path, 'rb') as file:
            df = _read_vector_file_to_df(df, file)
    file.close()
    return df


def _read_vector_file_to_df(df, file):
    for byte in file:
        line = byte.decode('UTF-8')
        if line.startswith('$'):
            df = _parse_vector_file_metadata(line, df)
        else:
            if df['type'] == 'vec' or df['type'] == 'som':
                c = df['vec_dim']
            else:
                c = df['xdim']
            if 'arr' not in df:
                df['arr'] = np.empty((0, c), dtype=float)
            df = _parse_weight_file_data(line, df, c)
    return df


def _parse_weight_file_data(line, df, c):
    splitted = line.rstrip().split(' ')
    try:
        if df['type'] == 'vec' or df['type'] == 'class_information' or df['type'] == 'som':
            res = np.array(splitted[0:c]).astype(float)
        else:
            res = np.array(splitted[0:c]).astype(str)
        df['arr'] = np.append(df['arr'], [res], axis=0)
    except Exception:
        raise ValueError('The input-vector file does not match its unit-dimension.')
    return df


def _parse_vector_file_metadata(line, df):
    splitted = line.strip().split(' ')
    if splitted[0] == '$TYPE':
        df['type'] = splitted[1]
    if splitted[0] == '$XDIM':
        df['xdim'] = int(splitted[1])
    elif splitted[0] == '$YDIM':
        df['ydim'] = int(splitted[1])
    elif splitted[0] == '$VEC_DIM':
        df['vec_dim'] = int(splitted[1])
    elif splitted[0] == '$CLASS_NAMES':
        df['classes_names'] = splitted[1:]
    return df

from pathlib import Path
import numpy as np
import gzip


DATASET_PATH = Path(__file__).parent.parent / 'datasets'


# load one of animals, iris, chainlink, 10clusters, BostonHousing
def load_dataset(name: str):
    input_data = read_weight_file(_db_file_string(name, 'vec'))
    components = read_weight_file(_db_file_string(name, 'tv'))
    weights = read_weight_file(_db_file_string(name, 'wgt.gz'))
    classinfo = read_weight_file(_db_file_string(name, 'cls'))
    return input_data, components, weights, classinfo


def _db_file_string(dataset_name: str, extension: str) -> str:
    return str((DATASET_PATH / dataset_name / f'{dataset_name}.{extension}').resolve())


def read_weight_file(file_path):
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

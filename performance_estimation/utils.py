# %%
import os
import json
from easydict import EasyDict
import csv

# %%
def make_file_dir(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def storFile(data,fileName):
    with open(fileName,'w',newline ='') as f:
        # mywrite = csv.writer(f)
        # mywrite.writerow(data)
        data.to_csv(fileName)

def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)
    args = EasyDict(args)
    return args

def save_json(args, filename):
    with open(filename, "w") as f:
        json.dump(args, f, indent=4)

def merge_parameter(base_params, override_params):
    """
    Update the parameters in ``base_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    Parameters
    ----------
    base_params : namespace or dict
        Base parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    namespace or dict
        The updated ``base_params``. Note that ``base_params`` will be updated inplace. The return value is
        only for convenience.
    """
    if override_params is None:
        return base_params
    is_dict = isinstance(base_params, dict)
    for k, v in override_params.items():
        if is_dict:
            if k not in base_params:
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(base_params[k]) != type(v) and base_params[k] is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(base_params[k]), type(v)))
            base_params[k] = v
        else:
            if not hasattr(base_params, k):
                raise ValueError('Key \'%s\' not found in base parameters.' % k)
            if type(getattr(base_params, k)) != type(v) and getattr(base_params, k) is not None:
                raise TypeError('Expected \'%s\' in override parameters to have type \'%s\', but found \'%s\'.' %
                                (k, type(getattr(base_params, k)), type(v)))
            setattr(base_params, k, v)
    return base_params



def train_test_split_ratio(data, test_ratio, ratio):
    np.random.seed(42)
    fo
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)) * test_ratio 
    test_indices = shuffled_indices[:test_ratio]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]
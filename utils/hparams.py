import yaml


def load_hparam(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        docs = yaml.load_all(f, Loader=yaml.FullLoader)
        hparam_dict = dict()
        for doc in docs:
            for k, v in doc.items():
                hparam_dict[k] = v
    return hparam_dict


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setattr__
    __delattr__ = dict.__delattr__

    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


class HParam(DotDict):
    def __init__(self, file):
        super(HParam, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = DotDict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = DotDict.__getattr__
    __setattr__ = DotDict.__setattr__
    __delattr__ = DotDict.__delattr__

from ml_collections import config_dict
from rec.models import PolyasUrnModel


def get_config():
    config = config_dict.ConfigDict()
    config.model = PolyasUrnModel
    config.dataset_name = config_dict.placeholder(str)
    config.max_num_edges = -1
    config.seed = 0
    config.bias = 1
    return config

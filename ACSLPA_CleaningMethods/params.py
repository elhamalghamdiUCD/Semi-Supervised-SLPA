from ruamel.yaml import YAML


class Params:
    '''Inspired by: 
    https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams'''
    def __init__(self, yaml_file, model_name):
        with open(yaml_file) as f:
            yaml = YAML(typ='safe')
            # Load and find
            yaml_map = yaml.load(f)
            params_dict = yaml_map[model_name]
            for param in params_dict:
                setattr(self, param, params_dict[param])




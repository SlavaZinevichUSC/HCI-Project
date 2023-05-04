import json
from core.Config import config


def WriteToFile(data, filename):
    if config.bias_print_method == 'json':
        with open(filename, 'w') as f:
            f.write(json.dumps(data))

from argparse import Namespace

import yaml


def save_namespace_to_yaml(namespace, file_path):
    # Convert the namespace object to a dictionary
    data = vars(namespace)

    # Save the dictionary as YAML
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def load_namespace_from_yaml(file_path):
    # Load the dictionary from YAML
    with open(file_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Convert the dictionary to a namespace object
    namespace = Namespace(**data)

    return namespace
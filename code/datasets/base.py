import sys

try:
    sys.path.insert(0, '../')
    import datasets.cancer as cancer
finally:
    pass


def Dataset(params):
    name = params['name']

    dataset = None
    if 'cancer' in name:
        dataset = getattr(cancer, name)

    return dataset(params)

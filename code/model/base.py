import sys

try:
    sys.path.insert(0, '../')
    import model.resnet as resnet
    import model.vgg as vgg
    import model.alexnet as alexnet
    import model.lenet as lenet
    import model.densenet as densenet
finally:
    pass


def Model(params):
    name = params['name']

    model = None
    if 'ResNet' in name:
        model = getattr(resnet, 'CancerResNet')
    elif 'VGG' in name:
        model = getattr(vgg, 'CancerVGGNet')
    elif 'Dense' in name:
        model = getattr(densenet, 'CancerDenseNet')

    constructor_params = {k: v for k, v in
                          filter(lambda item: item[0] in model.__init__.__code__.co_varnames, params.items())}
    return model(**constructor_params)


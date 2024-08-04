import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

try:
    sys.path.insert(0, '.')
    from utils import parse, get_logger, print_dct
finally:
    pass

warnings.filterwarnings('ignore')


def _plot(
    ax,
    idx,
    dataset,
    log_path,
    model,
    clip_min,
    clip_max,
    metric
):
    ax.grid(True)

    name = f'{log_path}/{model.lower()}'
    name += '/losses.npz'

    npz = np.load(name)

    epochs = npz['arr_0']
    train_loss = npz['arr_1']
    valid_loss = npz['arr_2']
    train_acc = npz['arr_3']
    valid_acc = npz['arr_4']

    train_loss = np.clip(
        train_loss,
        a_min=clip_min,
        a_max=clip_max
    )

    valid_loss = np.clip(
        valid_loss,
        a_min=clip_min,
        a_max=clip_max
    )

    train_acc = np.clip(
        train_acc,
        a_min=clip_min,
        a_max=clip_max
    )

    valid_acc = np.clip(
        valid_acc,
        a_min=clip_min,
        a_max=clip_max
    )

    if metric == 'Loss':
        if idx % 2 == 0:
            ax.plot(
                epochs+1,
                train_loss,
            )
        else:
            ax.plot(
                epochs+1,
                valid_loss,
            )
    elif metric == 'Accuracy':
        if idx % 2 == 0:
            ax.plot(
                epochs+1,
                train_acc,
            )
        else:
            ax.plot(
                epochs+1,
                valid_acc,
            )


def plot():
    args = parse()
    args_str = print_dct(args)

    num_subplots = len(args['datasets'])

    models = args['models']
    datasets = args['datasets']

    clip_min, clip_max = args['clip']

    plot_path = args['plot_path']
    log_path = args['log_path']

    dpi = args['dpi']
    os.makedirs(plot_path, exist_ok=True)

    logger = get_logger(plot_path + 'plot.log')
    logger.info(args_str)

    metrics = ['Loss', 'Accuracy']
    datasets_type = ['Train', 'Valid']
    for metric in metrics:
        _, axes = plt.subplots(num_subplots, 2, figsize=(15, 5), dpi=dpi)
        for i, ax in enumerate(axes):
            for model in models:
                plt.subplots_adjust(
                    left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=0.3
                )

                _plot(
                    ax,
                    i,
                    datasets[i // 2],
                    log_path,
                    model,
                    clip_min,
                    clip_max,
                    metric
                )
            ax.set_title(datasets_type[i] + ' ' + metric)
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric)
            ax.legend(models)

        plt.savefig(f'{plot_path}{metric}.png', dpi=dpi)

    plt.cla()
    plt.clf()


if __name__ == '__main__':
    plot()

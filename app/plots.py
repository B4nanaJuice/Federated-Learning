# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config import config
from app.models import Server, NormalMLP

def compare_loss(files: List[str]) -> None:

    # colors = ['#13579B', '#579B13', '#9B1357', '#57139B', '#9B5713', '#139B9B', '#9B9B13']
    fig, ax = plt.subplots(1, 1)

    for idx in range(len(files)):
        file = files[idx]
        server: Server = Server(global_model = NormalMLP())
        server.load_metrics(filename = file)

        avg_loss: List[float] = [sum(_)/len(_) for _ in server.training_loss]
        ax.plot(avg_loss, label = file)
        # average = sum(avg_loss)/len(avg_loss)
        # ax.hlines(average, 0, len(avg_loss)-1, label = f'Avg {file}', linestyles = 'dashed', colors = colors[idx])

    ax.set_xlabel('Round id')
    ax.set_ylabel('Mean Square Error Loss')
    ax.set_title('Comparison of training loss')
    # plt.grid(axis = 'y')
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.show()
    return

def compare_MSE(files: List[str]) -> None:

    MAE: Dict[str, Tuple] = {}
    deltas: Dict[str, Tuple] = {}

    for idx in range(len(files)):
        file = files[idx]
        server: Server = Server(global_model = NormalMLP())
        server.load_metrics(filename = file)

        MAE[file] = (
            mean_absolute_error(server.test_predictions['load_true'], server.test_predictions['load']),
            mean_absolute_error(server.test_predictions['pv_true'], server.test_predictions['pv']),
            mean_absolute_error(server.test_predictions['net_true'], server.test_predictions['net'])
            )
        
        deltas[file] = tuple([
            abs(MAE[file][_] - MAE[files[0]][_])
            for _ in range(3)
        ])

    # colors = ['#13579B', '#579B13', '#9B1357', '#57139B', '#9B5713', '#139B9B']
    x = np.arange(len(['load', 'pv', 'net']))  # the label locations
    width = round(0.9/len(files), 2)  # the width of the bars
    multiplier = -1.5

    fig, ax = plt.subplots(layout = 'constrained')

    for attribute, measurement in MAE.items():
        offset = width * multiplier
        # rects = ax.bar(x + offset, measurement, width, label = attribute, color = colors[int(multiplier+1.5)])
        rects = ax.bar(x + offset, measurement, width, label = attribute)
        labels: List[str] = [
            f'{measurement[_]:.3f}' if attribute == files[0] else f'$\\Delta = {deltas[attribute][_]:.3f}$'
            for _ in range(len(measurement))
        ]
        ax.bar_label(rects, padding = 3, labels = labels)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('MAE for each target')
    ax.set_xticks(x + width, ['load', 'pv', 'net'])
    ax.legend()
    ax.set_yscale('log')

    plt.show()
    return

def compare_scoring(files: List[str]) -> None:

    fig, (ax1, ax2) = plt.subplots(2, 1, layout = 'constrained')
    colors = ['#13579B', '#579B13', '#9B1357', '#57139B']

    width = 0.15  # the width of the bars
    multiplier = -1

    for idx in range(len(files)):
        file: str = files[idx]

        with open(f'{config.SAVE_DATA_PATH}/{file}.json') as f:
            data = json.load(f)

        x = np.arange(len(data['rejected']))  # the label locations

        offset = width * multiplier
        rects = ax1.bar(x + offset, data['rejected'], width, label = file.replace("_", " ").capitalize(), color = colors[idx])

        ax1.bar_label(rects, padding = 3, labels = [] if 'similarity' in file else [f'${"\\sigma" if "distribution" not in file else "bins"}={_}$' for _ in data['parameters']], rotation = 90)
        multiplier += 1

        # ax1.plot(data['rejected'], label = file.replace("_", " ").capitalize(), color = colors[idx])

        for _ in range(len(data['RMSE'])):
            if data['rejected'][_] > 90:
                line = 'dotted'
            elif data['rejected'][_] > 5:
                line = 'dashed'
            else:
                line = 'solid'
            
            
            ax2.plot(data['RMSE'][_], color = colors[idx], linestyle = line)

    ax2.plot([], color = '#000000', label = '<5 rejected models', linestyle = 'solid')
    ax2.plot([], color = '#000000', label = '$\\in [5;90]$ rejected models', linestyle = 'dashed')
    ax2.plot([], color = '#000000', label = '>90 rejected models', linestyle = 'dotted')

    ax1.legend()
    ax2.legend()

    ax1.set_title('Percentage of rejected models by the server')
    ax1.set_xlabel('$\\sigma$')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_ylim([0, 110])

    ax2.set_title('Loss with filtered models')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Root Mean Square Error (RMSE)')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.show()

def compare_defenses():
    defenses: List[str] = ['fedavg', 'norm', 'cbaa', 'krum', 'mkrum', 'tmean', 'rfa', 'fltrust', 'distribution', 'distance']
    malicious_percentages: List[int | str] = [0, 5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 100, 'partial', 'total']

    # Data
    load = np.zeros((len(malicious_percentages), len(defenses)))
    pv = np.zeros((len(malicious_percentages), len(defenses)))
    net = np.zeros((len(malicious_percentages), len(defenses)))

    for _p in range(len(malicious_percentages)):
        p = malicious_percentages[_p]
        for _d in range(len(defenses)):
            d = defenses[_d]

            with open(f'save/defenses/{d}_{p}_grouped.json', mode = 'r', encoding = 'utf-8') as f:
                _data = json.load(f)

            load[_p, _d] = mean_absolute_error(_data['predictions']['load_true'], _data['predictions']['load'])
            pv[_p, _d] = mean_absolute_error(_data['predictions']['pv_true'], _data['predictions']['pv'])
            net[_p, _d] = mean_absolute_error(_data['predictions']['net_true'], _data['predictions']['net'])

    # Color line (True = colored, False = black 'n white)
    color_load: List[bool] = [min(_) in [_[-1], _[-2]] for _ in load]
    color_pv: List[bool] = [min(_) in [_[-1], _[-2]] for _ in pv]
    color_net: List[bool] = [min(_) in [_[-1], _[-2]] for _ in net]

    # RBG Image
    nrows, ncols = load.shape

    norm_load = (load - load.min()) / (load.max() - load.min())
    norm_pv = (pv - pv.min()) / (pv.max() - pv.min())
    norm_net = (net - net.min()) / (net.max() - net.min())

    img_load = np.zeros((nrows, ncols, 3))
    img_pv = np.zeros((nrows, ncols, 3))
    img_net = np.zeros((nrows, ncols, 3))

    # Colormaps
    cmap_color = plt.cm.viridis

    for i in range(nrows):

        img_load[i] = cmap_color(load[i])[:, :3] if color_load[i] else np.zeros((len(load[i]), 3)) + .5
        img_pv[i] = cmap_color(pv[i])[:, :3] if color_pv[i] else np.zeros((len(load[i]), 3)) + .5
        img_net[i] = cmap_color(net[i])[:, :3] if color_net[i] else np.zeros((len(load[i]), 3)) + .5

    # Plot

    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, layout = 'constrained')

    ax1.imshow(img_load)
    ax2.imshow(img_pv)
    # ax3.imshow(img_net)

    for i in range(nrows):
        for j in range(ncols):
            ax1.text(
                j, i,
                str(round(load[i, j], 2)),
                ha='center',
                va='center',
                color='white' if load[i, j] < 1 else 'black'
            )

            ax2.text(
                j, i,
                str(round(pv[i, j], 2)),
                ha='center',
                va='center',
                color='white' if pv[i, j] < 1 else 'black'
            )

            # ax3.text(
            #     j, i,
            #     str(round(net[i, j], 2)),
            #     ha='center',
            #     va='center',
            #     color='white' if net[i, j] < 1 else 'black'
            # )

    # Labels
    ax1.set_xticks(range(ncols))
    ax2.set_xticks(range(ncols))
    # ax3.set_xticks(range(ncols))
    ax1.set_yticks(range(nrows))
    ax2.set_yticks(range(nrows))
    # ax3.set_yticks(range(nrows))


    ax1.set_xticklabels(defenses, rotation = 90)
    ax2.set_xticklabels(defenses, rotation = 90)
    # ax3.set_xticklabels(defenses, rotation = 90)
    ax1.set_yticklabels([f'{_}%' if type(_) == int else _ for _ in malicious_percentages])
    ax2.set_yticklabels([f'{_}%' if type(_) == int else _ for _ in malicious_percentages])
    # ax3.set_yticklabels([f'{_}%' for _ in malicious_percentages])

    ax1.set_xlabel("Defense method")
    ax2.set_xlabel("Defense method")
    # ax3.set_xlabel("Defense method")
    ax1.set_ylabel("Malicious clients percentage")

    ax1.set_title('MAE on Load forecasting')
    ax2.set_title('MAE on PV forecasting')
    # ax3.set_title('MAE on Net forecasting')

    # plt.tight_layout()
    plt.show()
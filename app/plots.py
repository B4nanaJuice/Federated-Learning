# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from sklearn.metrics import mean_absolute_error

from config import config
from app.models import Server, NormalMLP

def compare_loss(files: List[str]) -> None:

    colors = ['#13579B', '#579B13', '#9B1357', '#57139B', '#9B5713']
    fig, ax = plt.subplots(1, 1)

    for idx in range(len(files)):
        file = files[idx]
        server: Server = Server(global_model = NormalMLP())
        server.load_metrics(filename = file)

        avg_loss: List[float] = [sum(_)/len(_) for _ in server.training_loss]
        ax.plot(avg_loss, label = file, color = colors[idx])
        average = sum(avg_loss)/len(avg_loss)
        ax.hlines(average, 0, len(avg_loss)-1, label = f'Avg {file}', linestyles = 'dashed', colors = colors[idx])

    ax.set_xlabel('Round id')
    ax.set_ylabel('Mean Square Error Loss')
    ax.set_title('Comparison of average training loss for clean run and data poisoning attacks')
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

    x = np.arange(len(['load', 'pv', 'net']))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = -1

    fig, ax = plt.subplots(layout = 'constrained')

    for attribute, measurement in MAE.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label = attribute)
        labels: List[str] = [
            f'{measurement[_]:.3f}' if attribute == files[0] else f'$\\Delta = {deltas[attribute][_]:.3f}$'
            for _ in range(len(measurement))
        ]
        ax.bar_label(rects, padding = 3, labels = labels)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Absolute Error')
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

        ax1.bar_label(rects, padding = 3, labels = [f'${"\\sigma" if "distribution" not in file else "bins"}={_}$' for _ in data['parameters']], rotation = 90)
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
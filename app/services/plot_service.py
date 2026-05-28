# Imports
import json
import numpy as np
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt

# Create static class wiht methods
class PlotService:
    
    # Plot metric (MAE, MSE, RMSE)
    @staticmethod
    def plot_metric(files: List[str], metric: str = 'mae') -> None:
        
        df: pd.DataFrame = PlotService._get_metric_stats(files = files, metric = metric)
        PlotService._plot_error_bar(data = df, files = files, metric = metric)
        return

    # Plot loss
    @staticmethod
    def plot_loss(files: List[str]) -> None:
        
        dfs: List[pd.DataFrame] = [PlotService._get_loss_stats(_) for _ in files]
        PlotService._plot_loss(dfs = dfs, files = files)
        return

    ### Utils methods
    # Generate dataframe for metric
    @staticmethod
    def _get_metric_stats(files: List[str], metric: str = 'mae') -> pd.DataFrame:

        metric = metric.upper()
        assert metric in ['MAE', 'MSE', 'RMSE']
        
        output_data: Dict[str, any] = {}

        for file in files:
            with open(f'save/grouping/{file}.json') as f:
                _data: Dict = json.load(fp = f)

            # Compute average (mean)
            output_data[file] = {
                k: np.array(_data[metric][k]).mean().item()
                for k in ['load', 'pv', 'net']
            }

            # Get minimum value
            output_data[f'{file}_min'] = {
                k: min(_data[metric][k])
                for k in ['load', 'pv', 'net']
            }

            # Get maximum value
            output_data[f'{file}_max'] = {
                k: max(_data[metric][k])
                for k in ['load', 'pv', 'net']
            }
        
        return pd.DataFrame(output_data)
    
    # Plot error bar form DataFrame
    @staticmethod
    def _plot_error_bar(data: pd.DataFrame, files: List[str], metric: str = 'mae') -> None:

        # Get metric (for axe label)
        metric = metric.upper()
        assert metric in ['MAE', 'MSE', 'RMSE']
        metric_name: str = {
            'MAE': 'Mean Absolute Error',
            'MSE': 'Mean Square Error',
            'RMSE': 'Root Mean Square Error'
        }.get(metric)
        
        # Create bar plot
        ax = data[files].plot.bar(rot = 0)
        plt.xlabel('')
        plt.ylabel(f'{metric_name} ({metric})')
        ax.set_yscale('log')

        # Add error for each bar 
        for i, patch in enumerate(ax.patches):
            _min = data[f'{files[i//3]}_min'][f'{["load", "pv", "net"][i%3]}']
            _max = data[f'{files[i//3]}_max'][f'{["load", "pv", "net"][i%3]}']
            plt.vlines(patch.get_x() + patch.get_width() / 2, _min, _max, color = 'k')
        
        plt.title('Mean Absolute Error for model testing after attack')
        plt.show()
        return
    
    # Get stats for training loss
    @staticmethod
    def _get_loss_stats(file: str) -> pd.DataFrame:
        
        output_data: Dict[str, List[float]] = {
            'avg': [],
            'min': [],
            'max': []
        }

        with open(f'save/grouping/{file}.json') as f:
            _data: Dict = json.load(fp = f)

        # Transpose
        _loss_data: np.ndarray = np.array(_data['training_loss']).T

        # Compute average, minimum and maximum
        for _round in _loss_data:
            output_data['avg'].append(sum(_round)/len(_round))
            output_data['min'].append(min(_round))
            output_data['max'].append(max(_round))

        return pd.DataFrame(output_data)
    
    # Plot training loss
    @staticmethod
    def _plot_loss(dfs: List[pd.DataFrame], files: List[str]) -> None:
        
        assert len(dfs) == len(files)
        x_range: List[int] = list(range(1, len(dfs[0]['avg']) + 1))

        for df, file in zip(dfs, files):
            # Plot average line
            line = plt.plot(x_range, df['avg'], label = file)[0]

            # Error range with a lighter color
            color: str = line.get_color().replace('#', '')
            color = PlotService.lighten_color(color = color, amount = 100)
            plt.fill_between(x_range, df['min'], df['max'], color = color)

            # Show the range of the loss if behind other range, optionnal (can be removed for clarity)
            # plt.plot(x_range, df['min'], '--', color = color)
            # plt.plot(x_range, df['max'], '--', color = color)

        plt.title('Training loss for different attacks')
        plt.xlabel('Round')
        plt.ylabel('Mean Square Error (MSE) Loss')
        plt.yscale('log')
        plt.xticks(x_range)
        plt.legend()
        plt.show()
        return

    # Method for lightening a color
    @staticmethod
    def lighten_color(color: str, amount: int) -> str:
        color: int = int(color.replace('#', ''), 16)
        r: int = min((color >> 16) + amount, 255)
        g: int = min(((color >> 8) & 0x00FF ) + amount, 255)
        b: int = min((color & 0x0000FF ) + amount, 255)
        return hex((r << 16) | (g << 8) | b).replace('0x', '#')
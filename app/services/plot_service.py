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
        """
        Plot bar graph with error bar for MSE, MAE or RMSE metric for Load, PV and Net consumption.

        Args:
            files (List[str]): List of data files. Each file has to have the same data structure.
            metric (str = 'mae'): Metric that will be displayed.
        """
        df: pd.DataFrame = PlotService._get_metric_stats(files = files, metric = metric)
        PlotService._plot_error_bar(data = df, files = files, metric = metric)
        return

    # Plot trainig loss
    @staticmethod
    def plot_loss(files: List[str]) -> None:
        """
        Plot loss plot for different training simulations.

        Args:
            files (List[str]): List of files containint training loss data. Each file has to have the same data structure.
        """
        dfs: List[pd.DataFrame] = [PlotService._get_loss_stats(_) for _ in files]
        PlotService._plot_loss(dfs = dfs, files = files)
        return

    # Plot different sigma for scoring comparison
    @staticmethod
    def plot_sigma_measurment(files: List[str] = None) -> None:
        """
        Plot number of rejected clients for different sigma values (works only for distance and external validation scoring metrics).

        Args:
            files (List[str]): List of files containing data.
        """
        files: List[str] = files or ['distance', 'dataset']
        ticks: set = set([])

        # fig, axs = plt.subplots(2, 1)
        for file in files:
            
            with open(f'save/sigma_testing/{file}.json', mode = 'r', encoding = 'utf-8') as f:
                _data = json.load(fp = f)

            plt.plot(_data['parameters'], _data['rejected'], label = f'{file.capitalize()} scoring')
            ticks = ticks | set(_data['parameters'])

        # plt.hlines(.1, min(ticks), max(ticks), linestyles = '--', label = 'Threshold', color = '#aaaaaa')

        plt.xlabel('$\\sigma$')
        plt.ylabel('Rejected models')
        plt.title(f'Number of rejected models for different scoring methods')
        plt.grid(axis = 'y')
        plt.xscale('log')
        plt.xticks(sorted(ticks))
        plt.legend()

        plt.show()
        return
        
    # Plot effect of different decays for the sigma
    @staticmethod
    def plot_decay_measurment(files: List[str] = None) -> None:
        """
        Plot number of rejected clients for different decay factors. Works only for decayed scoring defenses.

        Args:
            files (List[str]): List of files containing data.
        """
        files: List[str] = files or ['log_distance', 'log_dataset', 'root_distance', 'root_dataset']
        ticks: set = set([])

        fig, axs = plt.subplots(2, 2)
        for file_idx, file in enumerate(files):
            
            with open(f'save/sigma_testing/decay_{file}.json', mode = 'r', encoding = 'utf-8') as f:
                _data = json.load(fp = f)

            ax = axs[file_idx%2, file_idx//2]

            for idx, param in enumerate(_data['parameters']):
                ax.plot(range(1, len(_data['rejected'][idx]) + 1), _data['rejected'][idx], label = f'$\\alpha = {param}$')
            ticks = ticks | set(_data['parameters'])
            ax.set_title(f'{file.split("_")[1]} scoring with {file.split("_")[0]} decay')
            ax.legend(loc = 0)
            ax.set_xticks(range(1, len(_data['rejected'][idx]) + 1))
            ax.set_ylabel('Number of rejected models')
            ax.set_xlabel('Round')

        fig.suptitle('Comparison of rejected models for distance and dataset scoring with different decays')
        plt.subplots_adjust(hspace = .4)
        plt.show()

    # Plot matrix display for defense conparison
    @staticmethod
    def plot_matrix_display(rows: List[str], columns: List[str], filename_format: str = '{col} {row}.json', metric: str = 'mae') -> None:
        """
        Plot a tile graph (like a confusion matrix) for error metric for differnet simulations.

        Args:
            rows (List[str]): List of rows of the matrix display.
            cols (List[str]): List of colulmns of the matrix display.
            filename_format (str = '{col} {row}.json'): File format used to get the error metric. Based on rows and columns.
            metric (str = 'mae'): Metric shown in the display.
        """
        matrix: pd.DataFrame = PlotService._get_matrix_data(rows, columns, filename_format, metric)
        PlotService._plot_matrix_as_tiles(df = matrix)

    ### Utils methods
    # Generate dataframe for metric
    @staticmethod
    def _get_metric_stats(files: List[str], metric: str = 'mae') -> pd.DataFrame:
        """
        Compute statistics about the error metrics used by the `plot_metric` method.

        Args:
            files (List[str]): List of files containing the error metric data.
            metric (str = 'mae'): Metric shown in the bar plot.

        Returns:
            pd.DataFrame: Dataframe containing average, min and max of the data.
        """

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
        """
        Plot bar plot from dataframe.

        Args:
            data (pd.DataFrame): Computed statistics used for the bar plot.
            files (List[str]): List of files used for the data.
            metric (str = 'mae'): Metric shown.
        """

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
        """
        Compute statistics about the training loss used by the `plot_loss` method.

        Args:
            files (List[str]): List of files containing the error metric data.

        Returns:
            pd.DataFrame: Dataframe containing average, min and max of the data.
        """
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
        """
        Plot losses from dataframe.

        Args:
            data (pd.DataFrame): Computed statistics used for the bar plot.
            files (List[str]): List of files used for the data.
        """
        assert len(dfs) == len(files)
        x_range: List[int] = list(range(1, len(dfs[0]['avg']) + 1))

        ax = plt.subplot(111)

        for df, file in zip(dfs, files):
            # Plot average line
            line = ax.plot(x_range, df['avg'], label = file.split('/')[-1])[0]

            # Error range with a lighter color
            color: str = line.get_color().replace('#', '')
            color = PlotService.lighten_color(color = color, amount = 100)
            ax.fill_between(x_range, df['min'], df['max'], color = color, alpha = .4)

        plt.title('Training loss for global model partial and total poisoning on 30 rounds')
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean Square Error (MSE) Loss')
        ax.set_yscale('log')
        ax.set_xticks(x_range)
        # ax.set_xlim(1, 20)
        ax.grid(axis = 'y')
        ax.legend()
        ax.spines[['top', 'right']].set_visible(False)
        plt.show()
        return

    # Method for lightening a color
    @staticmethod
    def lighten_color(color: str, amount: int) -> str:
        """
        Lighten a color and returns a new color in hex format.

        Args:
            color (str): Input color. Format must follow `#RRGGBB`.
            amount (int): Amoung of light added in the color. 

        Returns:
            str: The new color in a `#RRGGBB` format.
        """
        color: int = int(color.replace('#', ''), 16)
        r: int = min((color >> 16) + amount, 255)
        g: int = min(((color >> 8) & 0x00FF ) + amount, 255)
        b: int = min((color & 0x0000FF ) + amount, 255)
        return hex((r << 16) | (g << 8) | b).replace('0x', '#')
    
    # Generate dataframe for matrix display
    @staticmethod
    def _get_matrix_data(rows: List[str], columns: List[str], filename_format: str, metric: str) -> pd.DataFrame:
        """
        Compute statistics about the test error metric used by the `plot_matrix_display` method.

        Args:
            rows (List[str]): List of rows of the matrix display.
            cols (List[str]): List of colulmns of the matrix display.
            filename_format (str): File format used to get the error metric. Based on rows and columns.
            metric (str): Metric shown in the display.

        Returns:
            pd.DataFrame: Dataframe containing the data.
        """
        data: Dict[str, any] = {}

        for col in columns:
            data[col] = {}
            
            for row in rows:
                _filename: str = filename_format.format(row = row, col = col)
                with open(f'save/grouping/{_filename}', mode = 'r') as f:
                    _data = json.load(fp = f)

                data[col][row] = round(np.array(_data[metric.upper()]['net']).mean(), 2)

        return pd.DataFrame(data)

    # Plot dataframe as tiles
    @staticmethod
    def _plot_matrix_as_tiles(df: pd.DataFrame) -> None:
        """
        Plot the input dataframe as a tile plot.

        Args:
            df (pd.DataFrame): Input dataframe.
        """
        fig, ax = plt.subplots()
        im = ax.imshow(df, cmap="Blues")

        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns, rotation = 90)
        ax.set_yticklabels(df.index)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                ax.text(
                    j, i,
                    df.iloc[i, j],
                    ha="center",
                    va="center",
                    color="black"
                )

        plt.colorbar(im)

        plt.tight_layout()
        plt.show()
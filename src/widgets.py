import os
import logging

import pandas as pd
from IPython.display import display

logger = logging.getLogger()


def create_show_data(index_col, datafolder):
    """Closure to build data viewer for entire dataset."""
    data = None

    def show_data(file, index_col=index_col, datafolder=datafolder):
        filename = os.path.join(datafolder, file)
        nonlocal data  # only here to show-case data for report
        try:
            data = pd.read_csv(filename, index_col=index_col)
        except ValueError:
            data = pd.read_csv(filename)
            logger.warning(f'Could not find or use provided index_col: {index_col}')
        display(data.head())
    return show_data


def create_show_selected_proteins(data):
    """Closure to build data viewer for a selection of columns."""
    def show_selected_proteins(columns):
        nonlocal data  # only here to show-case data for report
        if len(columns) > 0:
            display(data[list(columns)])
            display(data[list(columns)].describe())
        else:
            print('Select proteins')
    return show_selected_proteins

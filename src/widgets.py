import os
import difflib
from typing import Iterable

import pandas as pd
import ipywidgets as widgets

def create_show_data(index_col, datafolder):
    """Closure to build data viewer for entire dataset."""
    data = None
    def show_data(file, index_col=index_col, datafolder=datafolder):
        filename = os.path.join(datafolder, file)
        nonlocal data # only here to show-case data for report
        try:
            data = pd.read_csv(filename, index_col=index_col)
        except:
            data = pd.read_csv(filename)
        display(data.head())
    return show_data

def create_show_selected_proteins(data):
    """Closure to build data viewer for a selection of columns."""
    def show_selected_proteins(columns):
        nonlocal data # only here to show-case data for report
        if len(columns)> 0:
            display(data[list(columns)])
            display(data[list(columns)].describe())
        else:
            print('Select proteins')
    return show_selected_proteins


def multi_checkbox_widget(descriptions:Iterable):
    """ Widget with a search field and lots of checkboxes 
    
    Parameters
    ----------
    descriptions: 
    """
    
    search_widget = widgets.Text()
    options_dict = {description: widgets.Checkbox(description=description, value=False) for description in descriptions}
    options = [options_dict[description] for description in descriptions]
    options_widget = widgets.VBox(options, layout= widgets.Layout(flex_flow='row wrap'))
    multi_select = widgets.VBox([search_widget, options_widget])

    # Wire the search field to the checkboxes
    def on_text_change(change):
        search_input = change['new']
        if search_input == '':
            # Reset search field
            new_options = options # [options_dict[description] for description in descriptions]
        else:
            # Filter by search field using difflib.
            close_matches = difflib.get_close_matches(search_input, descriptions, cutoff=0.0)
            new_options = [options_dict[description] for description in close_matches]
        options_widget.children = new_options

    search_widget.observe(on_text_change, names='value')
    return multi_select
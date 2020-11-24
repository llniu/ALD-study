from pathlib import Path

FOLDER_DATA = "datasets"
FOLDER_DATA = Path(FOLDER_DATA)

DATA_PL_MEDIAN = FOLDER_DATA / 'PLoverlap_median.csv'
DATA_LIVER_HEATMAP = FOLDER_DATA / 'liver_heatmap.csv'
DATA_PLCORR = FOLDER_DATA / 'PLoverlap_data.csv'
DATA_PLASMA_LONG = FOLDER_DATA / 'data_plasma_long.csv'

FOLDER_IMAGES = Path('images')
FNAME_IMAGE1 = Path('..') / 'figures' / 'Study overview.jpg'
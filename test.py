from IPython.display import IFrame
import matplotlib.pyplot as plt
import getpass
import xarray as xr
# import panel.widgets as pnw
# import panel as pn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copernicusmarine

# To avoid warning messages
import warnings
warnings.filterwarnings('ignore')

datasetID = 'cmems_mod_ibi_phy_anfc_0.027deg-3D_P1M-m'

DS = copernicusmarine.open_dataset(dataset_id = datasetID)
DS
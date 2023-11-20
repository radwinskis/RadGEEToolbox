[![PyPI version](https://badge.fury.io/py/RadGEEToolbox.svg)](https://pypi.org/project/RadGEEToolbox/)
# RadGEEToolbox 🛠️

### 🌎 Python package simplifying large-scale operations using Google Earth Engine (GEE) for users who utilize Landsat and Sentinel 

### [See documentation here](https://radgeetoolbox.readthedocs.io/en/latest/)

Initially created by Mark Radwin to help simplify processing imagery for PhD studies and general image exploration, this package offers helpful functionality with an outlook to add furthur functionality to aid assorted Earth observation specialists. 

The package is divided into four modules:
- LandsatCollection
   - Define and process Landsat 5 TM, 8 OLI, and 9 OLI surface reflectance imagery
- Sentinel2Collection
   - Define and process Sentinel 2 MSI surface reflectance imagery
- CollectionStitch
   - Accessory module to perform mosaicing functions on traditional GEE collections
- GetPalette
   - Retrieve color palettes compatible with visualiztion GEE layers
- VisParams
   - Alternative to visualization parameters dictionaries, define vis params using a function and retrieve palettes from GetPalette - makes visualizing images a bit easier


LandsatCollection.py and Sentinel2Collection.py are the main two modules for the majority of image processing. 

Almost all functionality is server-side friendly AND most results are cached, providing very fast processing times.

You can easily go back-and-forth from RadGEEToolbox and GEE objects to maximize efficiency in workflow.

### 🤔 Why use RadGEEToolbox?

If you are a new or expert user of GEE Landsat or Sentinel imagery, these tools greatly help reduce the amount of code needed for satellite image filtering, processing, analysis, and visualization. 

Although similar toolset packages exist, RadGEEToolbox offers differing functionality and additional functionality not found in other existing toolsets. 

### Features:
- Streamlined image collection definition and filtering
- Retrieve dates of images
- Mask clouds in image collection
- Mask water in image collection via two methods
- Mask to water in image collection via two methods
- Mask image collection inside geometry
- Mask image collection outside geometry
- Mosaic image collections which share dates of observations and copy all image properties from collection of choice
- Mosaic images that share the same date from a single image collection
- Easily choose image from collection for visualizing or processing using index or date string
- Calculate a variety of spectral index products: NDWI, NDVI, LST (celcius), NDTI (turbidity), relative chlorophyll-a concentrations, halite index (see Radwin & Bowen, 2021), and gypsum index (modified from Radwin & Bowen, 2021)
- Temporally reduce image collections using: minimum, maximum, median, or mean
- Calculate geodesically corrected surface area from pixels corresponding to values greater than a defined threshold from any singleband image
- Calculate geodesically corrected surface area from NDWI (water) pixels using dynamic thresholding via Otsu methods
- Easily call in a variety of useful color palettes for image visualization
- Easily define visualization parameter dictionaries
- AND more with a continual list of growing features

### ⌨️ Basic Usage
RadGEEToolbox is organized so that all functions are associated with each LandsatCollection and Sentinel2Collection class modules. You will need to define a base class collection, using arguments standard to defining ee.ImageCollections, and then you can call any of the class property attributes, methods, or static methods to complete processing. Utilizing class attribute propertues allows for very short code lines and very fast processing, while utilizing the methods allows for expanded customization of processing but requires more arguments and interaction from the user. The choice is up to you!

- #### See the /Example Notebooks folder for examples showing how to define and filter collections, process the rasters for multispectral or other spectral products, and easily access color palettes and visualization parameter dictionaries for image visualization. 

- #### See below for basic examples using the LandsatCollection class module. 

- #### See https://radgeetoolbox.readthedocs.io/en/latest/ for documentation.
```
# Create base class image collection
image_collection = LandsatCollection(start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

# retrieve latest image in collection as eeImage
latest_image = image_collection.image_grab(-1) 

# Get eeImageCollection from class collection
ee_image_collection = image_collection.collection 

# return cloud-masked LandsatCollection image collection
cloud_masked_collection = image_collection.masked_clouds_collection 

# return cloud-masked land surface temperature collection
LST_cloudless_collection = cloud_masked_collection.LST 

# return NDWI LandsatCollection image collection
NDWI_collection = image_collection.ndwi 

# Example showing how class functions work with any LandsatCollection image collection object, returning latest ndwi image
latest_NDWI_image = NDWI_collection.image_grab(-1) 

# See example notebooks for more details of usage
```



## 🚀 Installation Instructions

### 🔍 Prerequisites

- **Python**: Ensure you have version 3.6 or higher installed.
- **pip**: This is Python's package installer. 
- **conda-forge**: Conda channel installer (Coming soon...)

### 📦 Installing via pip

To install `RadGEEToolbox` version 1.4.3 using pip (NOTE: it is recommended to create a new virtual environment):

```bash
pip install RadGEEToolbox==1.4.3
```

### 📦 Installing via Conda (Coming soon...)

To install `RadGEEToolbox` version 1.4.3 using conda-forge (NOTE: it is recommended to create a new virtual environment):

```bash
conda install -c conda-forge RadGEEToolbox
```

### 🔧 Manual Installation from Source

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/radwinskis/RadGEEToolbox.git
   ```

2. **Navigate to Directory**: 
   ```bash
   cd RadGEEToolbox
   ```

3. **Install the Package**:
   ```bash
   pip install .
   ```

### ✅ Verifying the Installation

To verify that `RadGEEToolbox` was installed correctly:

```python
python -c "import RadGEEToolbox; print(RadGEEToolbox.__version__)"
```

You should see `1.4.3` printed as the version number.

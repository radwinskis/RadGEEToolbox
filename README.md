[![PyPI version](https://badge.fury.io/py/RadGEEToolbox.svg)](https://pypi.org/project/RadGEEToolbox/)
# RadGEEToolbox 🛠️

### 🌎 Python package simplifying large-scale operations using Google Earth Engine (GEE) for users who utilize Landsat and Sentinel 

### [See documentation here](https://radgeetoolbox.readthedocs.io/en/latest/)

Initially created by Mark Radwin to help simplify processing imagery for PhD studies and general image exploration, this package offers helpful functionality with an outlook to add furthur functionality to aid assorted Earth observation specialists. 

The package is divided into four modules:
- LandsatCollection
- Sentinel2Collection
- CollectionStitch
- GetPalette


where LandsatCollection.py and Sentinel2Collection.py are the main two modules for the majority of image processing. 

Almost all functionality is server-side friendly.

You can easily go back-and-forth from RadGEEToolbox and GEE objects to maximize efficiency in workflow.

### 🤔 Why use RadGEEToolbox?

If you are a new or expert user of GEE Landsat or Sentinel imagery, these tools should help reduce the amount of code needed for most situations by at least 2x. 

Although similar toolset packages exist, RadGEEToolbox offers differing functionality and additional functionality not found in other existing toolsets. 

### Features:
- Streamlined image collection definition and filtering
- Retrieve dates of images
- Mask clouds in image collection
- Mask water in image collection
- Mask image collection using geometry
- Mosaic image image collections which share dates of observations and copy all image properties from collection of choice
- Easily choose image from collection for visualizing or processing using index or date string
- Calculate a variety of spectral index products: NDWI, NDVI, LST (celcius), halite index (see Radwin & Bowen, 2021), and gypsum index (modified from Radwin & Bowen, 2021)
- Temporally reduce image collections using: minimum, maximum, median, or mean
- Calculate geodesically corrected surface area from pixels corresponding to values greater than a defined threshold from any singleband image
- Calculate geodesically corrected surface area from NDWI (water) pixels using dynamic thresholding via Otsu methods
- Easily call in a variety of useful color palettes for image visualization

### ⌨️ Basic Usage
RadGEEToolbox is organized so that all functions are associated with each LandsatCollection and Sentinel2Collection class modules. You will need to define a base class collection, using arguments standard to defining eeImageCollections, and then you can call any of the class attributes, methods, or static methods to complete processing. Utilizing class attributes allows for very short code lines and very fast processing, while utilizing the methods allows for expanded customization of processing but requires more arguments and interaction from the user. See below for basic examples using the LandsatCollection class module. See https://radgeetoolbox.readthedocs.io/en/latest/ for documentation.
```
#Create base class image collection
image_collection = LandsatCollection(start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

#retrieve latest image in collection as eeImage
latest_image = image_collection.image_grab(-1) 

#Get eeImageCollection from class collection
ee_image_collection = image_collection.collection 

#return cloud-masked LandsatCollection image collection
cloud_masked_collection = image_collection.masked_clouds_collection 

#return cloud-masked land surface temperature collection
LST_cloudless_collection = cloud_masked_collection.LST 

#return NDWI LandsatCollection image collection
NDWI_collection = image_collection.ndwi 

#Example showing how class functions work with any LandsatCollection image collection object, returning latest ndwi image
latest_NDWI_image = NDWI_collection.image_grab(-1) 
```



## 🚀 Installation Instructions

### 🔍 Prerequisites

- **Python**: Ensure you have version 3.6 or higher installed.
- **pip**: This is Python's package installer. 

### 📦 Installing via pip

To install `RadGEEToolbox` version 1.3 using pip (NOTE: it is recommended to create a new virtual environment):

```bash
pip install RadGEEToolbox==1.3
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

You should see `1.3` printed as the version number.

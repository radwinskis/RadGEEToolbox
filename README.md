[![PyPI version](https://badge.fury.io/py/RadGEEToolbox.svg)](https://pypi.org/project/RadGEEToolbox/)

# RadGEEToolbox 🛠

### 🌎 Streamlined Multispectral & SAR Analysis for Google Earth Engine Python API

### [See documentation here](https://radgeetoolbox.readthedocs.io/en/latest/)

`RadGEEToolbox` is an open-source Python package that facilitates the processing and analysis of both multispectral and Synthetic Aperture Radar (SAR) satellite imagery using `Google Earth Engine (GEE)`, including Landsat TM & OLI (courtesy of U.S. Geological Survey), Sentinel-1 SAR (courtesy of European Space Agency), and Sentinel-2 MSI (courtesy of European Space Agency) datasets. By providing an intuitive, modular framework, it simplifies complex remote sensing workflows, enabling researchers to efficiently manage, process, and analyze satellite imagery across a range of environmental and geospatial applications. The package is designed for time series analyses of large-scale remote sensing datasets, reducing the amount of custom scripting required in `GEE` while ensuring reproducibility and efficiency. Researchers working in landscape change, environmental monitoring, natural hazards, and land cover classification can benefit from `RadGEEToolbox`’s ability to streamline workflows from data acquisition to final analysis.

The package includes specialized tools for both multispectral and SAR data processing, offering capabilities such as image subsetting, mosaic generation, cloud/water masking, spectral index calculations, speckle filtering, multilooking, backscatter transformations, transect extractions, spatial statistics, and more. One of the advantages of `RadGEEToolbox` is its ability to seamlessly integrate user-defined `GEE` functions or operations, allowing researchers to modify and extend their analyses without being restricted to predefined operations offered by this package. `RadGEEToolbox` also includes preconfigured visualization parameters, facilitating intuitive exploration of geospatial datasets.

Originally developed by Mark Radwin to streamline imagery processing for PhD research and general image analysis, this package provides robust functionality with plans for future enhancements to support various Earth observation professionals. 

______

### Package structure

The package is divided into six modules:

- `LandsatCollection`
   - Filter and process Landsat 5 TM, 8 OLI/TIRS, and 9 OLI/TIRS surface reflectance imagery
- `Sentinel1Collection`
   - Filter and process Sentinel-1 Synthetic Aperture Radar (SAR) GRB backscatter imagery
- `Sentinel2Collection`
   - Filter and process Sentinel-2 MSI surface reflectance imagery
- `CollectionStitch`
   - Accessory module to perform mosaicing functions on traditional GEE collections
- `GetPalette`
   - Retrieve color palettes compatible with visualiztion GEE layers
- `VisParams`
   - Alternative to visualization parameters dictionaries, define vis params using a function and retrieve palettes from GetPalette - makes visualizing images a bit easier


`LandsatCollection.py`, `Sentinel1Collection.py`, and `Sentinel2Collection.py` are the main modules for the majority of image processing. 

`CollectionStitch.py`, `GetPalette.py`, and `VisParams.py` are supplemental for additional processing and image display.

Almost all functionality is server-side friendly AND most results are cached, providing faster processing times.

You can easily convert back-and-forth from RadGEEToolbox and GEE objects to maximize efficiency in workflow and implement custom image processing.

______

### 🌎 Why use RadGEEToolbox?

If you are a new or expert user of GEE and/or Landsat or Sentinel imagery, these tools greatly help reduce the amount of code needed for satellite image filtering, processing, analysis, and visualization. 

Although similar toolset packages exist, RadGEEToolbox offers differing functionality and additional functionality not found in other existing toolsets. 

The most commonly used remote sensing data management and image processing workflows are built-in to RadGEEToolbox, so you can more easily get to your goal: an informative dataset, stored in a friendly format. 

`RadGEEToolbox` is about **quality-of-life** for everday Google Earth Engine users. Every function is very useful for everday tasks not otherwise readily achievable using the GEE API alone. 

_________

### 📜 Feature List:
#### Data Management
- Streamlined **image collection definition and filtering**
- Retrieve **dates** of images
- Mask image collection inside geometry/polygon
- Mask image collection outside geometry/polygon
- **Mosaic image collections** which share dates of observations and copy all image properties from collection of choice
- **Mosaic images that share the same date** from a single image collection
- **Easily select an image from an image collection** for visualizing or furthur processing using a positional index or date string
- Easily call in a variety of useful **color palettes** for image visualization
- Easily define **visualization** parameter dictionaries

#### Multispectral Image Processing
- **Mask clouds** in image collection
- **Mask water** in image collection via two methods
- Mask to water in image collection via two methods
- Calculate a **variety of spectral index products**: NDWI, NDVI, LST (celcius), NDTI (turbidity), relative chlorophyll-a concentrations, halite index (see Radwin & Bowen, 2021), and gypsum index (modified from Radwin & Bowen, 2021)

#### SAR Image Processing
- **Easy to define/filter** the type of Sentinel-1 data to use (instrument mode, polarization, pixel size, orbit direction, etc)
- **Multilooking**
- **Speckle filtering** (Lee-Sigma)
- Convert **dB to sigma naught, or sigma naught to dB**

#### Spatial / Zonal Statistic Extraction (Time Series Analysis)
- Calculate **geodesically corrected surface area** from pixels corresponding to values greater than a defined threshold from any singleband image
- Calculate geodesically corrected surface area from NDWI (water) pixels using **dynamic thresholding** via Otsu methods
- **Extract singleband pixel values along a transect** (line) or multiple transects for every image in a collection, with options to save the data to a .csv file (output data organized with image dates)
- **Extract regionally reduced statistics** (mean, median, etc.) within a circular buffer for one or more coordinates for every image in a collection, with options to change the buffer size, save the data to a .csv file, and more (output data organized with image dates)

#### Temporal Reductions
- **Temporally reduce** image collections using: minimum, maximum, median, or mean

_____________

# Basic Usage ⌨
Each module is composed of a single class with: **attributes, property attributes, methods, and static method functions**. A comprehensive list of functions for each module can be found at the [RadGEEToolbox documentation page](https://radgeetoolbox.readthedocs.io/en/latest/index.html). The initial class object creates an image collection fitting the parameters of interest based on the required arguments for the initial class object (start date, end date, location, cloud percentage threshold, etc.). 


*For example, the following defines an initial LandsatCollection object from the `LandsatCollection` module.*

```
ImageCollection = RadGEEToolbox.LandsatCollection.LandsatCollection(
            start_date='2015-01-01', end_date='2025-01-01',
            cloud_percentage_threshold=25, boundary=GEE_geometry_of_study_area)
``` 


Once a class object is defined, **usage of the property attributes and methods provide full functionality for image processing** ([be sure to explore the documentation](https://radgeetoolbox.readthedocs.io/en/latest/index.html)). Class attributes store values and cache results, and should not be directly called aside from the `collection` attribute (see the *Converting Between RadGEEToolbox and GEE objects* section). Static methods are base functions used when the property attributes or method functions are called, and are thus ran in the background but can be imported to be performed on an Earth Engine image object outside of the RadGEEToolbox class-based workflow.

*For example, `dates = ImageCollection.dates` defines a variable by using the `dates` property attribute which contains a list of dates for the defined image collection. Similarly, `mean_image = ImageCollection.mean` defines a temporal reduction of the image collection using the `mean` property attribute to calculate the mean image from a collection of images.*

For functionality which requires an input argument from the user, class methods must be used. 

*For example, the following uses the class method `mask_to_polygon()` to mask out pixels outside of a region of interest (referencing `ImageCollection` defined above).* 

```
masked_collection = ImageCollection.mask_to_polygon(
                     polygon=GEE_geometry_of_study_area)
``` 

The majority of the class property attributes and class methods are shared between the `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` modules, however, some functionality differs between the multispectral and SAR specialized modules.

#### See the /Example Notebooks folder for examples showing how to define and filter collections, process the rasters for multispectral or other spectral products, and easily access color palettes and visualization parameter dictionaries for image visualization. 
______________________

### Basic examples using the LandsatCollection class module:


```
# Create base class image collection
image_collection = RadGEEToolbox.LandsatCollection.LandsatCollection(
   start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

# retrieve latest image in collection as eeImage
latest_image = image_collection.image_grab(-1) 

# Convert to ee.ImageCollection for custom processing
ee_image_collection = image_collection.collection 

# return cloud-masked LandsatCollection image collection
cloud_masked_collection = image_collection.masked_clouds_collection 

# return cloud-masked land surface temperature collection
LST_cloudless_collection = cloud_masked_collection.LST 

# return NDWI LandsatCollection image collection
NDWI_collection = image_collection.ndwi 

# Example showing how method functions work with any LandsatCollection image collection object, returning latest ndwi image
latest_NDWI_image = NDWI_collection.image_grab(-1) 

# See example notebooks for more details of usage 
# and documentation for comprehensive list of available tools
```

__________

## 🚀 Installation Instructions

### 🔍 Prerequisites

- **Python**: Ensure you have version 3.6 or higher installed.
- **pip**: This is Python's package installer. 
- **conda-forge**: Community led Conda package installer channel

### 📦 Installing via pip

To install `RadGEEToolbox` version 1.6.0 using pip (NOTE: it is recommended to create a new virtual environment):

```bash
pip install RadGEEToolbox==1.6.0
```

### 📦 Installing via Conda

To install `RadGEEToolbox` version 1.6.0 using conda-forge (NOTE: it is recommended to create a new virtual environment):

```bash
conda install conda-forge::radgeetoolbox
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

You should see `1.6.0` printed as the version number.

________

### Interested in Contributing?

We welcome contributions to **RadGEEToolbox**! Whether you're looking to **fix bugs, add new features, improve documentation, or optimize existing workflows**, your contributions can help enhance the package for the geospatial research community.

### How to Contribute  
1. **Check the Issues Tab** – Browse the [GitHub Issues](https://github.com/radwinskis/RadGEEToolbox/issues) for open tasks, feature requests, or bug reports. Feel free to suggest new features or improvements!  
2. **Fork the Repository** – Clone the project and create a new branch for your contributions.  
3. **Follow Coding Guidelines** – Maintain consistency with existing code structure and ensure your changes are well-documented.  
4. **Submit a Pull Request** – Once your changes are tested and complete, open a pull request explaining your updates.  

### Ways to Contribute  
- 🛠️ **Code Contributions** – Add new functionalities, improve performance, or refactor existing code.  
- 📖 **Documentation Improvements** – Enhance tutorials, clarify explanations, or add examples to the [documentation](https://radgeetoolbox.readthedocs.io/en/latest/).  
- 🐛 **Bug Reports** – If you encounter any issues, submit a detailed report to help us diagnose and fix them.  
- 🌍 **Feature Requests** – Have an idea for a new feature? Open an issue to discuss its implementation!  

### 💬 Get in Touch  
If you have questions or want to discuss potential contributions, feel free to start a discussion in the **Issues** section or reach out via [GitHub Discussions](https://github.com/radwinskis/RadGEEToolbox/discussions).

Thank you for your interest in making **RadGEEToolbox** better! 🚀


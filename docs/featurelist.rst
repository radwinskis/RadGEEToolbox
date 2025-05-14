Why use RadGEEToolbox?
======================

If you are a new or expert user of GEE and/or Landsat or Sentinel imagery, these tools greatly help reduce the amount of code needed for satellite image filtering, processing, analysis, and visualization. 

Although similar toolset packages exist, RadGEEToolbox offers differing functionality and additional functionality not found in other existing toolsets. 

The most commonly used remote sensing data management and image processing workflows are built-in to RadGEEToolbox, so you can more easily get to your goal: an informative dataset, stored in a friendly format. 

RadGEEToolbox is about **quality-of-life** for everday Google Earth Engine users. Every function is very useful for everday tasks not otherwise readily achievable using the GEE API alone. 

Feature List
============

Data Management
---------------

- Streamlined **image collection definition and filtering**
- Retrieve **dates** of images
- Mask image collection inside a geometry/polygon
- Mask image collection outside a geometry/polygon
- **Mosaic image collections** that share dates of observation and copy all image properties from the collection of choice
- **Mosaic images that share the same date** from a single image collection
- **Select an image from an image collection** using a positional index or date string
- Access a variety of useful **color palettes** for image visualization
- Define **visualization** parameter dictionaries easily

Multispectral Image Processing
------------------------------

- **Mask clouds** in image collections
- **Mask water** in image collections via two methods
- Mask to water in image collections via two methods
- Calculate a variety of **spectral index products**: NDWI, NDVI, LST (Celsius), NDTI (turbidity), relative chlorophyll-a concentrations, halite index (see Radwin & Bowen, 2021), and gypsum index (modified from Radwin & Bowen, 2021)

SAR Image Processing
--------------------

- **Easily define/filter** the type of Sentinel-1 data (instrument mode, polarization, pixel size, orbit direction, etc.)
- **Multilooking**
- **Speckle filtering** (Lee-Sigma)
- Convert between **dB and sigma naught**

Spatial / Zonal Statistic Extraction (Time Series Analysis)
-----------------------------------------------------------

- Calculate **geodesically corrected surface area** from pixels above a defined threshold from any singleband image
- Calculate geodesically corrected surface area from NDWI (water) pixels using **dynamic thresholding** (Otsu method)
- **Extract singleband pixel values along a transect** (or multiple transects) for every image in a collection, with options to save to CSV (data organized by image date)
- **Extract regionally reduced statistics** (mean, median, etc.) within a circular buffer for one or more coordinates for every image in a collection, with options to change buffer size, save as CSV, and more (data organized by image date)

Temporal Reductions
-------------------

- **Temporally reduce** image collections using minimum, maximum, median, or mean operations
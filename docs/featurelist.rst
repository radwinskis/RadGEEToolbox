Why use RadGEEToolbox?
======================

If you are a new or expert user of GEE and/or Landsat or Sentinel imagery, these tools greatly help reduce the amount of code needed for satellite image filtering, processing, analysis, and visualization. 

Although similar toolset packages exist, RadGEEToolbox offers differing functionality and additional functionality not found in other existing toolsets (Table 1). 

The most commonly used remote sensing data management and image processing workflows are built-in to RadGEEToolbox, so you can more easily get to your goal: an informative dataset, stored in a friendly format. 

RadGEEToolbox is about **quality-of-life** and **comprehensive features** for everyday Google Earth Engine users. Each function is very useful for common tasks not otherwise readily achievable using the GEE API alone. 

As of version `1.7.1`, `RadGEEToolbox` supports any generic image collection via the `GenericCollection` module which allows for utilization of the same data management, temporal reduction, zonal statistics, and data export tools available for the `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` modules. This allows users to provide their own image collection of choice, such as PRISM or MODIS data, to benefit from the tools available with `RadGEEToolbox`.


**Table 1.** Comparison of functionality between RadGEEToolbox, eemont, and geetools packages.

+----------------------------------------------------+-------------------+------------+--------------+
| Capability                                         | **RadGEEToolbox** | **eemont** | **geetools** |
+====================================================+===================+============+==============+
| **Dataset & Workflow Specific API's**              | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Synthetic Aperture Radar (S1) Support**          | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Zonal Time-series Extraction**                   | **YES**           | **YES**    | **YES**      |
+----------------------------------------------------+-------------------+------------+--------------+
| **Area Time-series Extraction**                    | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Transect Time-series Extraction**                | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Comprehensive Preprocessing Operations**         | **YES**           | **YES**    | **YES**      |
+----------------------------------------------------+-------------------+------------+--------------+
| **Reflectance Scaling**                            | **YES**           | **YES**    | **YES**      |
+----------------------------------------------------+-------------------+------------+--------------+
| **Land Surface Temperature Calculation (Landsat)** | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Narrowband to Broadband Albedo Calculation**     | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Built-in Spectral Index Calculations**           | **YES**           | **YES**    | **YES**      |
+----------------------------------------------------+-------------------+------------+--------------+
| **Anomaly (Deviation from Mean) Calculations**     | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Image Masking for Classified Images**            | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Merging of Multiple Collections**                | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Image Selection by Date or Index**               | **YES**           | **YES**    | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Visualization Presets/Tools**                    | **YES**           | NO         | NO           |
+----------------------------------------------------+-------------------+------------+--------------+
| **Batch Export to GEE Asset**                      | **YES**           | **YES**    | **YES**      |
+----------------------------------------------------+-------------------+------------+--------------+

Statement of Need
=================

Working in `Google Earth Engine` often involves steep learning curves and complex scripting. `RadGEEToolbox` simplifies this by automating common multispectral and SAR workflows while remaining fully compatible with native GEE workflows. It's designed for students, researchers, and analysts working with Landsat and Sentinel imagery, enabling efficient time series analysis, surface property extraction, and visualization of environmental indicators. Ultimately, RadGEEToolbox makes it easier to perform multi-temporal analyses, extract surface properties or time series, generate land cover statistics, and visualize key environmental indicators by handling much of the complexity inherent in Google Earth Engine API interactions and image processing tasks.

While several Python packages extend Google Earth Engine capabilities, including geemap for interactive mapping, eemont for simplified spectral index calculations, and geetools for broad utility functions, RadGEEToolbox takes a different approach. Rather than providing a general-purpose set of method extensions to Earth Engine objects, RadGEEToolbox offers a modular, workflow-oriented API centered around user-defined collection classes in addition to a set of Earth Engine method extensions. These objects bundle common operations, such as filtering, cloud masking, mosaicking, spectral index calculation, and zonal statistics into streamlined, reusable workflows with minimal boilerplate code. This design emphasizes rapid implementation of end-to-end pipelines for scientific studies, particularly for hydrology, geomorphology, forestry, and time series landscape analysis. In this way, RadGEEToolbox complements existing packages while providing a focused solution for efficient, reproducible remote sensing workflows.


Feature List
============

Data Management
---------------

- Streamlined **image collection definition and filtering**
- Retrieve **dates** of images
- Mask image collection inside a geometry/polygon
- Mask image collection outside a geometry/polygon
- Mask images based on pixel values
- Convert singleband images to **binary masks** (classified images) based on pixel values
- Rename bands in singleband or multiband image collections
- **Mosaic image collections** that share dates of observation and copy all image properties from the collection of choice
- **Mosaic images that share the same date** from a single image collection
- **Select an image from an image collection** using a positional index or date string
- Scale Landsat and Sentinel-2 DN pixel values to **reflectance** (surface reflectance)
- Merge multiple singleband or multiband collections into a single collection
- Access a variety of useful **color palettes** for image visualization
- Define **visualization** parameter dictionaries easily
- Automatically **batch export image collections to GEE assets**

Multispectral Image Processing
------------------------------

- **Mask clouds and/or cloud shadows** in image collections
- **Mask water** in image collections via two methods
- **Mask to water** in image collections via two methods
- **Calculate the anomalies** (deviation from mean) for each image in an image collection
- Calculate any of the following **spectral index products**: 

    - Normalized Difference Water Index (NDWI)
    - Modified Normalized Difference Water Index (MNDWI)
    - Normalized Difference Vegetation Index (NDVI)
    - Enhanced Vegetation Index (EVI)
    - Soil Adjusted Vegetation Index (SAVI)
    - Modified Soil Adjusted Vegetation Index (MSAVI)
    - Normalized Difference Moisture Index (NDMI)
    - Normalized Difference Turbidity Index (NDTI)
    - Chlorophyll-a Index (different for Landsat vs Sentinel-2)
    - Normalized Burn Ratio (NBR)
    - Normalized Difference Snow Index (NDSI)
    - Land Surface Temperature (LST) in Celsius (Landsat only)
    - Halite Index (Radwin & Bowen, 2021)
    - Gypsum Index (modified from Radwin & Bowen, 2021)
    - Broadband Albedo

- Binary mask images based on a **threshold** of a spectral index, providing methods for retaining pixel values of classified pixels or setting them to a pixel value of 1.

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
- **Extract regionally reduced statistics** (mean, median, etc.) with one or more geometries for every image in a collection, with options to save as CSV or as DataFrame 
- Easily export aggregated image properties or spatiotemporal statistics to pandas DataFrame or CSV

Temporal Reductions
-------------------

- **Temporally reduce** image collections using minimum, maximum, median, or mean operations
- Create **monthly median composites** from an image collection, with image count metadata
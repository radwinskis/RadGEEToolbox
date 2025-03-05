---
title: 'RadGEEToolbox: Streamlined Multispectral & SAR Analysis for Google Earth Engine'
tags:
  - Python
  - remote sensing
  - geospatial
  - landscape evolution
  - Google Earth Engine
authors:
  - name: Mark Radwin
    orcid: 0000-0002-7236-1425
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Dept. of Geology & Geophysics, University of Utah, United States
   index: 1
date: 3 March 2025
bibliography: paper.bib
---

# Summary

`RadGEEToolbox` is an open-source Python package that facilitates the processing and analysis of both multispectral and Synthetic Aperture Radar (SAR) satellite imagery using `Google Earth Engine` [@Gorelick:2017], including Landsat TM & OLI (courtesy of U.S. Geological Survey), Sentinel-1 SAR (courtesy of European Space Agency), and Sentinel-2 MSI (courtesy of European Space Agency) datasets. By providing an intuitive, modular framework, it simplifies complex remote sensing workflows, enabling researchers to efficiently manage, process, and analyze satellite imagery across a range of environmental and geospatial applications. The package is designed for time series analyses of large-scale remote sensing datasets, reducing the amount of custom scripting required in `Google Earth Engine` while ensuring reproducibility and efficiency. Researchers working in landscape change, environmental monitoring, natural hazards, and land cover classification can benefit from `RadGEEToolbox`’s ability to streamline workflows from data acquisition to final analysis.

The package includes specialized tools for both multispectral and SAR data processing, offering capabilities such as image subsetting, mosaic generation, cloud/water masking, spectral index calculations, speckle filtering, multilooking, backscatter transformations, transect extractions, spatial statistics, and more. One of the advantages of `RadGEEToolbox` is its ability to seamlessly integrate user-defined `Google Earth Engine` functions or operations, allowing researchers to modify and extend their analyses without being restricted to predefined operations offered by this package. `RadGEEToolbox` also includes preconfigured visualization parameters, facilitating intuitive exploration of geospatial datasets.


# Statement of need

Processing and analyzing remote sensing data in `Google Earth Engine` requires significant programming expertise, often leading to steep learning curves for new users and complex, bloated code. While `Google Earth Engine` provides a powerful API for accessing and manipulating Earth observation datasets, performing even routine tasks such as filtering image collections, computing spectral indices, or transforming SAR backscatter values often requires extensive scripting and re-use of the same functions or code across multiple projects. `RadGEEToolbox` addresses these issues by streamlining and automating many of these repetitive and complex tasks while still allowing advanced users the flexibility to customize their analyses. `RadGEEToolbox` is designed for students, researchers, remote sensing analysts, and environmental scientists at any level of experience who are working with Landsat, Sentinel-1, or Sentinel-2 datasets. Ultimately, `RadGEEToolbox` makes it easier to perform multi-temporal analyses, extract surface properties or time series, generate land cover statistics, and visualize key environmental indicators by handling much of the complexity inherent in `Google Earth Engine` API interactions and image processing tasks.

# Functionality Overview
## Structure and General Usage
 The package is divided into six modules:

`LandsatCollection`, `Sentinel1Collection`, `Sentinel2Collection`, `CollectionStitch`, `GetPalette`, and `VisParams`.

Each module is composed of a single class with: attributes, property attributes, methods, and static methods. A comprehensive list of functions for each module can be found at the [RadGEEToolbox documentation page](https://radgeetoolbox.readthedocs.io/en/latest/index.html) or [GitHub page](https://github.com/radwinskis/RadGEEToolbox). The initial class object creates an image collection fitting the parameters of interest based on the required arguments for the initial class object (start date, end date, location, cloud percentage threshold, etc.). 


*For example, the following defines an initial LandsatCollection object from the `LandsatCollection` module:*

```
ImageCollection = RadGEEToolbox.LandsatCollection.LandsatCollection(
            start_date='2015-01-01', end_date='2025-01-01',
            cloud_percentage_threshold=25, boundary=GEE_geometry_of_study_area)
``` 


Once a class object is defined, usage of the property attributes and methods provide full functionality for image processing. Class attributes store values and cache results, and should not be directly called aside from the `collection` attribute (see the *Converting Between RadGEEToolbox and GEE objects* section). Static methods are base functions used when the property attributes or method functions are called, and are thus ran in the background but can be imported to be performed on an Earth Engine image object outside of the `RadGEEToolbox` class-based workflow.

*For example, `dates = ImageCollection.dates` defines a variable by using the `dates` property attribute which contains a list of dates for the defined image collection. Similarly, `mean_image = ImageCollection.mean` defines a temporal reduction of the image collection using the `mean` property attribute to calculate the mean image from a collection of images.*

For functionality which requires an input argument from the user, class methods must be used. 

*For example, the following uses the class method `mask_to_polygon()` to mask out pixels outside of a region of interest (referencing `ImageCollection` defined above):* 

```
masked_collection = ImageCollection.mask_to_polygon(
                     polygon=GEE_geometry_of_study_area)
``` 


The majority of the class property attributes and class methods are shared between the `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` modules, however, some functionality differs between the multispectral and SAR specialized modules. The functionality shared between modules is mainly for standard data management practices, such as creating raster mosaics and masking imagery via a polygon. 

## Multispectral Specific Functionality

The Landsat and Sentinel-2 datasets hosted on `Google Earth Engine` include image metadata useful for filtering and identifying clouds and water. `RadGEEToolbox` allows users to filter image collections by cloud percentage, to automatically avoid cloudy images, as well as the ability to automatically mask out any cloud or water pixels identified by the advanced algorithms utilized for the image metadata. Alternatively, users can mask to water pixels using an identical approach, or use the built-in NDWI calculations to mask out or to water.

The `LandsatCollection` and `Sentinel2Collection` modules offer functionality to quickly and easily process the multispectral rasters to predefined spectral indices for chlorophyll-a relative concentrations (3BDA for Landsat and 2BDA for Sentinel-2), water detection and water quality (Normalized Difference Water Index - NDWI), water turbidity detection (Normalized Difference Turbidity Index - NDTI), Land Surface Temperature (LST - for Landsat 8 & 9 data only), vegetation detection and vegetation health (Normalized Difference Vegetation Index - NDVI), halite detection (Halite Index), and gypsum detection (Gypsum Index). The user can generate classification rasters by applying a pixel value threshold for binary masking of a selected spectral index, with the option to set separate thresholds for Landsat TM and OLI sensors to account for differences in spectral sensitivity when working with long time series.

For chlorophyll-a indices, the 3BDA index uses the BLUE, GREEN, and RED bands (see @Boucher:2018 for more details) while the 2BDA index uses the RED-EDGE1 and RED bands (see @Buma&Lee:2020). For water indices, the NDWI uses the GREEN and NIR bands (see @McFeeters:1996) and the NDTI uses the RED and GREEN bands (see @Lacaux:2007). LST is calculated from Landsat TIRS thermal band data by utilizing the corrections and workflow provided by @Sekertekin&Bonafoni:2020. For evaporite indices, the Halite Index uses the RED and SWIR1 bands (see @Radwin&Bowen:2021) and the Gypsum Index uses the SWIR1 and SWIR2 bands (see @Radwin&Bowen:2024). More built-in indices will be added in future updates, extending the offered functionality for the spectral targeting of surface materials and material properties. 


## SAR Specific Functionality

The `Sentinel1Collection` module offers functionality for filtering the Sentinel-1 SAR data to imagery of interest and for performing standard SAR operations. When defining a `Sentinel1Collection` object, `RadGEEToolbox` allows for filtering the SAR data based on start date, end date, relative orbit number start, relative orbit number stop, instrument mode, polarization, bands, orbit direction, and product resolution. This greatly simplifies the process for acquiring the correct SAR imagery for a research task while providing significant control. For example, a user can easily filter to multiple tiles from a specific ascending swath for a specific polarization, then quickly mosaic all images that share the same acquisition date to result in the desired collection of mosaiced SAR rasters. 

The class property attributes for the `Sentinel1Collection` module provide functionality for SAR operations such as converting the backscatter signal from dB to Sigma Naught (σ~o~) or σ~o~ to dB, multilooking, and speckle filtering (Lee Sigma method). Other class methods of `Sentinel1Collection` are a subset of those also found in the `LandsatCollection` and `Sentinel2Collection` modules.

## Data Management, Spatial Statistics, and Temporal Reductions

Many researchers study landscapes which span areas greater than which can be observed by a single satellite image. In these cases, a mosaic may need to be produced to encapsulate an entire study area. `RadGEEToolbox` offers the `MosaicByDate` property attribute for each `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` module, which mosaics images acquired on the same date for an entire image collection.

One of the most critically informative aspects of remote sensing is the usage of time series datasets to understand how landscapes change over time, and `RadGEEToolbox` offers multiple ways to temporally explore the filtered and processed datasets available for the `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` modules. Surface area of any class or pixel values above/below a threshold value can be calculated and stored using the `PixelAreaSumCollection()` class method, which can easily be used to create a dataframe of the values of interest over time. Time series of zonal statistics can also be extracted using the `iterate_zonal_stats()` method, which stores extracted values from multiple coordinates either as a dataframe or saves the data locally as a .csv file. Users can also retrieve a time series of values along one or more transects for each image in a collection using the `transect_iterator()` class method. These tools are fundamental for extracting valuable information from a series of satellite images.

In addition to extracting values for time series analyses, multiple temporal reduction methods are offered to reduce a series of image to a single image. For each `LandsatCollection`, `Sentinel1Collection`, and `Sentinel2Collection` module, the property attributes `max`, `median`, `mean`, and `min` are available to temporally reduce any RadGEEToolbox image collection with more than one image.

## Converting Between RadGEEToolbox and Google Earth Engine objects

To be flexible and modular, `RadGEEToolbox` allows the user to easily convert any `RadGEEToolbox` image collection object to a `Google Earth Engine` image collection (*ee.ImageCollection*) object via the class attribute `collection`. This allows a user to perform initial image filtering and processing with `RadGEEToolbox` but then perform custom operations using user defined code for working directly with `Google Earth Engine` objects. 

*For example, `GEE_collection = ImageCollection.collection` converts the `RadGEEToolbox` object to an `ee` object.*

Likewise, any Landsat, Sentinel-1, or Sentinel-2 `ee.ImageCollection` object can be converted to a `RadGEEToolbox` object using the corresponding module and providing the `ee.ImageCollection` object as the class object argument of `collection`. This is an important feature, as a custom spectral index can be calculated for an `ee.ImageCollection` then converted to a `RadGEEToolbox` image collection to perform zonal statistics calculations using `iterate_zonal_stats()`, for example, allowing significant flexibility. 

*For example, `Imported_Collection = RadGEEToolbox.LandsatCollection.LandsatCollection(collection=GEE_collection)` converts the previously converted `ee.ImageCollection` object to a `LandsatCollection` object and full functionality is retained while honoring the operations performed on the `ee.ImageCollection` object.*

## Image Visualization

The `GetPalette` module provides easy to access color palettes useful for visualizing singleband rasters and indices, many of which are colorblind friendly. Additionally, the `VisParams` module provides pre-defined visualization parameter dictionaries for each of the products and sensors incorporated into the `RadGEEToolbox` package, to be used for visualizations using Leaflet or Folium (such as the [geemap](https://geemap.org/) package).

# Usefulness for Geospatial & Geoscience Communities

There are an array of existing packages which supplement the utility of `Google Earth Engine` and the `Google Earth Engine` API, however, `RadGEEToolbox` was created out of a need for additional tools and enhanced functionality to quickly compute standard multispectral and SAR remote sensing workflows. The functionality offered by `RadGEEToolbox` extends beyond what is available in other `Google Earth Engine` oriented Python packages and is mindful of client-side requests, avoiding calls like `.getInfo()` wherever possible in addition to caching results. `RadGEEToolbox` is a valuable tool for researchers across geospatial and geoscience disciplines, including those studying landscape evolution, environmental change, natural hazards, resource management, and hydrological processes. By simplifying the processing of multispectral and SAR imagery, the package enhances workflows for applications such as land-cover classification, water resource monitoring, vegetation and soil analysis, forestry, coastal and marine studies, agriculture, and geohazard assessment. Its ability to integrate automated image processing, spectral indices, temporal analysis, and spatial statistics within `Google Earth Engine` makes it a powerful asset for large-scale research on erosion, flooding, drought, land use, surface deformation, and ecosystem dynamics.

Thus, `RadGEEToolbox` offers enhanced functionality and efficiency for those who frequently utilize `Google Earth Engine` and specifically process Landsat, Sentinel-2, or Sentinel-1 data, all while resulting in cleaner and more minimal code. Finally, `RadGEEToolbox` lowers the technical barrier for advanced remote sensing analysis in `Google Earth Engine`, making it easier for students and researchers to get started or advance their work with remote sensing data.


# References
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
date: 14 May 2025
bibliography: paper.bib
---

# Summary

`RadGEEToolbox` is an open-source Python package for processing and analyzing multispectral and Synthetic Aperture Radar (SAR) imagery using `Google Earth Engine` [@Gorelick:2017]. It supports Landsat TM/OLI, Sentinel-1 SAR, and Sentinel-2 MSI datasets, with plans to support additional platforms in the future (MODIS, VIIRS, etc). `RadGEEToolbox` provides a modular, streamlined workflow for landscape change detection, environmental monitoring, and land classification. It reduces the need for repetitive code while promoting reproducibility and efficiency.

Features include subsetting, mosaicking, cloud/water masking, spectral index calculations, speckle filtering, backscatter conversions, spatial statistics, and integration of custom Earth Engine functions.

# Statement of Need

Working in `Google Earth Engine` often involves steep learning curves and complex scripting. `RadGEEToolbox` simplifies this by automating common multispectral and SAR workflows while remaining fully compatible with native GEE workflows. It's designed for students, researchers, and analysts working with Landsat and Sentinel imagery, enabling efficient time series analysis, surface property extraction, and visualization of environmental indicators. Ultimately, RadGEEToolbox makes it easier to perform multi-temporal analyses, extract surface properties or time series, generate land cover statistics, and visualize key environmental indicators by handling much of the complexity inherent in Google Earth Engine API interactions and image processing tasks.

While several Python packages extend Google Earth Engine capabilities, including geemap for interactive mapping [@Wu:2020], eemont for simplified spectral index calculations [@Montero:2021], and geetools [@geetools] for broad utility functions, RadGEEToolbox takes a different approach. Rather than providing a general-purpose set of method extensions to Earth Engine objects, RadGEEToolbox offers a modular, workflow-oriented API centered around user-defined collection classes in addition to a set of Earth Engine method extensions (Table 1). These objects bundle common operations, such as filtering, cloud masking, mosaicking, spectral index calculation, and zonal statistics into streamlined, reusable workflows with minimal boilerplate code. If desired, RadGEEToolbox can be used in conjunction with geemap, eemont, or geetools. 

This design emphasizes rapid implementation of end-to-end pipelines for scientific studies, particularly for hydrology, geomorphology, forestry, and time series landscape analysis. In this way, RadGEEToolbox complements existing packages while providing a focused solution for efficient, reproducible remote sensing workflows. 


**Table 1.** Comparison of functionality between RadGEEToolbox, eemont, and geetools.

| Capability | **RadGEEToolbox** | **eemont** | **geetools** |
|:--------------:|:---:|:---:|:---:|
| **Dataset & Workflow Specific API's** | **YES** | NO | NO |
| **Synthetic Aperture Radar (S1) Support** | **YES** | NO | NO |
| **Zonal Time-series Extraction** | **YES** | **YES** | **YES** |
| **Area Time-series Extraction** | **YES** | NO | NO |
| **Transect Time-series Extraction** | **YES** | NO | NO |
| **Comprehensive Preprocessing Operations** | **YES** | **YES** | **YES** |
| **Reflectance Scaling (DN to ρ)** | **YES** | **YES** | **YES** |
| **Land Surface Temperature Calculation (Landsat)** | **YES** | NO | NO |
| **Image Selection by Date or Index** | **YES** | **YES** | NO |
| **Visualization Presets/Tools** | **YES** | NO | NO |


# Functionality Overview

## Modules and Structure

The toolbox comprises six modular classes: `LandsatCollection`, `Sentinel1Collection`, `Sentinel2Collection`, `CollectionStitch`, `GetPalette`, and `VisParams`. Each of the first three classes creates a filtered `ee.ImageCollection` based on user-defined parameters and provides attributes and methods for processing, with shared capabilities for mosaicking, masking, and zonal statistics calculations. A comprehensive list of functions for each module can be found at the [RadGEEToolbox documentation page](https://radgeetoolbox.readthedocs.io/en/latest/index.html) or [GitHub page](https://github.com/radwinskis/RadGEEToolbox).

## Multispectral Tools

The `LandsatCollection` and `Sentinel2Collection` modules support reflectance scaling, cloud/water filtering, and provide indices for vegetation (NDVI), water (NDWI [@McFeeters:1996] and MNDWI), aqueous turbidity (NDTI) [@Lacaux:2007]), aqueous chlorophyll-a (3BDA [@Boucher:2018], 2BDA [@Buma&Lee:2020]), Land Surface Temperature (LST) [@Sekertekin&Bonafoni:2020], halite [@Radwin&Bowen:2021], and gypsum [@Radwin&Bowen:2024]. Users can generate classification rasters by choosing a pixel value threshold for binary masking of a selected spectral index, with the option to designate separate thresholds for Landsat TM and OLI sensors to account for differences in spectral sensitivity when working with long time series.

## SAR Tools

The `Sentinel1Collection` module offers functionality for filtering the Sentinel-1 SAR data to imagery of interest and for performing standard SAR operations. `Sentinel1Collection` supports filtering by orbit, date, polarization, and resolution. It includes SAR-specific tools for multilooking, speckle filtering (Lee Sigma), and dB/σ⁰ conversions. It also allows efficient mosaicking of swaths by acquisition date.

## Data Management

Many researchers study landscapes which span areas greater than which can be observed by a single satellite image. In these cases, a mosaic may need to be produced to encapsulate an entire study area. RadGEEToolbox offers the MosaicByDate property attribute for each LandsatCollection, Sentinel1Collection, and Sentinel2Collection module, which mosaics images acquired on the same date for an entire image collection

## Time Series & Statistics

One of the most critically informative aspects of remote sensing is the usage of time series datasets to understand how landscapes change over time. All collection modules support temporal reductions (mean, median, min, max) and offer methods for time series and zonal analysis. Key tools include:

- `PixelAreaSumCollection()` – surface area time series of classified pixels
- `iterate_zonal_stats()` – zonal stats time series from coordinates
- `transect_iterator()` – extract time series of values along transect(s)

## Interoperability

Users can convert between `RadGEEToolbox` and `ee.ImageCollection` objects using the `.collection` attribute or class constructor. This enables hybrid workflows that mix `RadGEEToolbox` capabilities with custom Earth Engine scripts. This is an important feature, as a custom spectral index can be calculated for an `ee.ImageCollection` then converted to a RadGEEToolbox image collection to perform zonal statistics calculations using `iterate_zonal_stats()`, for example, allowing significant flexibility. 

# Usefulness for Geospatial & Geoscience Communities

`RadGEEToolbox` extends the capabilities of GEE by enabling fast, reproducible processing of Landsat and Sentinel data while avoiding performance-heavy client-side calls like `.getInfo()` wherever possible in addition to caching results. It supports applications in land-use monitoring, water resources, vegetation health, agriculture, and hazard mapping. Its balance of automation and flexibility makes it ideal for both novice and advanced users seeking to streamline their Earth Engine workflows.

# References
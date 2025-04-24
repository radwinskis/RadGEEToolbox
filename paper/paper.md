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
date: 24 April 2025
bibliography: paper.bib
---

# Summary

`RadGEEToolbox` is an open-source Python package for processing and analyzing multispectral and Synthetic Aperture Radar (SAR) imagery using `Google Earth Engine` [@Gorelick:2017]. It supports Landsat TM/OLI, Sentinel-1 SAR, and Sentinel-2 MSI datasets, providing a modular, streamlined workflow for landscape change detection, environmental monitoring, and land classification. It reduces the need for repetitive code while promoting reproducibility and efficiency.

Features include subsetting, mosaicking, cloud/water masking, spectral index calculations, speckle filtering, backscatter conversions, spatial statistics, and integration of custom Earth Engine functions.

# Statement of Need

Working in `Google Earth Engine` often involves steep learning curves and complex scripting. `RadGEEToolbox` simplifies this by automating common multispectral and SAR workflows while remaining fully compatible with native GEE workflows. It's designed for students, researchers, and analysts working with Landsat and Sentinel imagery, enabling efficient time series analysis, surface property extraction, and visualization of environmental indicators. Ultimately, RadGEEToolbox makes it easier to perform multi-temporal analyses, extract surface properties or time series, generate land cover statistics, and visualize key environmental indicators by handling much of the complexity inherent in Google Earth Engine API interactions and image processing tasks.

# Functionality Overview

## Modules and Structure

The toolbox comprises six modular classes: `LandsatCollection`, `Sentinel1Collection`, `Sentinel2Collection`, `CollectionStitch`, `GetPalette`, and `VisParams`. Each of the first three classes creates a filtered `ee.ImageCollection` based on user-defined parameters and provides attributes and methods for processing, with shared capabilities for mosaicking, masking, and zonal statistics calculations. A comprehensive list of functions for each module can be found at the [RadGEEToolbox documentation page](https://radgeetoolbox.readthedocs.io/en/latest/index.html) or [GitHub page](https://github.com/radwinskis/RadGEEToolbox).

## Multispectral Tools

The `LandsatCollection` and `Sentinel2Collection` modules support cloud/water filtering and provide indices for vegetation (NDVI), water (NDWI [@McFeeters:1996], NDTI [@Lacaux:2007]), turbidity, chlorophyll-a (3BDA [@Boucher:2018], 2BDA [@Buma&Lee:2020]), LST [@Sekertekin&Bonafoni:2020], halite [@Radwin&Bowen:2021], and gypsum [@Radwin&Bowen:2024]. Users can generate classification rasters by applying a pixel value threshold for binary masking of a selected spectral index, with the option to set separate thresholds for Landsat TM and OLI sensors to account for differences in spectral sensitivity when working with long time series.

## SAR Tools

The `Sentinel1Collection` module offers functionality for filtering the Sentinel-1 SAR data to imagery of interest and for performing standard SAR operations. `Sentinel1Collection` supports filtering by orbit, date, polarization, and resolution. It includes SAR-specific tools for multilooking, speckle filtering (Lee Sigma), and dB/σ⁰ conversions. It also allows efficient mosaicking of swaths by acquisition date.

## Data Management

Many researchers study landscapes which span areas greater than which can be observed by a single satellite image. In these cases, a mosaic may need to be produced to encapsulate an entire study area. RadGEEToolbox offers the MosaicByDate property attribute for each LandsatCollection, Sentinel1Collection, and Sentinel2Collection module, which mosaics images acquired on the same date for an entire image collection

## Time Series & Statistics

One of the most critically informative aspects of remote sensing is the usage of time series datasets to understand how landscapes change over time. All collection modules support temporal reductions (mean, median, min, max) and offer methods for time series and zonal analysis. Key tools include:

- `PixelAreaSumCollection()` – pixel area stats over time
- `iterate_zonal_stats()` – zonal stats from coordinates
- `transect_iterator()` – extract values along transects

## Interoperability

Users can convert between `RadGEEToolbox` and `ee.ImageCollection` objects using the `.collection` attribute or class constructor. This enables hybrid workflows that mix `RadGEEToolbox` capabilities with custom Earth Engine scripts. This is an important feature, as a custom spectral index can be calculated for an `ee.ImageCollection` then converted to a RadGEEToolbox image collection to perform zonal statistics calculations using `iterate_zonal_stats()`, for example, allowing significant flexibility. 

# Usefulness for Geospatial & Geoscience Communities

`RadGEEToolbox` extends the capabilities of GEE by enabling fast, reproducible processing of Landsat and Sentinel data while avoiding performance-heavy client-side calls like `.getInfo()` whererever possible in addition to caching results. It supports applications in land-use monitoring, water resources, vegetation health, agriculture, and hazard mapping. Its balance of automation and flexibility makes it ideal for both novice and advanced users seeking to streamline their Earth Engine workflows.

# References
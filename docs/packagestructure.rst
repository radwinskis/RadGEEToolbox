Package Structure
=================

The package is divided into six modules:

- ``LandsatCollection``  
  Filter and process Landsat 5 TM, 8 OLI/TIRS, and 9 OLI/TIRS surface reflectance imagery.

- ``Sentinel1Collection``  
  Filter and process Sentinel-1 Synthetic Aperture Radar (SAR) Ground Range Detected (GRD) backscatter imagery.

- ``Sentinel2Collection``  
  Filter and process Sentinel-2 MSI surface reflectance imagery.

- ``GenericCollection``  
  Filter and process any ee.ImageCollection.

- ``CollectionStitch``  
  Accessory module to perform mosaicking functions on traditional Earth Engine collections.

- ``GetPalette``  
  Retrieve color palettes compatible with visualizing GEE layers.

- ``VisParams``  
  Alternative to hard-coded visualization parameter dictionaries. Define vis params using a function and retrieve palettes from ``GetPalette`` to simplify image styling.

- ``Export``  
  Export processed collections or images to Google Drive

The main processing modules are ``LandsatCollection.py``, ``Sentinel1Collection.py``, ``Sentinel2Collection.py``, and ``GenericCollection``. These handle the majority of remote sensing workflows.

The supplemental modules — ``CollectionStitch.py``,  ``GetPalette.py``, ``VisParams.py``, and ``Export.py`` — support advanced processing and image display customization.

Most functionality is server-side friendly, and many results are cached to improve performance and reduce recomputation time.

Users can easily convert between RadGEEToolbox objects and native Earth Engine objects to maximize workflow efficiency and enable custom processing pipelines.

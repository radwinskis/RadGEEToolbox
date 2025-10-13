import ee
import pandas as pd
import numpy as np


# ---- Reflectance scaling for Landsat Collection 2 SR ----
_LS_SR_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
_LS_SCALE = 0.0000275
_LS_OFFSET = -0.2

def _scale_landsat_sr(img):
    """
    Converts Landsat C2 SR DN values to reflectance values for SR_B1..SR_B7 (overwrite bands).

    Args:
        img (ee.Image): Input Landsat image without scaled bands.

    Returns:
        ee.Image: Image with scaled reflectance bands.
    """
    img = ee.Image(img)
    is_scaled = ee.Algorithms.IsEqual(img.get('rgt:scaled'), 'landsat_sr')
    scaled = img.select(_LS_SR_BANDS).multiply(_LS_SCALE).add(_LS_OFFSET)
    out = img.addBands(scaled, None, True).set('rgt:scaled', 'landsat_sr')
    return ee.Image(ee.Algorithms.If(is_scaled, img, out))

class LandsatCollection:
    """
    Represents a user-defined collection of NASA/USGS Landsat 5, 8, and 9 TM & OLI surface reflectance satellite images at 30 m/px from Google Earth Engine (GEE).

    This class enables simplified definition, filtering, masking, and processing of multispectral Landsat imagery.
    It supports multiple spatial and temporal filters, caching for efficient computation, and direct computation of
    key spectral indices like NDWI, NDVI, halite index, and more. It also includes utilities for cloud masking,
    mosaicking, zonal statistics, and transect analysis.

    Initialization can be done by providing filtering parameters or directly passing in a pre-filtered GEE collection.

    Inspect the documentation or source code for details on the methods and properties available.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. Required unless `collection` is provided.
        end_date (str): End date in 'YYYY-MM-DD' format. Required unless `collection` is provided.
        tile_row (int or list of int): WRS-2 tile row(s) to filter by. Ignored if `boundary` or `collection` is provided. See https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files
        tile_path (int or list of int): WRS-2 tile row(s) to filter by. Ignored if `boundary` or `collection` is provided. See https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files
        cloud_percentage_threshold (int, optional): Max allowed cloud cover percentage. Defaults to 100.
        boundary (ee.Geometry, optional): A geometry for filtering to images that intersect with the boundary shape. Overrides `tile_path` and `tile_row` if provided.
        collection (ee.ImageCollection, optional): A pre-filtered Landsat ee.ImageCollection object to be converted to a LandsatCollection object. Overrides all other filters.
        scale_bands (bool, optional): If True, all SR bands will be scaled from DN values to reflectance values. Defaults to False.

    Attributes:
        collection (ee.ImageCollection): The filtered or user-supplied image collection converted to an ee.ImageCollection object.

    Raises:
        ValueError: Raised if required filter parameters are missing, or if both `collection` and other filters are provided.

    Note:
        See full usage examples in the documentation or notebooks:
        https://github.com/radwinskis/RadGEEToolbox/tree/main/Example%20Notebooks

    Examples:
        >>> from RadGEEToolbox import LandsatCollection
        >>> import ee
        >>> ee.Initialize()
        >>> image_collection = LandsatCollection(
        ...     start_date='2023-06-01',
        ...     end_date='2023-06-30',
        ...     tile_row=32,
        ...     tile_path=38,
        ...     cloud_percentage_threshold=20
        ... )
        >>> cloud_masked = image_collection.masked_clouds_collection
        >>> latest_image = cloud_masked.image_grab(-1)
        >>> ndwi_collection = image_collection.ndwi
    """

    def __init__(
        self,
        start_date=None,
        end_date=None,
        tile_row=None,
        tile_path=None,
        cloud_percentage_threshold=None,
        boundary=None,
        collection=None,
        scale_bands=False,
    ):
        if collection is None and (start_date is None or end_date is None):
            raise ValueError(
                "Either provide all required fields (start_date, end_date, tile_row, tile_path ; or boundary in place of tiles) or provide a collection."
            )
        if (
            tile_row is None
            and tile_path is None
            and boundary is None
            and collection is None
        ):
            raise ValueError(
                "Provide either tile or boundary/geometry specifications to filter the image collection"
            )
        if collection is None:
            self.start_date = start_date
            self.end_date = end_date
            self.tile_row = tile_row
            self.tile_path = tile_path
            self.boundary = boundary

            if cloud_percentage_threshold is None:
                cloud_percentage_threshold = 100
                self.cloud_percentage_threshold = cloud_percentage_threshold
            else:
                self.cloud_percentage_threshold = cloud_percentage_threshold

            if isinstance(tile_row, list):
                pass
            else:
                self.tile_row = [tile_row]

            if isinstance(tile_path, list):
                pass
            else:
                self.tile_path = [tile_path]

            # Filter the collection
            if tile_row and tile_path is not None:
                self.collection = self.get_filtered_collection()
            elif boundary is not None:
                self.collection = self.get_boundary_filtered_collection()
            elif boundary and tile_row and tile_path is not None:
                self.collection = self.get_boundary_filtered_collection()
        else:
            self.collection = collection
        if scale_bands:
            self.collection = self.collection.map(_scale_landsat_sr)

        self._dates_list = None
        self._dates = None
        self.ndwi_threshold = -1
        self.mndwi_threshold = -1
        self.ndvi_threshold = -1
        self.halite_threshold = -1
        self.gypsum_threshold = -1
        self.turbidity_threshold = -1
        self.chlorophyll_threshold = -0.5
        self.ndsi_threshold = -1
        self.evi_threshold = -1
        self.savi_threshold = -1.5
        self.msavi_threshold = -1
        self.ndmi_threshold = -1
        self.nbr_threshold = -1
        self._masked_clouds_collection = None
        self._masked_water_collection = None
        self._masked_to_water_collection = None
        self._geometry_masked_collection = None
        self._geometry_masked_out_collection = None
        self._median = None
        self._mean = None
        self._max = None
        self._min = None
        self._albedo = None
        self._ndwi = None
        self._mndwi = None
        self._ndvi = None
        self._ndsi = None
        self._evi = None
        self._savi = None
        self._msavi = None
        self._ndmi = None
        self._nbr = None
        self._halite = None
        self._gypsum = None
        self._turbidity = None
        self._chlorophyll = None
        self._LST = None
        self._MosaicByDate = None
        self._PixelAreaSumCollection = None
        self._Reflectance = None

    @staticmethod
    def image_dater(image):
        """
        Adds date to image properties as 'Date_Filter'.

        Args:
            image (ee.Image): Input image

        Returns:
            ee.Image: Image with date in properties.
        """
        date = ee.Number(image.date().format("YYYY-MM-dd"))
        return image.set({"Date_Filter": date})

    @staticmethod
    def landsat5bandrename(img):
        """
        Renames Landsat 5 bands to match Landsat 8 & 9.

        Args:
            image (ee.Image): input image

        Returns:
            ee.Image: image with renamed bands
        """
        return img.select(
            "SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"
        ).rename("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL")

    @staticmethod
    def landsat_ndwi_fn(image, threshold, ng_threshold=None):
        """
        Calculates ndwi from GREEN and NIR bands (McFeeters, 1996 - https://doi.org/10.1080/01431169608948714) for Landsat imagery and mask image based on threshold.

        Can specify separate thresholds for Landsat 5 vs 8 & 9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8 & 9.

        Args:
            image (ee.Image): input image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: ndwi image
        """
        ndwi_calc = image.normalizedDifference(
            ["SR_B3", "SR_B5"]
        )  # green-NIR / green+NIR -- full NDWI image
        water = (
            ndwi_calc.updateMask(ndwi_calc.gte(threshold))
            .rename("ndwi")
            .copyProperties(image)
        )
        if ng_threshold != None:
            water = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                ndwi_calc.updateMask(ndwi_calc.gte(threshold))
                .rename("ndwi")
                .copyProperties(image)
                .set("threshold", threshold),
                ndwi_calc.updateMask(ndwi_calc.gte(ng_threshold))
                .rename("ndwi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            water = (
                ndwi_calc.updateMask(ndwi_calc.gte(threshold))
                .rename("ndwi")
                .copyProperties(image)
            )
        return water

    @staticmethod
    def landsat_mndwi_fn(image, threshold, ng_threshold=None):
        """
        Calculates Modified Normalized Difference Water Index (MNDWI) from GREEN and SWIR bands for Landsat imagery and mask image based on threshold.

        Can specify separate thresholds for Landsat 5 vs 8 & 9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8 & 9.

        Args:
            image (ee.Image): input image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: ndwi image
        """
        mndwi_calc = image.normalizedDifference(
            ["SR_B3", "SR_B6"]
        )  # green-SWIR / green+SWIR -- full NDWI image
        water = (
            mndwi_calc.updateMask(mndwi_calc.gte(threshold))
            .rename("ndwi")
            .copyProperties(image)
        )
        if ng_threshold != None:
            water = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                mndwi_calc.updateMask(mndwi_calc.gte(threshold))
                .rename("ndwi")
                .copyProperties(image)
                .set("threshold", threshold),
                mndwi_calc.updateMask(mndwi_calc.gte(ng_threshold))
                .rename("ndwi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            water = (
                mndwi_calc.updateMask(mndwi_calc.gte(threshold))
                .rename("ndwi")
                .copyProperties(image)
            )
        return water

    @staticmethod
    def landsat_ndvi_fn(image, threshold, ng_threshold=None):
        """
        Calculates ndvi from NIR and RED bands (Huang et al., 2020 - https://link.springer.com/10.1007/s11676-020-01155-1) for Landsat imagery and mask image based on threshold.

        Can specify separate thresholds for Landsat 5 vs 8 & 9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: ndvi ee.Image
        """
        ndvi_calc = image.normalizedDifference(
            ["SR_B5", "SR_B4"]
        )  # NIR-RED/NIR+RED -- full NDVI image
        if ng_threshold != None:
            vegetation = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                ndvi_calc.updateMask(ndvi_calc.gte(threshold))
                .rename("ndvi")
                .copyProperties(image)
                .set("threshold", threshold),
                ndvi_calc.updateMask(ndvi_calc.gte(ng_threshold))
                .rename("ndvi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            vegetation = (
                ndvi_calc.updateMask(ndvi_calc.gte(threshold))
                .rename("ndvi")
                .copyProperties(image)
            )
        return vegetation

    @staticmethod
    def landsat_halite_fn(image, threshold, ng_threshold=None):
        """
        Calculates multispectral halite index from RED and SWIR1 bands (Radwin & Bowen, 2021 - https://onlinelibrary.wiley.com/doi/10.1002/esp.5089) for Landsat imagery and mask image based on threshold.

        Can specify separate thresholds for Landsat 5 vs 8 & 9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8 & 9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: halite ee.Image
        """
        halite_index = image.normalizedDifference(["SR_B4", "SR_B6"])
        if ng_threshold != None:
            halite = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                halite_index.updateMask(halite_index.gte(threshold))
                .rename("halite")
                .copyProperties(image)
                .set("threshold", threshold),
                halite_index.updateMask(halite_index.gte(ng_threshold))
                .rename("halite")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            halite = (
                halite_index.updateMask(halite_index.gte(threshold))
                .rename("halite")
                .copyProperties(image)
            )
        return halite

    @staticmethod
    def landsat_gypsum_fn(image, threshold, ng_threshold=None):
        """
        Calculates multispectral gypsum index from SWIR1 and SWIR2 bands(Radwin & Bowen, 2024 - https://onlinelibrary.wiley.com/doi/10.1002/esp.5089) for Landsat imagery and mask image based on threshold.

        Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: gypsum ee.Image
        """
        gypsum_index = image.normalizedDifference(["SR_B6", "SR_B7"])
        if ng_threshold != None:
            gypsum = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                gypsum_index.updateMask(gypsum_index.gte(threshold))
                .rename("gypsum")
                .copyProperties(image)
                .set("threshold", threshold),
                gypsum_index.updateMask(gypsum_index.gte(ng_threshold))
                .rename("gypsum")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            gypsum = (
                gypsum_index.updateMask(gypsum_index.gte(threshold))
                .rename("gypsum")
                .copyProperties(image)
            )
        return gypsum

    @staticmethod
    def landsat_ndti_fn(image, threshold, ng_threshold=None):
        """
        Calculates turbidity of water pixels using Normalized Difference Turbidity Index (NDTI; Lacaux et al., 2007 - https://doi.org/10.1016/j.rse.2006.07.012)
        and mask image based on threshold. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: turbidity ee.Image
        """
        NDTI = image.normalizedDifference(["SR_B4", "SR_B3"])
        if ng_threshold != None:
            turbidity = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                NDTI.updateMask(NDTI.gte(threshold))
                .rename("ndti")
                .copyProperties(image)
                .set("threshold", threshold),
                NDTI.updateMask(NDTI.gte(ng_threshold))
                .rename("ndti")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            turbidity = (
                NDTI.updateMask(NDTI.gte(threshold))
                .rename("ndti")
                .copyProperties(image)
            )
        return turbidity

    @staticmethod
    def landsat_kivu_chla_fn(image, threshold, ng_threshold=None):
        """
        Calculates relative chlorophyll-a concentrations of water pixels using 3BDA/KIVU index
        (see Boucher et al., 2018 for review - https://esajournals.onlinelibrary.wiley.com/doi/10.1002/eap.1708) and mask image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold
        argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            ee.Image: chlorophyll-a ee.Image
        """
        KIVU = image.expression(
            "(BLUE - RED) / GREEN",
            {
                "BLUE": image.select("SR_B2"),
                "RED": image.select("SR_B4"),
                "GREEN": image.select("SR_B3"),
            },
        )
        if ng_threshold != None:
            chlorophyll = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                KIVU.updateMask(KIVU.gte(threshold))
                .rename("kivu")
                .copyProperties(image)
                .set("threshold", threshold),
                KIVU.updateMask(KIVU.gte(ng_threshold))
                .rename("kivu")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            chlorophyll = (
                KIVU.updateMask(KIVU.gte(threshold))
                .rename("kivu")
                .copyProperties(image)
            )
        return chlorophyll

    @staticmethod
    def landsat_broadband_albedo_fn(image):
        """
        Calculates broadband albedo for Landsat TM and OLI images, based on Liang, 2001 
        (https://doi.org/10.1016/S0034-4257(00)00205-4) and Wang et al., 2016 (https://doi.org/10.1016/j.rse.2016.02.059).

        Args:
            image (ee.Image): input ee.Image

        Returns:
            ee.Image: broadband albedo ee.Image
        """
        # Conversion using Liang, 2001 as reference
        TM_expression = '0.356*b("SR_B2") + 0.130*b("SR_B4") + 0.373*b("SR_B5") + 0.085*b("SR_B6") + 0.072*b("SR_B8") - 0.0018'
        # Conversion using Wang et al., 2016 as reference
        OLI_expression = '0.2453*b("SR_B2") + 0.0508*b("SR_B3") + 0.1804*b("SR_B4") + 0.3081*b("SR_B5") + 0.1332*b("SR_B6") + 0.0521*b("SR_B7") + 0.0011'
        # If spacecraft is Landsat 5 TM, use the correct expression,
        # otherwise treat as OLI and copy properties after renaming band to "albedo"
        albedo = ee.Algorithms.If(
            ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
            image.expression(TM_expression).rename("albedo").copyProperties(image),
            image.expression(OLI_expression).rename("albedo").copyProperties(image))
        return albedo

    @staticmethod
    def landsat_ndsi_fn(image, threshold, ng_threshold=None):
        """
        Calculates the Normalized Difference Snow Index (NDSI) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where NDSI pixels greater than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where NDSI pixels greater than threshold are masked.

        Returns:
            ee.Image: NDSI ee.Image
        """
        ndsi_calc = image.normalizedDifference(["SR_B3", "SR_B6"])
        if ng_threshold != None:
            ndsi = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                ndsi_calc.updateMask(ndsi_calc.gte(threshold))
                .rename("ndsi")
                .copyProperties(image)
                .set("threshold", threshold),
                ndsi_calc.updateMask(ndsi_calc.gte(ng_threshold))
                .rename("ndsi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            ndsi = ndsi_calc.updateMask(ndsi_calc.gte(threshold)).rename("ndsi").copyProperties(image).set("threshold", threshold)
        return ndsi

    @staticmethod
    def landsat_evi_fn(image, threshold, ng_threshold=None, gain_factor=2.5, l=1, c1=6, c2=7.5):
        """
        Calculates the Enhanced Vegetation Index (EVI) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.
        See https://www.usgs.gov/landsat-missions/landsat-enhanced-vegetation-index

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where EVI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where EVI pixels less than threshold are masked.
            gain_factor (float, optional): Gain factor, typically set to 2.5. Defaults to 2.5.
            l (float, optional): Canopy background adjustment factor, typically set to 1. Defaults to 1.
            c1 (float, optional): Coefficient for the aerosol resistance term, typically set to 6. Defaults to 6.
            c2 (float, optional): Coefficient for the aerosol resistance term, typically set to 7.5. Defaults to 7.5.

        Returns:
            ee.Image: EVI ee.Image
        """
        evi_expression = f'{gain_factor} * ((b("SR_B5") - b("SR_B4")) / (b("SR_B5") + {c1} * b("SR_B4") - {c2} * b("SR_B2") + {l}))'
        evi_calc = image.expression(evi_expression)
        if ng_threshold != None:
            evi = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                evi_calc.updateMask(evi_calc.gte(threshold))
                .rename("evi")
                .copyProperties(image)
                .set("threshold", threshold),
                evi_calc.updateMask(evi_calc.gte(ng_threshold))
                .rename("evi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            evi = evi_calc.updateMask(evi_calc.gte(threshold)).rename("evi").copyProperties(image).set("threshold", threshold)
        return evi
    
    @staticmethod
    def landsat_savi_fn(image, threshold, ng_threshold=None, l=0.5):
        """
        Calculates the Soil-Adjusted Vegetation Index (SAVI) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.
        See Huete, 1988 - https://doi.org/10.1016/0034-4257(88)90106-X

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where SAVI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where SAVI pixels less than threshold are masked.
            l (float, optional): Soil brightness correction factor, typically set to 0.5. Defaults to 0.5.
        Returns:
            ee.Image: SAVI ee.Image
        """
        savi_expression = f'((b("SR_B5") - b("SR_B4")) / (b("SR_B5") + b("SR_B4") + {l})) * (1 + {l})'
        savi_calc = image.expression(savi_expression)
        if ng_threshold != None:
            savi = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                savi_calc.updateMask(savi_calc.gte(threshold))
                .rename("savi")
                .copyProperties(image)
                .set("threshold", threshold),
                savi_calc.updateMask(savi_calc.gte(ng_threshold))
                .rename("savi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            savi = savi_calc.updateMask(savi_calc.gte(threshold)).rename("savi").copyProperties(image).set("threshold", threshold)
        return savi
    
    @staticmethod
    def landsat_msavi_fn(image, threshold, ng_threshold=None):
        """
        Calculates the Modified Soil-Adjusted Vegetation Index (MSAVI) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.
        See Qi et al., 1994 - https://doi.org/10.1016/0034-4257(94)90134-1 and https://www.usgs.gov/landsat-missions/landsat-modified-soil-adjusted-vegetation-index

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where MSAVI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where MSAVI pixels less than threshold are masked.
        
        Returns:
            ee.Image: MSAVI ee.Image
        """
        msavi_expression = '0.5 * (2 * b("SR_B5") + 1 - ((2 * b("SR_B5") + 1) ** 2 - 8 * (b("SR_B5") - b("SR_B4"))) ** 0.5)'
        msavi_calc = image.expression(msavi_expression)
        if ng_threshold != None:
            msavi = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                msavi_calc.updateMask(msavi_calc.gte(threshold))
                .rename("msavi")
                .copyProperties(image)
                .set("threshold", threshold),
                msavi_calc.updateMask(msavi_calc.gte(ng_threshold))
                .rename("msavi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            msavi = msavi_calc.updateMask(msavi_calc.gte(threshold)).rename("msavi").copyProperties(image).set("threshold", threshold)
        return msavi

    @staticmethod
    def landsat_ndmi_fn(image, threshold, ng_threshold=None):
        """
        Calculates the Normalized Difference Moisture Index (NDMI) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.
        See Wilson & Sader, 2002 - https://doi.org/10.1016/S0034-4257(02)00074-7

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where NDMI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where NDMI pixels less than threshold are masked.
        
        Returns:
            ee.Image: NDMI ee.Image
        """
        ndmi_expression = '(b("SR_B5") - b("SR_B6")) / (b("SR_B5") + b("SR_B6"))'
        ndmi_calc = image.expression(ndmi_expression)
        if ng_threshold != None:
            ndmi = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                ndmi_calc.updateMask(ndmi_calc.gte(threshold))
                .rename("ndmi")
                .copyProperties(image)
                .set("threshold", threshold),
                ndmi_calc.updateMask(ndmi_calc.gte(ng_threshold))
                .rename("ndmi")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            ndmi = ndmi_calc.updateMask(ndmi_calc.gte(threshold)).rename("ndmi").copyProperties(image).set("threshold", threshold)
        return ndmi

    @staticmethod
    def landsat_nbr_fn(image, threshold, ng_threshold=None):
        """
        Calculates the Normalized Burn Ratio (NBR) for Landsat images. Masks image based on threshold. Can specify separate thresholds
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where NBR pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where NBR pixels less than threshold are masked.

        Returns:
            ee.Image: NBR ee.Image
        """
        nbr_expression = '(b("SR_B5") - b("SR_B7")) / (b("SR_B5") + b("SR_B7"))'
        nbr_calc = image.expression(nbr_expression)
        if ng_threshold != None:
            nbr = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                nbr_calc.updateMask(nbr_calc.gte(threshold))
                .rename("nbr")
                .copyProperties(image)
                .set("threshold", threshold),
                nbr_calc.updateMask(nbr_calc.gte(ng_threshold))
                .rename("nbr")
                .copyProperties(image)
                .set("threshold", ng_threshold),
            )
        else:
            nbr = nbr_calc.updateMask(nbr_calc.gte(threshold)).rename("nbr").copyProperties(image).set("threshold", threshold)
        return nbr

    @staticmethod
    def MaskWaterLandsat(image):
        """
        Masks water pixels based on Landsat image QA band.

        Args:
            image (ee.Image): input ee.Image

        Returns:
            ee.Image: ee.Image with water pixels masked.
        """
        WaterBitMask = ee.Number(2).pow(7).int()
        qa = image.select("QA_PIXEL")
        water_extract = qa.bitwiseAnd(WaterBitMask).eq(0)
        masked_image = image.updateMask(water_extract).copyProperties(image)
        return masked_image

    @staticmethod
    def MaskWaterLandsatByNDWI(image, threshold, ng_threshold=None):
        """
        Masks water pixels (mask land and cloud pixels) for all bands based on NDWI and a set threshold where
        all pixels less than NDWI threshold are masked out. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9

        Args:
            image (ee.Image): input image
            threshold (float): value between -1 and 1 where NDWI pixels greater than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where NDWI pixels greater than threshold are masked

        Returns:
            ee.Image: ee.Image with water pixels masked
        """
        ndwi_calc = image.normalizedDifference(
            ["SR_B3", "SR_B5"]
        )  # green-NIR / green+NIR -- full NDWI image
        water = (
            ndwi_calc.updateMask(ndwi_calc.gte(threshold))
            .rename("ndwi")
            .copyProperties(image)
        )
        if ng_threshold != None:
            water = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                image.updateMask(ndwi_calc.lt(threshold)).set("threshold", threshold),
                image.updateMask(ndwi_calc.lt(ng_threshold)).set(
                    "threshold", ng_threshold
                ),
            )
        else:
            water = image.updateMask(ndwi_calc.lt(threshold)).set(
                "threshold", threshold
            )
        return water

    @staticmethod
    def MaskToWaterLandsat(image):
        """
        Masks image to water pixels by masking land and cloud pixels based on Landsat image QA band.

        Args:
            image (ee.Image): input ee.Image

        Returns:
            ee.Image: ee.Image with water pixels masked.
        """
        WaterBitMask = ee.Number(2).pow(7).int()
        qa = image.select("QA_PIXEL")
        water_extract = qa.bitwiseAnd(WaterBitMask).neq(0)
        masked_image = image.updateMask(water_extract).copyProperties(image)
        return masked_image

    @staticmethod
    def MaskToWaterLandsatByNDWI(image, threshold, ng_threshold=None):
        """
        Masks water pixels using NDWI based on threshold. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9

        Args:
            image (ee.Image): input image
            threshold (float): value between -1 and 1 where NDWI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where NDWI pixels less than threshold are masked

        Returns:
            ee.Image: ee.Image with water pixels masked.
        """
        ndwi_calc = image.normalizedDifference(
            ["SR_B3", "SR_B5"]
        )  # green-NIR / green+NIR -- full NDWI image
        water = (
            ndwi_calc.updateMask(ndwi_calc.gte(threshold))
            .rename("ndwi")
            .copyProperties(image)
        )
        if ng_threshold != None:
            water = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                image.updateMask(ndwi_calc.gte(threshold)).set("threshold", threshold),
                image.updateMask(ndwi_calc.gte(ng_threshold)).set(
                    "threshold", ng_threshold
                ),
            )
        else:
            water = image.updateMask(ndwi_calc.gte(threshold)).set(
                "threshold", threshold
            )
        return water

    @staticmethod
    def halite_mask(image, threshold, ng_threshold=None):
        """
        Masks halite pixels after specifying index to isolate/mask-to halite pixels.

        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
            ng_threshold (float, optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            image (ee.Image): masked ee.Image
        """
        halite_index = image.normalizedDifference(
            ["SR_B4", "SR_B6"]
        )  # red-swir1 / red+swir1
        if ng_threshold != None:
            mask = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                image.updateMask(halite_index.lt(threshold)).copyProperties(image),
                image.updateMask(halite_index.lt(ng_threshold)).copyProperties(image),
            )
        else:
            mask = image.updateMask(halite_index.lt(threshold)).copyProperties(image)
        return mask

    @staticmethod
    def gypsum_and_halite_mask(
        image,
        halite_threshold,
        gypsum_threshold,
        halite_ng_threshold=None,
        gypsum_ng_threshold=None,
    ):
        """
        Masks both gypsum and halite pixels. Must specify threshold for isolating halite and gypsum pixels.

        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9.

        Args:
            image (ee.Image): input ee.Image
            halite_threshold (float): integer threshold for halite where pixels less than threshold are masked, applies to landsat 5 when ng_threshold is also set.
            gypsum_threshold (float): integer threshold for gypsum where pixels less than threshold are masked, applies to landsat 5 when ng_threshold is also set.
            halite_ng_threshold (float, optional): integer threshold for halite to be applied to landsat 8 or 9 where pixels less than threshold are masked
            gypsum_ng_threshold (float, optional): integer threshold for gypsum to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
            image (ee.Image): masked ee.Image
        """
        halite_index = image.normalizedDifference(
            ["SR_B4", "SR_B6"]
        )  # red-swir1 / red+swir1
        gypsum_index = image.normalizedDifference(["SR_B6", "SR_B7"])
        if halite_ng_threshold and gypsum_ng_threshold != None:
            mask = ee.Algorithms.If(
                ee.String(image.get("SPACECRAFT_ID")).equals("LANDSAT_5"),
                gypsum_index.updateMask(halite_index.lt(halite_threshold))
                .updateMask(gypsum_index.lt(gypsum_threshold))
                .rename("carbonate_muds")
                .copyProperties(image),
                gypsum_index.updateMask(halite_index.lt(halite_ng_threshold))
                .updateMask(gypsum_index.lt(gypsum_ng_threshold))
                .rename("carbonate_muds")
                .copyProperties(image),
            )
        else:
            mask = (
                gypsum_index.updateMask(halite_index.lt(halite_threshold))
                .updateMask(gypsum_index.lt(gypsum_threshold))
                .rename("carbonate_muds")
                .copyProperties(image)
            )
        return mask

    @staticmethod
    def maskL8clouds(image):
        """
        Masks clouds baseed on Landsat 8 QA band.

        Args:
            image (ee.Image): input ee.Image

        Returns:
            ee.Image: ee.Image
        """
        cloudBitMask = ee.Number(2).pow(3).int()
        CirrusBitMask = ee.Number(2).pow(2).int()
        qa = image.select("QA_PIXEL")
        cloud_mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        cirrus_mask = qa.bitwiseAnd(CirrusBitMask).eq(0)
        return image.updateMask(cloud_mask).updateMask(cirrus_mask)

    @staticmethod
    def temperature_bands(img):
        """
        Renames bands for temperature calculations.

        Args:
            img: input ee.Image

        Returns:
            ee.Image: ee.Image
        """
        # date = ee.Number(img.date().format('YYYY-MM-dd'))
        scale1 = ["ST_ATRAN", "ST_EMIS"]
        scale2 = ["ST_DRAD", "ST_TRAD", "ST_URAD"]
        scale1_names = ["transmittance", "emissivity"]
        scale2_names = ["downwelling", "B10_radiance", "upwelling"]
        scale1_bands = (
            img.select(scale1).multiply(0.0001).rename(scale1_names)
        )  # Scaled to new L8 collection
        scale2_bands = (
            img.select(scale2).multiply(0.001).rename(scale2_names)
        )  # Scaled to new L8 collection
        return img.addBands(scale1_bands).addBands(scale2_bands).copyProperties(img)

    @staticmethod
    def landsat_LST(image):
        """
        Calculates land surface temperature (LST) from landsat TIR bands.
        Based on Sekertekin, A., & Bonafoni, S. (2020) https://doi.org/10.3390/rs12020294

        Args:
            image (ee.Image): input ee.Image

        Returns:
            ee.Image: LST ee.Image
        """
        # Based on Sekertekin, A., & Bonafoni, S. (2020) https://doi.org/10.3390/rs12020294

        k1 = 774.89
        k2 = 1321.08
        LST = image.expression(
            "(k2/log((k1/((B10_rad - upwelling - transmittance*(1 - emissivity)*downwelling)/(transmittance*emissivity)))+1)) - 273.15",
            {
                "k1": k1,
                "k2": k2,
                "B10_rad": image.select("B10_radiance"),
                "upwelling": image.select("upwelling"),
                "transmittance": image.select("transmittance"),
                "emissivity": image.select("emissivity"),
                "downwelling": image.select("downwelling"),
            },
        ).rename("LST")
        return image.addBands(LST).copyProperties(image)  # Outputs temperature in C

    @staticmethod
    def PixelAreaSum(
        image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12
    ):
        """
        Calculates the summation of area for pixels of interest (above a specific threshold) in a geometry
        and store the value as image property (matching name of chosen band). If multiple band names are provided in a list,
        the function will calculate area for each band in the list and store each as a separate property.

        NOTE: The resulting value has units of square meters.

        Args:
            image (ee.Image): input ee.Image
            band_name (string or list of strings): name of band(s) (string) for calculating area. If providing multiple band names, pass as a list of strings.
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
            threshold (float): integer threshold to specify masking of pixels below threshold (defaults to -1). If providing multiple band names, the same threshold will be applied to all bands. Best practice in this case is to mask the bands prior to passing to this function and leave threshold at default of -1.
            scale (int): integer scale of image resolution (meters) (defaults to 30)
            maxPixels (int): integer denoting maximum number of pixels for calculations

        Returns:
            ee.Image: ee.Image with area calculation in square meters stored as property matching name of band
        """
        # Ensure band_name is a server-side ee.List for consistent processing. Wrap band_name in a list if it's a single string.
        bands = ee.List(band_name) if isinstance(band_name, list) else ee.List([band_name])
        # Create an image representing the area of each pixel in square meters
        area_image = ee.Image.pixelArea()

        # Function to iterate over each band and calculate area, storing the result as a property on the image
        def calculate_and_set_area(band, img_accumulator):
            # Explcitly cast inputs to expected types
            img_accumulator = ee.Image(img_accumulator)
            band = ee.String(band)

            # Create a mask from the input image for the current band
            mask = img_accumulator.select(band).gte(threshold)
            # Combine the original image with the area image
            final = img_accumulator.addBands(area_image)

            # Calculation of area for a given band, utilizing other inputs
            stats = (
                final.select("area").updateMask(mask)
                .rename(band) # renames 'area' to band name like 'ndwi'
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geometry,
                    scale=scale,
                    maxPixels=maxPixels,
                )
            )
            # Retrieving the area value from the stats dictionary with stats.get(band), as the band name is now the key
            reduced_area = stats.get(band)
            # Checking whether the calculated area is valid and replaces with 0 if not. This avoids breaking the loop for erroneous images.
            area_value = ee.Algorithms.If(reduced_area, reduced_area, 0)
            
            # Set the property on the image, named after the band
            return img_accumulator.set(band, area_value)

        # Call to iterate the calculate_and_set_area function over the list of bands, starting with the original image
        final_image = ee.Image(bands.iterate(calculate_and_set_area, image))
        return final_image

    def PixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12, output_type='ImageCollection', area_data_export_path=None
    ):
        """
        Calculates the geodesic summation of area for pixels of interest (above a specific threshold)
        within a geometry and stores the value as an image property (matching name of chosen band) for an entire
        image collection. Optionally exports the area data to a CSV file.

        NOTE: The resulting value has units of square meters.

        Args:
            band_name (string or list of strings): name of band(s) (string) for calculating area. If providing multiple band names, pass as a list of strings.
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
            threshold (float): integer threshold to specify masking of pixels below threshold (defaults to -1). If providing multiple band names, the same threshold will be applied to all bands. Best practice in this case is to mask the bands prior to passing to this function and leave threshold at default of -1.
            scale (int): integer scale of image resolution (meters) (defaults to 30)
            maxPixels (int): integer denoting maximum number of pixels for calculations
            output_type (str): 'ImageCollection' to return an ee.ImageCollection, 'LandsatCollection' to return a LandsatCollection object (defaults to 'ImageCollection')
            area_data_export_path (str, optional): If provided, the function will save the resulting area data to a CSV file at the specified path.

        Returns:
            ee.ImageCollection or LandsatCollection: Image collection of images with area calculation (square meters) stored as property matching name of band. Type of output depends on output_type argument.
        """
        # If the area calculation has not been computed for this LandsatCollection instance, the area will be calculated for the provided bands
        if self._PixelAreaSumCollection is None:
            collection = self.collection
            # Area calculation for each image in the collection, using the PixelAreaSum function
            AreaCollection = collection.map(
                lambda image: LandsatCollection.PixelAreaSum(
                    image,
                    band_name=band_name,
                    geometry=geometry,
                    threshold=threshold,
                    scale=scale,
                    maxPixels=maxPixels,
                )
            )
            # Storing the result in the instance variable to avoid redundant calculations
            self._PixelAreaSumCollection = AreaCollection

        # If an export path is provided, the area data will be exported to a CSV file
        if area_data_export_path:
            LandsatCollection(collection=self._PixelAreaSumCollection).ExportProperties(property_names=band_name, file_path=area_data_export_path+'.csv')

        # Returning the result in the desired format based on output_type argument or raising an error for invalid input
        if output_type == 'ImageCollection':
            return self._PixelAreaSumCollection
        elif output_type == 'LandsatCollection':
            return LandsatCollection(collection=self._PixelAreaSumCollection)
        else:
            raise ValueError("output_type must be 'ImageCollection' or 'LandsatCollection'")

    def merge(self, other):
        """
        Merges the current LandsatCollection with another LandsatCollection, where images/bands with the same date are combined to a single multiband image.

        Args:
            other (LandsatCollection): Another LandsatCollection to merge with current collection.

        Returns:
            LandsatCollection: A new LandsatCollection containing images from both collections.
        """
        # Checking if 'other' is an instance of LandsatCollection
        if not isinstance(other, LandsatCollection):
            raise ValueError("The 'other' parameter must be an instance of LandsatCollection.")
        
        # Merging the collections using the .combine() method
        merged_collection = self.collection.combine(other.collection)
        return LandsatCollection(collection=merged_collection)

    @staticmethod
    def dNDWIPixelAreaSum(image, geometry, band_name="ndwi", scale=30, maxPixels=1e12):
        """
        Dynamically calulates the summation of area for water pixels of interest and store the value as image property named 'ndwi'
        Uses Otsu thresholding to dynamically choose the best threshold rather than needing to specify threshold.
        Note: An offset of 0.15 is added to the Otsu threshold.

        Args:
            image (ee.Image): input ee.Image
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
            band_name (string): name of ndwi band (string) for calculating area (defaults to 'ndwi')
            scale (int): integer scale of image resolution (meters) (defaults to 30)
            maxPixels (int): integer denoting maximum number of pixels for calculations

        Returns:
            ee.Image: ee.Image with area calculation stored as property matching name of band
        """

        def OtsuThreshold(histogram):
            counts = ee.Array(ee.Dictionary(histogram).get("histogram"))
            means = ee.Array(ee.Dictionary(histogram).get("bucketMeans"))
            size = means.length().get([0])
            total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
            sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
            mean = sum.divide(total)
            indices = ee.List.sequence(1, size)

            def func_xxx(i):
                aCounts = counts.slice(0, 0, i)
                aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
                aMeans = means.slice(0, 0, i)
                aMean = (
                    aMeans.multiply(aCounts)
                    .reduce(ee.Reducer.sum(), [0])
                    .get([0])
                    .divide(aCount)
                )
                bCount = total.subtract(aCount)
                bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
                return aCount.multiply(aMean.subtract(mean).pow(2)).add(
                    bCount.multiply(bMean.subtract(mean).pow(2))
                )

            bss = indices.map(func_xxx)
            return means.sort(bss).get([-1])

        area_image = ee.Image.pixelArea()
        histogram = image.select(band_name).reduceRegion(
            reducer=ee.Reducer.histogram(255, 2),
            geometry=geometry.geometry().buffer(6000),
            scale=scale,
            bestEffort=True,
        )
        threshold = OtsuThreshold(histogram.get(band_name)).add(0.15)
        mask = image.select(band_name).gte(threshold)
        final = image.addBands(area_image)
        stats = (
            final.select("area")
            .updateMask(mask)
            .rename(band_name)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=scale,
                maxPixels=maxPixels,
            )
        )
        return image.set(band_name, stats.get(band_name))

    @property
    def dates_list(self):
        """
        Property attribute to retrieve list of dates as server-side (GEE) object.

        Returns:
            ee.List: Server-side ee.List of dates.
        """
        if self._dates_list is None:
            dates = self.collection.aggregate_array("Date_Filter")
            self._dates_list = dates
        return self._dates_list

    @property
    def dates(self):
        """
        Property attribute to retrieve list of dates as readable and indexable client-side list object.

        Returns:
            list: list of date strings.
        """
        if self._dates_list is None:
            dates = self.collection.aggregate_array("Date_Filter")
            self._dates_list = dates
        if self._dates is None:
            dates = self._dates_list.getInfo()
            self._dates = dates
        return self._dates

    def ExportProperties(self, property_names, file_path=None):
        """
        Fetches and returns specified properties from each image in the collection as a list, and returns a pandas DataFrame and optionally saves the results to a csv file.

        Args:
            property_names (list or str): A property name or list of property names to retrieve. The 'Date_Filter' property is always included to provide temporal context.
            file_path (str, optional): If provided, the function will save the resulting DataFrame to a CSV file at this path. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the requested properties for each image, sorted chronologically by 'Date_Filter'.
        """
        # Ensure property_names is a list for consistent processing
        if isinstance(property_names, str):
            property_names = [property_names]

        # Ensure properties are included without duplication, including 'Date_Filter'
        all_properties_to_fetch = list(set(['Date_Filter'] + property_names))

        # Defining the helper function to create features with specified properties
        def create_feature_with_properties(image):
            """A function to map over the collection and store the image properties as an ee.Feature.
            Args:
                image (ee.Image): An image from the collection.
            Returns:
                ee.Feature: A feature containing the specified properties from the image.
            """
            properties = image.toDictionary(all_properties_to_fetch)
            return ee.Feature(None, properties)

        # Map the feature creation function over the server-side collection.
        # The result is an ee.FeatureCollection where each feature holds the properties of one image.
        mapped_collection = self.collection.map(create_feature_with_properties)
        # Explicitly cast to ee.FeatureCollection for clarity
        feature_collection = ee.FeatureCollection(mapped_collection)

        # Use the existing ee_to_df static method. This performs the single .getInfo() call
        # and converts the structured result directly to a pandas DataFrame.
        df = LandsatCollection.ee_to_df(feature_collection, columns=all_properties_to_fetch)
        
        # Sort by date for a clean, chronological output.
        if 'Date_Filter' in df.columns:
            df = df.sort_values(by='Date_Filter').reset_index(drop=True)
        
        # Check condition for saving to CSV
        if file_path:
            # Check whether file_path ends with .csv, if not, append it
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            # Save DataFrame to CSV
            df.to_csv(file_path, index=True)
            print(f"Properties saved to {file_path}")
            
        return df

    def get_filtered_collection(self):
        """
        Filters image collection based on LandsatCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from LandsatCollection object
        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(
            LandsatCollection.landsat5bandrename
        )  # Replace with the correct Landsat 5 collection ID
        filtered_collection = (
            landsat8.merge(landsat9)
            .merge(landsat5)
            .filterDate(self.start_date, self.end_date)
            .filter(
                ee.Filter.And(
                    ee.Filter.inList("WRS_PATH", self.tile_path),
                    ee.Filter.inList("WRS_ROW", self.tile_row),
                )
            )
            .filter(ee.Filter.lte("CLOUD_COVER", self.cloud_percentage_threshold))
            .map(LandsatCollection.image_dater)
            .sort("Date_Filter")
        )
        return filtered_collection

    def get_boundary_filtered_collection(self):
        """
        Filters and masks image collection based on LandsatCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from LandsatCollection object

        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(
            LandsatCollection.landsat5bandrename
        )  # Replace with the correct Landsat 5 collection ID
        filtered_collection = (
            landsat8.merge(landsat9)
            .merge(landsat5)
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.boundary)
            .filter(ee.Filter.lte("CLOUD_COVER", self.cloud_percentage_threshold))
            .map(LandsatCollection.image_dater)
            .sort("Date_Filter")
        )
        return filtered_collection
    
    @property
    def scale_to_reflectance(self):
        """
        Scales each band in the Landsat collection from DN values to surface reflectance values.

        Returns:
            LandsatCollection: A new LandsatCollection object with bands scaled to reflectance.
        """
        if self._Reflectance is None:
            self._Reflectance = self.collection.map(_scale_landsat_sr)
        return LandsatCollection(collection=self._Reflectance)


    @property
    def median(self):
        """
        Property attribute function to calculate median image from image collection. Results are calculated once per class object then cached for future use.

        Returns:
            ee.Image: median image from entire collection.
        """
        if self._median is None:
            col = self.collection.median()
            self._median = col
        return self._median

    @property
    def mean(self):
        """
        Property attribute function to calculate mean image from image collection. Results are calculated once per class object then cached for future use.

        Returns:
            ee.Image: mean image from entire collection.

        """
        if self._mean is None:
            col = self.collection.mean()
            self._mean = col
        return self._mean

    @property
    def max(self):
        """
        Property attribute function to calculate max image from image collection. Results are calculated once per class object then cached for future use.

        Returns:
            ee.Image: max image from entire collection.
        """
        if self._max is None:
            col = self.collection.max()
            self._max = col
        return self._max

    @property
    def min(self):
        """
        Property attribute function to calculate min image from image collection. Results are calculated once per class object then cached for future use.

        Returns:
            ee.Image: min image from entire collection.
        """
        if self._min is None:
            col = self.collection.min()
            self._min = col
        return self._min

    @property
    def ndwi(self):
        """
        Property attribute to calculate and access the NDWI (Normalized Difference Water Index) imagery of the LandsatCollection.
        This property initiates the calculation of NDWI using a default threshold of -1 (or a previously set threshold of self.ndwi_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._ndwi is None:
            self._ndwi = self.ndwi_collection(self.ndwi_threshold)
        return self._ndwi

    @property
    def mndwi(self):
        """
        Property attribute to calculate and access the MNDWI (Modified Normalized Difference Water Index) imagery of the LandsatCollection.
        This property initiates the calculation of MNDWI using a default threshold of -1 (or a previously set threshold of self.mndwi_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._mndwi is None:
            self._mndwi = self.mndwi_collection(self.mndwi_threshold)
        return self._mndwi

    def ndwi_collection(self, threshold, ng_threshold=None):
        """
        Calculates ndwi and returns collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for NDWI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()

        if available_bands.contains("SR_B3") and available_bands.contains("SR_B5"):
            pass
        else:
            raise ValueError("Insufficient Bands for ndwi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_ndwi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    def mndwi_collection(self, threshold, ng_threshold=None):
        """
        Calculates mndwi and returns collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the mndwi property attribute.

        Args:
            threshold (float): specify threshold for MNDWI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()

        if available_bands.contains("SR_B3") and available_bands.contains("SR_B6"):
            pass
        else:
            raise ValueError("Insufficient bands for mndwi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_mndwi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def ndvi(self):
        """
        Property attribute to calculate and access the NDVI (Normalized Difference Vegetation Index) imagery of the LandsatCollection.
        This property initiates the calculation of NDVI using a default threshold of -1 (or a previously set threshold of self.ndvi_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._ndvi is None:
            self._ndvi = self.ndvi_collection(self.ndvi_threshold)
        return self._ndvi

    def ndvi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the NDVI (Normalized Difference Vegetation Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for NDVI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B5"):
            pass
        else:
            raise ValueError("Insufficient Bands for ndwi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_ndvi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    @property
    def evi(self):
        """
        Property attribute to calculate and access the EVI (Enhanced Vegetation Index) imagery of the LandsatCollection.
        This property initiates the calculation of EVI and caches the result. The calculation is performed only once when 
        the property is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._evi is None:
            self._evi = self.evi_collection(self.evi_threshold)
        return self._evi

    def evi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the EVI (Enhanced Vegetation Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the evi property attribute.

        Args:
            threshold (float): specify threshold for EVI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B5") and available_bands.contains("SR_B2"):
            pass
        else:
            raise ValueError("Insufficient Bands for evi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_evi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    @property
    def savi(self):
        """
        Property attribute to calculate and access the SAVI (Soil Adjusted Vegetation Index) imagery of the LandsatCollection.
        This property initiates the calculation of SAVI and caches the result. The calculation is performed only once when the 
        property is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._savi is None:
            self._savi = self.savi_collection(self.savi_threshold)
        return self._savi

    def savi_collection(self, threshold, ng_threshold=None, l=0.5):
        """
        Function to calculate the SAVI (Soil Adjusted Vegetation Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the savi property attribute.

        Args:
            threshold (float): specify threshold for SAVI function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 SAVI function (values less than threshold are masked)
            l (float, optional): Soil brightness correction factor, typically set to 0.5 for intermediate vegetation cover. Defaults to 0.5.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B5"):
            pass
        else:
            raise ValueError("Insufficient Bands for savi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_savi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold, l=l
            )
        )
        return LandsatCollection(collection=col)

    @property
    def msavi(self):
        """
        Property attribute to calculate and access the MSAVI (Modified Soil Adjusted Vegetation Index) imagery of the LandsatCollection.
        This property initiates the calculation of MSAVI and caches the result. The calculation is performed only once when the property 
        is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._msavi is None:
            self._msavi = self.msavi_collection(self.msavi_threshold)
        return self._msavi  
    
    def msavi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the MSAVI (Modified Soil Adjusted Vegetation Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the msavi property attribute.

        Args:
            threshold (float): specify threshold for MSAVI function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 MSAVI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B5"):
            pass
        else:
            raise ValueError("Insufficient Bands for msavi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_msavi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def ndmi(self):
        """
        Property attribute to calculate and access the NDMI (Normalized Difference Moisture Index) imagery of the LandsatCollection.
        This property initiates the calculation of NDMI and caches the result. The calculation is performed only once when the property 
        is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._ndmi is None:
            self._ndmi = self.ndmi_collection(self.ndmi_threshold)
        return self._ndmi
    
    def ndmi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the NDMI (Normalized Difference Moisture Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9 (when `ng_threshold` is specified, otherwise `threshold` applies to all imagery). This function can be called as a method but is called
        by default when using the ndmi property attribute.

        Args:
            threshold (float): specify threshold for NDMI function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 NDMI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B5") and available_bands.contains("SR_B6"):
            pass
        else:
            raise ValueError("Insufficient Bands for ndmi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_ndmi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    @property
    def nbr(self):
        """
        Property attribute to calculate and access the NBR (Normalized Burn Ratio) imagery of the LandsatCollection.
        This property initiates the calculation of NBR using a default threshold of -1 (or a previously set threshold of self.nbr_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._nbr is None:
            self._nbr = self.nbr_collection(self.nbr_threshold)
        return self._nbr
    
    def nbr_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the NBR (Normalized Burn Ratio) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the nbr property attribute.

        Args:
            threshold (float): specify threshold for NBR function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 NBR function (values less than threshold are masked)
        
        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B5") and available_bands.contains("SR_B7"):
            pass
        else:
            raise ValueError("Insufficient Bands for nbr calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_nbr_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    @property
    def ndsi(self):
        """
        Property attribute to calculate and access the NDSI (Normalized Difference Snow Index) imagery of the LandsatCollection.
        This property initiates the calculation of NDSI and caches the result. The calculation is performed only once when the 
        property is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._ndsi is None:
            self._ndsi = self.ndsi_collection(self.ndsi_threshold)
        return self._ndsi
    
    def ndsi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the NDSI (Normalized Difference Snow Index) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndsi property attribute.

        Args:
            threshold (float): specify threshold for NDSI function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 NDSI function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B3") and available_bands.contains("SR_B6"):
            pass
        else:
            raise ValueError("Insufficient Bands for ndsi calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_ndsi_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def halite(self):
        """
        Property attribute to calculate and access the halite index (see Radwin & Bowen, 2021) imagery of the LandsatCollection.
        This property initiates the calculation of halite using a default threshold of -1 (or a previously set threshold of self.halite_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._halite is None:
            self._halite = self.halite_collection(self.halite_threshold)
        return self._halite

    def halite_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the halite index (see Radwin & Bowen, 2021) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for halite function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 halite function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B6"):
            pass
        else:
            raise ValueError("Insufficient Bands for halite calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_halite_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def gypsum(self):
        """
        Property attribute to calculate and access the gypsum/sulfate index (see Radwin & Bowen, 2021) imagery of the LandsatCollection.
        This property initiates the calculation of gypsum using a default threshold of -1 (or a previously set threshold of self.gypsum_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._gypsum is None:
            self._gypsum = self.gypsum_collection(self.gypsum_threshold)
        return self._gypsum

    def gypsum_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the gypsum index (see Radwin & Bowen, 2021) and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for gypsum function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B6") and available_bands.contains("SR_B7"):
            pass
        else:
            raise ValueError("Insufficient Bands for gypsum calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_gypsum_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)
    
    @property
    def albedo(self):
        """
        Property attribute to calculate albedo imagery, based on Liang, 2001 
        (https://doi.org/10.1016/S0034-4257(00)00205-4) and Wang et al., 2016 (https://doi.org/10.1016/j.rse.2016.02.059).
        This property initiates the calculation of albedo and caches the result. The calculation is performed only once when the property 
        is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._albedo is None:
            self._albedo = self.albedo_collection()
        return self._albedo


    def albedo_collection(self):
        """
        Calculates albedo and returns collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.
        Albedo calculation based on Liang, 2001 (https://doi.org/10.1016/S0034-4257(00)00205-4) 
        and Wang et al., 2016 (https://doi.org/10.1016/j.rse.2016.02.059).

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if (
            available_bands.contains("SR_B1")
            and available_bands.contains("SR_B2")
            and available_bands.contains("SR_B3")
            and available_bands.contains("SR_B4")
            and available_bands.contains("SR_B5")
            and available_bands.contains("SR_B6")
            and available_bands.contains("SR_B7")
        ):
            pass
        else:
            raise ValueError("Insufficient Bands for albedo calculation")
        col = self.collection.map(LandsatCollection.landsat_broadband_albedo_fn)
        return LandsatCollection(collection=col)

    @property
    def turbidity(self):
        """
        Property attribute to calculate and access the turbidity (NDTI) imagery of the LandsatCollection.
        This property initiates the calculation of turbidity using a default threshold of -1 (or a previously set threshold of self.turbidity_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._turbidity is None:
            self._turbidity = self.turbidity_collection(self.turbidity_threshold)
        return self._turbidity

    def turbidity_collection(self, threshold, ng_threshold=None):
        """
        Calculates the turbidity (NDTI) index and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for the turbidity function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 turbidity function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains("SR_B4") and available_bands.contains("SR_B3"):
            pass
        else:
            raise ValueError("Insufficient Bands for turbidity calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_ndti_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )

        return LandsatCollection(collection=col)


    @property
    def chlorophyll(self):
        """
        Property attribute to calculate and access the chlorophyll (NDTI) imagery of the LandsatCollection.
        This property initiates the calculation of chlorophyll using a default threshold of -1 (or a previously set threshold of self.chlorophyll_threshold)
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned
        on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        if self._chlorophyll is None:
            self._chlorophyll = self.chlorophyll_collection(self.chlorophyll_threshold)
        return self._chlorophyll

    def chlorophyll_collection(self, threshold, ng_threshold=None):
        """
        Calculates the KIVU chlorophyll index and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called
        by default when using the ndwi property attribute.

        Args:
            threshold (float): specify threshold for the turbidity function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 turbidity function (values less than threshold are masked)

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if (
            available_bands.contains("SR_B4")
            and available_bands.contains("SR_B3")
            and available_bands.contains("SR_B2")
        ):
            pass
        else:
            raise ValueError("Insufficient Bands for chlorophyll calculation")
        col = self.collection.map(
            lambda image: LandsatCollection.landsat_kivu_chla_fn(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def masked_water_collection(self):
        """
        Property attribute to mask water and return collection as class object.

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        if self._masked_water_collection is None:
            col = self.collection.map(LandsatCollection.MaskWaterLandsat)
            self._masked_water_collection = LandsatCollection(collection=col)
        return self._masked_water_collection

    def masked_water_collection_NDWI(self, threshold):
        """
        Masks water pixels based on NDWI and user set threshold.

        Args:
            threshold (float): specify threshold for NDWI function (values greater than threshold are masked)

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        col = self.collection.map(
            lambda image: LandsatCollection.MaskWaterLandsatByNDWI(
                image, threshold=threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def masked_to_water_collection(self):
        """
        Property attribute to mask image to water, removing land and cloud pixels, and return collection as class object.

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        if self._masked_to_water_collection is None:
            col = self.collection.map(LandsatCollection.MaskToWaterLandsat)
            self._masked_to_water_collection = LandsatCollection(collection=col)
        return self._masked_to_water_collection

    def masked_to_water_collection_NDWI(self, threshold):
        """
        Function to mask all but water pixels based on NDWI and user set threshold.

        Args:
            threshold (float): specify threshold for NDWI function (values less than threshold are masked)

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        col = self.collection.map(
            lambda image: LandsatCollection.MaskToWaterLandsatByNDWI(
                image, threshold=threshold
            )
        )
        return LandsatCollection(collection=col)

    @property
    def masked_clouds_collection(self):
        """
        Property attribute to mask clouds and return collection as class object.

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        if self._masked_clouds_collection is None:
            col = self.collection.map(LandsatCollection.maskL8clouds)
            self._masked_clouds_collection = LandsatCollection(collection=col)
        return self._masked_clouds_collection

    @property
    def LST(self):
        """
        Property attribute to calculate and access the LST (Land Surface Temperature - in Celcius) imagery of the LandsatCollection.
        This property initiates the calculation of LST and caches the result. The calculation is performed only once
        when the property is first accessed, and the cached result is returned on subsequent accesses.

        Returns:
            LandsatCollection: A LandsatCollection image collection object containing LST imagery (temperature in Celcius).
        """
        if self._LST is None:
            self._LST = self.surface_temperature_collection()
        return self._LST

    def surface_temperature_collection(self):
        """
        Function to calculate LST (Land Surface Temperature - in Celcius) and return collection as class object.

        Returns:
            LandsatCollection: A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if (
            available_bands.contains("ST_ATRAN")
            and available_bands.contains("ST_EMIS")
            and available_bands.contains("ST_DRAD")
            and available_bands.contains("ST_TRAD")
            and available_bands.contains("ST_URAD")
        ):
            pass
        else:
            raise ValueError("Insufficient Bands for temperature calculation")
        col = (
            self.collection.map(LandsatCollection.temperature_bands)
            .map(LandsatCollection.landsat_LST)
            .map(LandsatCollection.image_dater)
        )
        return LandsatCollection(collection=col)

    def mask_to_polygon(self, polygon):
        """
        Function to mask LandsatCollection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args:
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            LandsatCollection: masked LandsatCollection image collection

        """
        if self._geometry_masked_collection is None:
            # Convert the polygon to a mask
            mask = ee.Image.constant(1).clip(polygon)

            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(mask))

            # Update the internal collection state
            self._geometry_masked_collection = LandsatCollection(
                collection=masked_collection
            )

        # Return the updated object
        return self._geometry_masked_collection

    def mask_out_polygon(self, polygon):
        """
        Function to mask LandsatCollection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args:
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            LandsatCollection: masked LandsatCollection image collection

        """
        if self._geometry_masked_out_collection is None:
            # Convert the polygon to a mask
            full_mask = ee.Image.constant(1)

            # Use paint to set pixels inside polygon as 0
            area = full_mask.paint(polygon, 0)

            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(area))

            # Update the internal collection state
            self._geometry_masked_out_collection = LandsatCollection(
                collection=masked_collection
            )

        # Return the updated object
        return self._geometry_masked_out_collection

    def mask_halite(self, threshold, ng_threshold=None):
        """
        Masks halite and returns collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images
        where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
            threshold (float): specify threshold for gypsum function (values less than threshold are masked)
            ng_threshold (float, optional): specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked).

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        col = self.collection.map(
            lambda image: LandsatCollection.halite_mask(
                image, threshold=threshold, ng_threshold=ng_threshold
            )
        )
        return LandsatCollection(collection=col)

    def mask_halite_and_gypsum(
        self,
        halite_threshold,
        gypsum_threshold,
        halite_ng_threshold=None,
        gypsum_ng_threshold=None,
    ):
        """
        Masks halite and gypsum and returns collection as class object.
        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9.

        Args:
            halite_threshold (float): specify threshold for halite function (values less than threshold are masked)
            halite_ng_threshold (float, optional): specify threshold for Landsat 8&9 halite function (values less than threshold are masked)
            gypsum_threshold (float): specify threshold for gypsum function (values less than threshold are masked)
            gypsum_ng_threshold (float, optional): specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        col = self.collection.map(
            lambda image: LandsatCollection.gypsum_and_halite_mask(
                image,
                halite_threshold=halite_threshold,
                gypsum_threshold=gypsum_threshold,
                halite_ng_threshold=halite_ng_threshold,
                gypsum_ng_threshold=gypsum_ng_threshold,
            )
        )
        return LandsatCollection(collection=col)
    
    def binary_mask(self, threshold=None, band_name=None):
        """
        Function to create a binary mask (value of 1 for pixels above set threshold and value of 0 for all other pixels) of the LandsatCollection image collection based on a specified band.
        If a singleband image is provided, the band name is automatically determined.
        If multiple bands are available, the user must specify the band name to use for masking.

        Args:
            band_name (str, optional): The name of the band to use for masking. Defaults to None.

        Returns:
            LandsatCollection: LandsatCollection singleband image collection with binary masks applied.
        """
        if self.collection.size().eq(0).getInfo():
            raise ValueError("The collection is empty. Cannot create a binary mask.")
        if band_name is None:
            first_image = self.collection.first()
            band_names = first_image.bandNames()
            if band_names.size().getInfo() == 0:
                raise ValueError("No bands available in the collection.")
            if band_names.size().getInfo() > 1:
                raise ValueError("Multiple bands available, please specify a band name.")
            else:
                band_name = band_names.get(0).getInfo()
        if threshold is None:
            raise ValueError("Threshold must be specified for binary masking.")

        col = self.collection.map(
            lambda image: image.select(band_name).gte(threshold).rename(band_name)
        )
        return LandsatCollection(collection=col)

    def image_grab(self, img_selector):
        """
        Selects ("grabs") an image by index from the collection. Easy way to get latest image or browse imagery one-by-one.

        Args:
            img_selector: index of image in the collection for which user seeks to select/"grab".

        Returns:
            ee.Image: ee.Image of selected image
        """
        # Convert the collection to a list
        image_list = self.collection.toList(self.collection.size())

        # Get the image at the specified index
        image = ee.Image(image_list.get(img_selector))

        return image

    def custom_image_grab(self, img_col, img_selector):
        """
        Function to select ("grab") image of a specific index from an ee.ImageCollection object.

        Args:
            img_col: ee.ImageCollection with same dates as another LandsatCollection image collection object.
            img_selector: index of image in list of dates for which user seeks to "select".

        Returns:
            ee.Image: ee.Image of selected image
        """
        # Convert the collection to a list
        image_list = img_col.toList(img_col.size())

        # Get the image at the specified index
        image = ee.Image(image_list.get(img_selector))

        return image

    def image_pick(self, img_date):
        """
        Selects ("grabs") image of a specific date in format of 'YYYY-MM-DD' - will not work correctly if collection is composed of multiple images of the same date.

        Args:
            img_date: date (str) of image to select in format of 'YYYY-MM-DD'

        Returns:
            ee.Image: ee.Image of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq("Date_Filter", img_date))
        return new_col.first()

    def CollectionStitch(self, img_col2):
        """
        Function to mosaic two LandsatCollection objects which share image dates.
        Mosaics are only formed for dates where both image collections have images.
        Image properties are copied from the primary collection. Server-side friendly.

        Args:
            img_col2: secondary LandsatCollection image collection to be mosaiced with the primary image collection

        Returns:
            LandsatCollection: LandsatCollection image collection
        """
        dates_list = (
            ee.List(self._dates_list).cat(ee.List(img_col2.dates_list)).distinct()
        )
        filtered_dates1 = self._dates_list
        filtered_dates2 = img_col2._dates_list

        filtered_col2 = img_col2.collection.filter(
            ee.Filter.inList("Date_Filter", filtered_dates1)
        )
        filtered_col1 = self.collection.filter(
            ee.Filter.inList(
                "Date_Filter", filtered_col2.aggregate_array("Date_Filter")
            )
        )

        # Create a function that will be mapped over filtered_col1
        def mosaic_images(img):
            # Get the date of the image
            date = img.get("Date_Filter")

            # Get the corresponding image from filtered_col2
            img2 = filtered_col2.filter(ee.Filter.equals("Date_Filter", date)).first()

            # Create a mosaic of the two images
            mosaic = ee.ImageCollection.fromImages([img, img2]).mosaic()

            # Copy properties from the first image and set the 'Date_Filter' property
            mosaic = (
                mosaic.copyProperties(img)
                .set("Date_Filter", date)
                .set("system:time_start", img.get("system:time_start"))
            )

            return mosaic

        # Map the function over filtered_col1
        new_col = filtered_col1.map(mosaic_images)

        # Return a LandsatCollection instance
        return LandsatCollection(collection=new_col)

    @property
    def MosaicByDate(self):
        """
        Property attribute function to mosaic collection images that share the same date.

        The property CLOUD_COVER for each image is used to calculate an overall mean,
        which replaces the CLOUD_COVER property for each mosaiced image.
        Server-side friendly.

        NOTE: if images are removed from the collection from cloud filtering, you may have mosaics composed of only one image.

        Returns:
            LandsatCollection: LandsatCollection image collection with mosaiced imagery and mean CLOUD_COVER as a property
        """
        if self._MosaicByDate is None:
            input_collection = self.collection

            # Function to mosaic images of the same date and accumulate them
            def mosaic_and_accumulate(date, list_accumulator):
                # date = ee.Date(date)
                list_accumulator = ee.List(list_accumulator)
                date_filter = ee.Filter.eq("Date_Filter", date)
                date_collection = input_collection.filter(date_filter)
                # Convert the collection to a list
                image_list = date_collection.toList(date_collection.size())

                # Get the image at the specified index
                first_image = ee.Image(image_list.get(0))
                # Create mosaic
                mosaic = date_collection.mosaic().set("Date_Filter", date)

                # Calculate cumulative cloud and no data percentages
                cloud_percentage = date_collection.aggregate_mean("CLOUD_COVER")

                props_of_interest = [
                    "SPACECRAFT_ID",
                    "SENSOR_ID",
                    "PROCESSING_LEVEL",
                    "ACQUISITION_DATE",
                    "system:time_start",
                ]

                # mosaic = mosaic.copyProperties(self.image_grab(0), props_of_interest).set({
                #     'CLOUD_COVER': cloud_percentage
                # })
                mosaic = mosaic.copyProperties(first_image, props_of_interest).set(
                    {"CLOUD_COVER": cloud_percentage}
                )

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = LandsatCollection(collection=new_col)
            self._MosaicByDate = col

        # Convert the list of mosaics to an ImageCollection
        return self._MosaicByDate

    @staticmethod
    def ee_to_df(
        ee_object, columns=None, remove_geom=True, sort_columns=False, **kwargs
    ):
        """
        Converts an ee.FeatureCollection to pandas dataframe. Adapted from the geemap package (https://geemap.org/common/#geemap.common.ee_to_df)

        Args:
            ee_object (ee.FeatureCollection): ee.FeatureCollection.
            columns (list): List of column names. Defaults to None.
            remove_geom (bool): Whether to remove the geometry column. Defaults to True.
            sort_columns (bool): Whether to sort the column names. Defaults to False.
            kwargs: Additional arguments passed to ee.data.computeFeature.

        Raises:
            TypeError: ee_object must be an ee.FeatureCollection

        Returns:
            pd.DataFrame: pandas DataFrame
        """
        if isinstance(ee_object, ee.Feature):
            ee_object = ee.FeatureCollection([ee_object])

        if not isinstance(ee_object, ee.FeatureCollection):
            raise TypeError("ee_object must be an ee.FeatureCollection")

        try:
            property_names = ee_object.first().propertyNames().sort().getInfo()
            if remove_geom:
                data = ee_object.map(
                    lambda f: ee.Feature(None, f.toDictionary(property_names))
                )
            else:
                data = ee_object

            kwargs["expression"] = data
            kwargs["fileFormat"] = "PANDAS_DATAFRAME"

            df = ee.data.computeFeatures(kwargs)

            if isinstance(columns, list):
                df = df[columns]

            if remove_geom and ("geo" in df.columns):
                df = df.drop(columns=["geo"], axis=1)

            if sort_columns:
                df = df.reindex(sorted(df.columns), axis=1)

            return df
        except Exception as e:
            raise Exception(e)

    @staticmethod
    def extract_transect(
        image,
        line,
        reducer="mean",
        n_segments=100,
        dist_interval=None,
        scale=None,
        crs=None,
        crsTransform=None,
        tileScale=1.0,
        to_pandas=False,
        **kwargs,
    ):
        """
        Extracts transect from an image. Adapted from the geemap package (https://geemap.org/common/#geemap.common.extract_transect).

        Args:
            image (ee.Image): The image to extract transect from.
            line (ee.Geometry.LineString): The LineString used to extract transect from an image.
            reducer (str, optional): The ee.Reducer to use, e.g., 'mean', 'median', 'min', 'max', 'stdDev'. Defaults to "mean".
            n_segments (int, optional): The number of segments that the LineString will be split into. Defaults to 100.
            dist_interval (float, optional): The distance interval used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to None.
            scale (float, optional): A nominal scale in meters of the projection to work in. Defaults to None.
            crs (ee.Projection, optional): The projection to work in. If unspecified, the projection of the image's first band is used. If specified in addition to scale, rescaled to the specified scale. Defaults to None.
            crsTransform (list, optional): The list of CRS transform values. This is a row-major ordering of the 3x2 transform matrix. This option is mutually exclusive with 'scale', and will replace any transform already set on the projection. Defaults to None.
            tileScale (float, optional): A scaling factor used to reduce aggregation tile size; using a larger tileScale (e.g. 2 or 4) may enable computations that run out of memory with the default. Defaults to 1.
            to_pandas (bool, optional): Whether to convert the result to a pandas dataframe. Default to False.

        Raises:
            TypeError: If the geometry type is not LineString.
            Exception: If the program fails to compute.

        Returns:
            ee.FeatureCollection: The FeatureCollection containing the transect with distance and reducer values.
        """
        try:
            geom_type = line.type().getInfo()
            if geom_type != "LineString":
                raise TypeError("The geometry type must be LineString.")

            reducer = eval("ee.Reducer." + reducer + "()")
            maxError = image.projection().nominalScale().divide(5)

            length = line.length(maxError)
            if dist_interval is None:
                dist_interval = length.divide(n_segments)

            distances = ee.List.sequence(0, length, dist_interval)
            lines = line.cutLines(distances, maxError).geometries()

            def set_dist_attr(l):
                l = ee.List(l)
                geom = ee.Geometry(l.get(0))
                distance = ee.Number(l.get(1))
                geom = ee.Geometry.LineString(geom.coordinates())
                return ee.Feature(geom, {"distance": distance})

            lines = lines.zip(distances).map(set_dist_attr)
            lines = ee.FeatureCollection(lines)

            transect = image.reduceRegions(
                **{
                    "collection": ee.FeatureCollection(lines),
                    "reducer": reducer,
                    "scale": scale,
                    "crs": crs,
                    "crsTransform": crsTransform,
                    "tileScale": tileScale,
                }
            )

            if to_pandas:
                return LandsatCollection.ee_to_df(transect)
            return transect

        except Exception as e:
            raise Exception(e)

    @staticmethod
    def transect(
        image,
        lines,
        line_names,
        reducer="mean",
        n_segments=None,
        dist_interval=30,
        to_pandas=True,
    ):
        """
        Computes and stores the values along a transect for each line in a list of lines. Builds off of the extract_transect function from the geemap package
        where checks are ran to ensure that the reducer column is present in the transect data. If the reducer column is not present, a column of NaNs is created.
        An ee reducer is used to aggregate the values along the transect, depending on the number of segments or distance interval specified. Defaults to 'mean' reducer.

        Args:
            image (ee.Image): ee.Image object to use for calculating transect values.
            lines (list): List of ee.Geometry.LineString objects.
            line_names (list of strings): List of line string names.
            reducer (str): The ee reducer to use. Defaults to 'mean'.
            n_segments (int): The number of segments that the LineString will be split into. Defaults to None.
            dist_interval (float): The distance interval in meters used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 30.
            to_pandas (bool): Whether to convert the result to a pandas dataframe. Defaults to True.

        Returns:
            pd.DataFrame or ee.FeatureCollection: organized list of values along the transect(s)
        """
        # Create empty dataframe
        transects_df = pd.DataFrame()

        # Check if line is a list of lines or a single line - if single line, convert to list
        if isinstance(lines, list):
            pass
        else:
            lines = [lines]

        for i, line in enumerate(lines):
            if n_segments is None:
                transect_data = LandsatCollection.extract_transect(
                    image=image,
                    line=line,
                    reducer=reducer,
                    dist_interval=dist_interval,
                    to_pandas=to_pandas,
                )
                if reducer in transect_data.columns:
                    # Extract the 'mean' column and rename it
                    mean_column = transect_data[["mean"]]
                else:
                    # Handle the case where 'mean' column is not present
                    print(
                        f"{reducer} column not found in transect data for line {line_names[i]}"
                    )
                    # Create a column of NaNs with the same length as the longest column in transects_df
                    max_length = max(transects_df.shape[0], transect_data.shape[0])
                    mean_column = pd.Series([np.nan] * max_length)
            else:
                transect_data = LandsatCollection.extract_transect(
                    image=image,
                    line=line,
                    reducer=reducer,
                    n_segments=n_segments,
                    to_pandas=to_pandas,
                )
                if reducer in transect_data.columns:
                    # Extract the 'mean' column and rename it
                    mean_column = transect_data[["mean"]]
                else:
                    # Handle the case where 'mean' column is not present
                    print(
                        f"{reducer} column not found in transect data for line {line_names[i]}"
                    )
                    # Create a column of NaNs with the same length as the longest column in transects_df
                    max_length = max(transects_df.shape[0], transect_data.shape[0])
                    mean_column = pd.Series([np.nan] * max_length)

            transects_df = pd.concat([transects_df, mean_column], axis=1)

        transects_df.columns = line_names

        return transects_df

    def transect_iterator(
        self,
        lines,
        line_names,
        reducer="mean",
        dist_interval=30,
        n_segments=None,
        scale=30,
        processing_mode='aggregated',
        save_folder_path=None,
        sampling_method='line',
        point_buffer_radius=15
    ):
        """
        Computes and returns pixel values along transects for each image in a collection.

        This iterative function generates time-series data along one or more lines, and 
        supports two different geometric sampling methods ('line' and 'buffered_point') 
        for maximum flexibility and performance.

        There are two processing modes available, aggregated and iterative:
        - 'aggregated' (default; suggested): Fast, server-side processing. Fetches all results
            in a single request. Highly recommended. Returns a dictionary of pandas DataFrames.
        - 'iterative': Slower, client-side loop that processes one image at a time.
            Kept for backward compatibility (effectively depreciated). Returns None and saves individual CSVs.
            This method is not recommended unless absolutely necessary, as it is less efficient and may be subject to client-side timeouts.

        Args:
            lines (list): A list of one or more ee.Geometry.LineString objects that
                define the transects.
            line_names (list): A list of string names for each transect. The length
                of this list must match the length of the `lines` list.
            reducer (str, optional): The name of the ee.Reducer to apply at each
                transect point (e.g., 'mean', 'median', 'first'). Defaults to 'mean'.
            dist_interval (float, optional): The distance interval in meters for
                sampling points along each transect. Will be overridden if `n_segments` is provided.
                Defaults to 30. Recommended to increase this value when using the 
                'line' processing method, or else you may get blank rows.
            n_segments (int, optional): The number of equal-length segments to split
                each transect line into for sampling. This parameter overrides `dist_interval`. 
                Defaults to None.
            scale (int, optional): The nominal scale in meters for the reduction,
                which should typically match the pixel resolution of the imagery.
                Defaults to 30.
            processing_mode (str, optional): The method for processing the collection.
                - 'aggregated' (default): Fast, server-side processing. Fetches all
                  results in a single request. Highly recommended. Returns a dictionary
                  of pandas DataFrames.
                - 'iterative': Slower, client-side loop that processes one image at a
                  time. Kept for backward compatibility. Returns None and saves
                  individual CSVs.
            save_folder_path (str, optional): If provided, the function will save the
                resulting transect data to CSV files. The behavior depends on the
                `processing_mode`:
                - In 'aggregated' mode, one CSV is saved for each transect,
                  containing all dates. (e.g., 'MyTransect_transects.csv').
                - In 'iterative' mode, one CSV is saved for each date,
                  containing all transects. (e.g., '2022-06-15_transects.csv').
            sampling_method (str, optional): The geometric method used for sampling.
                - 'line' (default): Reduces all pixels intersecting each small line
                  segment. This can be unreliable and produce blank rows if
                  `dist_interval` is too small relative to the `scale`.
                - 'buffered_point': Reduces all pixels within a buffer around the
                  midpoint of each line segment. This method is more robust and
                  reliably avoids blank rows, but may not reduce all pixels along a line segment.
            point_buffer_radius (int, optional): The radius in meters for the buffer
                when `sampling_method` is 'buffered_point'. Defaults to 15.

        Returns:
            dict or None:
            - If `processing_mode` is 'aggregated', returns a dictionary where each
              key is a transect name and each value is a pandas DataFrame. In the
              DataFrame, the index is the distance along the transect and each
              column represents an image date. Optionally saves CSV files if
                `save_folder_path` is provided.
            - If `processing_mode` is 'iterative', returns None as it saves
              files directly.

        Raises:
            ValueError: If `lines` and `line_names` have different lengths, or if
                an unknown reducer or processing mode is specified.
        """
        # Validating inputs
        if len(lines) != len(line_names):
            raise ValueError("'lines' and 'line_names' must have the same number of elements.")
        ### Current, server-side processing method ###
        if processing_mode == 'aggregated':
            # Validating reducer type
            try:
                ee_reducer = getattr(ee.Reducer, reducer)()
            except AttributeError:
                raise ValueError(f"Unknown reducer: '{reducer}'.")
            ### Function to extract transects for a single image
            def get_transects_for_image(image):
                image_date = image.get('Date_Filter')
                # Initialize an empty list to hold all transect FeatureCollections
                all_transects_for_image = ee.List([])
                # Looping through each line and processing
                for i, line in enumerate(lines):
                    # Index line and name
                    line_name = line_names[i]
                    # Determine maxError based on image projection, used for geometry operations
                    maxError = image.projection().nominalScale().divide(5)
                    # Calculate effective distance interval
                    length = line.length(maxError) # using maxError here ensures consistency with cutLines
                    # Determine effective distance interval based on n_segments or dist_interval
                    effective_dist_interval = ee.Algorithms.If(
                        n_segments,
                        length.divide(n_segments),
                        dist_interval or 30 # Defaults to 30 if both are None
                    )
                    # Generate distances along the line(s) for segmentation
                    distances = ee.List.sequence(0, length, effective_dist_interval)
                    # Segmenting the line into smaller lines at the specified distances
                    cut_lines_geoms = line.cutLines(distances, maxError).geometries()
                    # Function to create features with distance attributes
                    # Adjusted to ensure consistent return types
                    def set_dist_attr(l):
                        # l is a list: [geometry, distance]
                        # Extracting geometry portion of line
                        geom_segment = ee.Geometry(ee.List(l).get(0))
                        # Extracting distance value for attribute
                        distance = ee.Number(ee.List(l).get(1))
                        ### Determine final geometry based on sampling method
                        # If the sampling method is 'buffered_point', 
                        # create a buffered point feature at the centroid of each segment,
                        # otherwise create a line feature
                        final_feature = ee.Algorithms.If(
                            ee.String(sampling_method).equals('buffered_point'),
                            # True Case: Create the buffered point feature
                            ee.Feature(
                                geom_segment.centroid(maxError).buffer(point_buffer_radius),
                                {'distance': distance}
                            ),
                            # False Case: Create the line segment feature
                            ee.Feature(geom_segment, {'distance': distance})
                        )
                        # Return either the line segment feature or the buffered point feature
                        return final_feature
                    # Creating a FeatureCollection of the cut lines with distance attributes
                    # Using map to apply the set_dist_attr function to each cut line geometry
                    line_features = ee.FeatureCollection(cut_lines_geoms.zip(distances).map(set_dist_attr))
                    # Reducing the image over the line features to get transect values
                    transect_fc = image.reduceRegions(
                        collection=line_features, reducer=ee_reducer, scale=scale
                    )
                    # Adding image date and line name properties to each feature
                    def set_props(feature):
                        return feature.set({'image_date': image_date, 'transect_name': line_name})
                    # Append to the list of all transects for this image
                    all_transects_for_image = all_transects_for_image.add(transect_fc.map(set_props))
                # Combine all transect FeatureCollections into a single FeatureCollection and flatten
                # Flatten is used to merge the list of FeatureCollections into one
                return ee.FeatureCollection(all_transects_for_image).flatten()
            # Map the function over the entire image collection and flatten the results
            results_fc = ee.FeatureCollection(self.collection.map(get_transects_for_image)).flatten()
            # Convert the results to a pandas DataFrame
            df = LandsatCollection.ee_to_df(results_fc, remove_geom=True)
            # Check if the DataFrame is empty
            if df.empty:
                print("Warning: No transect data was generated.")
                return {}
            # Initialize dictionary to hold output DataFrames for each transect
            output_dfs = {}
            # Loop through each unique transect name and create a pivot table
            for name in sorted(df['transect_name'].unique()):
                transect_df = df[df['transect_name'] == name]
                pivot_df = transect_df.pivot(index='distance', columns='image_date', values=reducer)
                pivot_df.columns.name = 'Date'
                output_dfs[name] = pivot_df
            # Optionally save each transect DataFrame to CSV
            if save_folder_path:
                for transect_name, transect_df in output_dfs.items():
                    safe_filename = "".join(x for x in transect_name if x.isalnum() or x in "._-")
                    file_path = f"{save_folder_path}{safe_filename}_transects.csv"
                    transect_df.to_csv(file_path)
                    print(f"Saved transect data to {file_path}")
            
            return output_dfs

        ### old, depreciated iterative client-side processing method ###
        elif processing_mode == 'iterative':
            if not save_folder_path:
                raise ValueError("`save_folder_path` is required for 'iterative' processing mode.")
            
            image_collection_dates = self.dates
            for i, date in enumerate(image_collection_dates):
                try:
                    print(f"Processing image {i+1}/{len(image_collection_dates)}: {date}")
                    image = self.image_grab(i)
                    transects_df = LandsatCollection.transect(
                        image, lines, line_names, reducer, n_segments, dist_interval, to_pandas=True
                    )
                    transects_df.to_csv(f"{save_folder_path}{date}_transects.csv")
                    print(f"{date}_transects saved to csv")
                except Exception as e:
                    print(f"An error occurred while processing image {i+1}: {e}")
        else:
            raise ValueError("`processing_mode` must be 'iterative' or 'aggregated'.")

    @staticmethod
    def extract_zonal_stats_from_buffer(
        image,
        coordinates,
        buffer_size=1,
        reducer_type="mean",
        scale=30,
        tileScale=1,
        coordinate_names=None,
    ):
        """
        Function to extract spatial statistics from an image for a list or single set of (long, lat) coordinates, providing individual statistics for each location.
        A radial buffer is applied around each coordinate to extract the statistics, which defaults to 1 meter.
        The function returns a pandas DataFrame with the statistics for each coordinate.

        NOTE: Be sure the coordinates are provided as longitude, latitude (x, y) tuples!

        Args:
            image (ee.Image): The image from which to extract statistics. Should be single-band.
            coordinates (list or tuple): A single (lon, lat) tuple or a list of (lon, lat) tuples.
            buffer_size (int, optional): The radial buffer size in meters. Defaults to 1.
            reducer_type (str, optional): The ee.Reducer to use ('mean', 'median', 'min', etc.). Defaults to 'mean'.
            scale (int, optional): The scale in meters for the reduction. Defaults to 30.
            tileScale (int, optional): The tile scale factor. Defaults to 1.
            coordinate_names (list, optional): A list of names for the coordinates.

        Returns:
            pd.DataFrame: A pandas DataFrame with the image's 'Date_Filter' as the index and a
                          column for each coordinate location.
        """
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            coordinates = [coordinates]
        elif not (
            isinstance(coordinates, list)
            and all(isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates)
        ):
            raise ValueError(
                "Coordinates must be a list of tuples with two elements each (longitude, latitude)."
            )

        if coordinate_names is not None:
            if not isinstance(coordinate_names, list) or not all(
                isinstance(name, str) for name in coordinate_names
            ):
                raise ValueError("coordinate_names must be a list of strings.")
            if len(coordinate_names) != len(coordinates):
                raise ValueError(
                    "coordinate_names must have the same length as the coordinates list."
                )
        else:
            coordinate_names = [f"Location {i+1}" for i in range(len(coordinates))]

        image_date = image.get('Date_Filter')

        points = [
            ee.Feature(
                ee.Geometry.Point(coord).buffer(buffer_size),
                {"location_name": str(name)},
            )
            for coord, name in zip(coordinates, coordinate_names)
        ]
        features = ee.FeatureCollection(points)

        try:
            reducer = getattr(ee.Reducer, reducer_type)()
        except AttributeError:
            raise ValueError(f"Unknown reducer_type: '{reducer_type}'.")

        stats_fc = image.reduceRegions(
            collection=features,
            reducer=reducer,
            scale=scale,
            tileScale=tileScale,
        )

        df = LandsatCollection.ee_to_df(stats_fc, remove_geom=True)

        if df.empty:
            print("Warning: No results returned. The points may not intersect the image.")
            empty_df = pd.DataFrame(columns=coordinate_names)
            empty_df.index.name = 'Date'
            return empty_df

        if reducer_type not in df.columns:
            print(f"Warning: Reducer type '{reducer_type}' not found in results. Returning raw data.")
            return df
            
        pivot_df = df.pivot(columns='location_name', values=reducer_type)
        pivot_df['Date'] = image_date.getInfo() # .getInfo() is needed here as it's a server object
        pivot_df = pivot_df.set_index('Date')
        return pivot_df

    def iterate_zonal_stats(
        self,
        geometries,
        band=None,
        reducer_type="mean",
        scale=30,
        geometry_names=None,
        buffer_size=1,
        tileScale=1,
        dates=None,
        file_path=None
    ):
        """
        Iterates over a collection of images and extracts spatial statistics (defaults to mean) for a given list of geometries or coordinates. Individual statistics are calculated for each geometry or coordinate provided.
        When coordinates are provided, a radial buffer is applied around each coordinate to extract the statistics, where the size of the buffer is determined by the buffer_size argument (defaults to 1 meter).
        The function returns a pandas DataFrame with the statistics for each coordinate and date, or optionally exports the data to a table in .csv format.

        Args:
            geometries (ee.Geometry, ee.Feature, ee.FeatureCollection, list, or tuple): Input geometries for which to extract statistics. Can be a single ee.Geometry, an ee.Feature, an ee.FeatureCollection, a list of (lon, lat) tuples, or a list of ee.Geometry objects. Be careful to NOT provide coordinates as (lat, lon)!
            band (str, optional): The name of the band to use for statistics. If None, the first band is used. Defaults to None.
            reducer_type (str, optional): The ee.Reducer to use, e.g., 'mean', 'median', 'max', 'sum'. Defaults to 'mean'. Any ee.Reducer method can be used.
            scale (int, optional): Pixel scale in meters for the reduction. Defaults to 30.
            geometry_names (list, optional): A list of string names for the geometries. If provided, must match the number of geometries. Defaults to None.
            buffer_size (int, optional): Radial buffer in meters around coordinates. Defaults to 1.
            tileScale (int, optional): A scaling factor to reduce aggregation tile size. Defaults to 1.
            dates (list, optional): A list of date strings ('YYYY-MM-DD') for filtering the collection, such that only images from these dates are included for zonal statistic retrieval. Defaults to None, which uses all dates in the collection.
            file_path (str, optional): File path to save the output CSV.

        Returns:
            pd.DataFrame or None: A pandas DataFrame with dates as the index and coordinate names
                                  as columns. Returns None if using 'iterative' mode with file_path.
        
        Raises:
            ValueError: If input parameters are invalid.
            TypeError: If geometries input type is unsupported.
        """
        img_collection_obj = self
        if band:
            img_collection_obj = LandsatCollection(collection=img_collection_obj.collection.select(band))
        else: 
            first_image = img_collection_obj.image_grab(0)
            first_band = first_image.bandNames().get(0)
            img_collection_obj = LandsatCollection(collection=img_collection_obj.collection.select([first_band]))
        # Filter collection by dates if provided
        if dates:
            img_collection_obj = LandsatCollection(
                collection=self.collection.filter(ee.Filter.inList('Date_Filter', dates))
            )

        # Initialize variables
        features = None
        validated_coordinates = [] 
        
        # Function to standardize feature names if no names are provided
        def set_standard_name(feature):
            has_geo_name = feature.get('geo_name')
            has_name = feature.get('name')
            has_index = feature.get('system:index')
            new_name = ee.Algorithms.If(
                has_geo_name, has_geo_name,
                ee.Algorithms.If(has_name, has_name,
                ee.Algorithms.If(has_index, has_index, 'unnamed_geometry')))
            return feature.set({'geo_name': new_name})

        if isinstance(geometries, (ee.FeatureCollection, ee.Feature)):
            features = ee.FeatureCollection(geometries)
            if geometry_names:
                 print("Warning: 'geometry_names' are ignored when the input is an ee.Feature or ee.FeatureCollection.")

        elif isinstance(geometries, ee.Geometry):
             name = geometry_names[0] if (geometry_names and geometry_names[0]) else 'unnamed_geometry'
             features = ee.FeatureCollection([ee.Feature(geometries).set('geo_name', name)])

        elif isinstance(geometries, list):
            if not geometries: # Handle empty list case
                raise ValueError("'geometries' list cannot be empty.")
            
            # Case: List of coordinates
            if all(isinstance(i, tuple) for i in geometries):
                validated_coordinates = geometries
                if geometry_names is None:
                    geometry_names = [f"Location_{i+1}" for i in range(len(validated_coordinates))]
                elif len(geometry_names) != len(validated_coordinates):
                     raise ValueError("geometry_names must have the same length as the coordinates list.")
                points = [
                    ee.Feature(ee.Geometry.Point(coord).buffer(buffer_size), {'geo_name': str(name)})
                    for coord, name in zip(validated_coordinates, geometry_names)
                ]
                features = ee.FeatureCollection(points)
            
            # Case: List of Geometries
            elif all(isinstance(i, ee.Geometry) for i in geometries):
                if geometry_names is None:
                    geometry_names = [f"Geometry_{i+1}" for i in range(len(geometries))]
                elif len(geometry_names) != len(geometries):
                     raise ValueError("geometry_names must have the same length as the geometries list.")
                geom_features = [
                    ee.Feature(geom).set({'geo_name': str(name)})
                    for geom, name in zip(geometries, geometry_names)
                ]
                features = ee.FeatureCollection(geom_features)
            
            else:
                raise TypeError("Input list must be a list of (lon, lat) tuples OR a list of ee.Geometry objects.")

        elif isinstance(geometries, tuple) and len(geometries) == 2:
            name = geometry_names[0] if geometry_names else 'Location_1'
            features = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point(geometries).buffer(buffer_size), {'geo_name': name})
            ])
        else:
            raise TypeError("Unsupported type for 'geometries'.")
        
        features = features.map(set_standard_name)

        try:
            reducer = getattr(ee.Reducer, reducer_type)()
        except AttributeError:
            raise ValueError(f"Unknown reducer_type: '{reducer_type}'.")

        def calculate_stats_for_image(image):
            image_date = image.get('Date_Filter')
            stats_fc = image.reduceRegions(
                collection=features, reducer=reducer, scale=scale, tileScale=tileScale
            )

            def guarantee_reducer_property(f):
                has_property = f.propertyNames().contains(reducer_type)
                return ee.Algorithms.If(has_property, f, f.set(reducer_type, -9999))
            fixed_stats_fc = stats_fc.map(guarantee_reducer_property)

            return fixed_stats_fc.map(lambda f: f.set('image_date', image_date))

        results_fc = ee.FeatureCollection(img_collection_obj.collection.map(calculate_stats_for_image)).flatten()
        df = LandsatCollection.ee_to_df(results_fc, remove_geom=True)

        # Checking for issues
        if df.empty: 
            # print("No results found for the given parameters. Check if the geometries intersect with the images, if the dates filter is too restrictive, or if the provided bands are empty.")
            # return df
            raise ValueError("No results found for the given parameters. Check if the geometries intersect with the images, if the dates filter is too restrictive, or if the provided bands are empty.")
        if reducer_type not in df.columns:
            print(f"Warning: Reducer '{reducer_type}' not found in results.")
            # return df

        # Get the number of rows before dropping nulls for a helpful message
        initial_rows = len(df)
        df.dropna(subset=[reducer_type], inplace=True)
        df = df[df[reducer_type] != -9999]
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"Warning: Discarded {dropped_rows} results due to failed reductions (e.g., no valid pixels in geometry).")

        # Reshape DataFrame to have dates as index and geometry names as columns
        pivot_df = df.pivot(index='image_date', columns='geo_name', values=reducer_type)
        pivot_df.index.name = 'Date'
        if file_path:
            # Check if file_path ends with .csv and remove it if so for consistency
            if file_path.endswith('.csv'):
                file_path = file_path[:-4]
            pivot_df.to_csv(f"{file_path}.csv")
            print(f"Zonal stats saved to {file_path}.csv")
            return
        return pivot_df
        




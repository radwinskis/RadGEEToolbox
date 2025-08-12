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
    already = ee.String(img.get('rgt:scaled')).eq('landsat_sr')
    scaled = img.select(_LS_SR_BANDS).multiply(_LS_SCALE).add(_LS_OFFSET)
    scaled = img.addBands(scaled, None, True).set('rgt:scaled','landsat_sr')
    return ee.Image(ee.Algorithms.If(already, img, scaled))

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
        self._masked_clouds_collection = None
        self._masked_water_collection = None
        self._masked_to_water_collection = None
        self._geometry_masked_collection = None
        self._geometry_masked_out_collection = None
        self._median = None
        self._mean = None
        self._max = None
        self._min = None
        self._ndwi = None
        self._mndwi = None
        self._ndvi = None
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
        and store the value as image property (matching name of chosen band).

        Args:
            image (ee.Image): input ee.Image
            band_name (string): name of band (string) for calculating area
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
            threshold (float): integer threshold to specify masking of pixels below threshold (defaults to -1)
            scale (int): integer scale of image resolution (meters) (defaults to 30)
            maxPixels (int): integer denoting maximum number of pixels for calculations

        Returns:
            ee.Image: ee.Image with area calculation stored as property matching name of band
        """
        area_image = ee.Image.pixelArea()
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

    def PixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12
    ):
        """
        Calculates the summation of area for pixels of interest (above a specific threshold)
        within a geometry and store the value as image property (matching name of chosen band) for an entire
        image collection.

        The resulting value has units of square meters.

        Args:
            band_name (string): name of band (string) for calculating area
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
            threshold (float): integer threshold to specify masking of pixels below threshold (defaults to -1)
            scale (int): integer scale of image resolution (meters) (defaults to 30)
            maxPixels (int): integer denoting maximum number of pixels for calculations

        Returns:
            ee.ImageCollection: Image with area calculation stored as property matching name of band.
        """
        if self._PixelAreaSumCollection is None:
            collection = self.collection
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
            self._PixelAreaSumCollection = AreaCollection
        return self._PixelAreaSumCollection

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
        """Converts an ee.FeatureCollection to pandas dataframe. Adapted from the geemap package (https://geemap.org/common/#geemap.common.ee_to_df)

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
        """Extracts transect from an image. Adapted from the geemap package (https://geemap.org/common/#geemap.common.extract_transect). Exists as an alternative to RadGEEToolbox 'transect' function.

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
        """Computes and stores the values along a transect for each line in a list of lines. Builds off of the extract_transect function from the geemap package
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
        save_folder_path,
        reducer="mean",
        n_segments=None,
        dist_interval=30,
        to_pandas=True,
    ):
        """Computes and stores the values along a transect for each line in a list of lines for each image in a LandsatCollection image collection, then saves the data for each image to a csv file. Builds off of the extract_transect function from the geemap package
            where checks are ran to ensure that the reducer column is present in the transect data. If the reducer column is not present, a column of NaNs is created.
            An ee reducer is used to aggregate the values along the transect, depending on the number of segments or distance interval specified. Defaults to 'mean' reducer.
            Naming conventions for the csv files follows as: "image-date_transects.csv"

        Args:
            lines (list): List of ee.Geometry.LineString objects.
            line_names (list of strings): List of line string names.
            save_folder_path (str): The path to the folder where the csv files will be saved.
            reducer (str): The ee reducer to use. Defaults to 'mean'.
            n_segments (int): The number of segments that the LineString will be split into. Defaults to None.
            dist_interval (float): The distance interval used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 10.
            to_pandas (bool): Whether to convert the result to a pandas dataframe. Defaults to True.

        Raises:
            Exception: If the program fails to compute.

        Returns:
            csv file: file for each image with an organized list of values along the transect(s)
        """
        image_collection = self  # .collection
        # image_collection_dates = self._dates
        image_collection_dates = self.dates
        for i, date in enumerate(image_collection_dates):
            try:
                print(f"Processing image {i+1}/{len(image_collection_dates)}: {date}")
                image = image_collection.image_grab(i)
                transects_df = LandsatCollection.transect(
                    image,
                    lines,
                    line_names,
                    reducer=reducer,
                    n_segments=n_segments,
                    dist_interval=dist_interval,
                    to_pandas=to_pandas,
                )
                image_id = date
                transects_df.to_csv(f"{save_folder_path}{image_id}_transects.csv")
                print(f"{image_id}_transects saved to csv")
            except Exception as e:
                print(f"An error occurred while processing image {i+1}: {e}")

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
        Function to extract spatial statistics from an image for a list of coordinates, providing individual statistics for each location.
        A radial buffer is applied around each coordinate to extract the statistics, which defaults to 1 meter.
        The function returns a pandas DataFrame with the statistics for each coordinate.

        Args:
            image (ee.Image): The image from which to extract the statistics. Must be a singleband image or else resulting values will all be zero!
            coordinates (list): Single tuple or list of tuples with the decimal degrees coordinates in the format of (longitude, latitude) for which to extract the statistics. NOTE the format needs to be [(x1, y1), (x2, y2), ...].
            buffer_size (int, optional): The radial buffer size around the coordinates in meters. Defaults to 1.
            reducer_type (str, optional): The reducer type to use. Defaults to 'mean'. Options are 'mean', 'median', 'min', and 'max'.
            scale (int, optional): The scale to use. Defaults to 30.
            tileScale (int, optional): The tile scale to use. Defaults to 1.
            coordinate_names (list, optional): A list of strings with the names of the coordinates. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame with the statistics for each coordinate, each column name corresponds to the name of the coordinate feature (which may be blank if no names are supplied).
        """

        # Check if coordinates is a single tuple and convert it to a list of tuples if necessary
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            coordinates = [coordinates]
        elif not (
            isinstance(coordinates, list)
            and all(
                isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates
            )
        ):
            raise ValueError(
                "Coordinates must be a list of tuples with two elements each (latitude, longitude)."
            )

        # Check if coordinate_names is a list of strings
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

        # Check if the image is a singleband image
        def check_singleband(image):
            band_count = image.bandNames().size()
            return ee.Algorithms.If(band_count.eq(1), image, ee.Image.constant(0))

        # image = ee.Image(check_singleband(image))
        image = ee.Image(check_singleband(image))

        # Convert coordinates to ee.Geometry.Point, buffer them, and add label/name to feature
        points = [
            ee.Feature(
                ee.Geometry.Point([coord[0], coord[1]]).buffer(buffer_size),
                {"name": str(coordinate_names[i])},
            )
            for i, coord in enumerate(coordinates)
        ]
        # Create a feature collection from the buffered points
        features = ee.FeatureCollection(points)
        # Reduce the image to the buffered points - handle different reducer types
        if reducer_type == "mean":
            img_stats = image.reduceRegions(
                collection=features,
                reducer=ee.Reducer.mean(),
                scale=scale,
                tileScale=tileScale,
            )
            mean_values = img_stats.getInfo()
            means = []
            names = []
            for feature in mean_values["features"]:
                names.append(feature["properties"]["name"])
                means.append(feature["properties"]["mean"])
            organized_values = pd.DataFrame([means], columns=names)
        elif reducer_type == "median":
            img_stats = image.reduceRegions(
                collection=features,
                reducer=ee.Reducer.median(),
                scale=scale,
                tileScale=tileScale,
            )
            median_values = img_stats.getInfo()
            medians = []
            names = []
            for feature in median_values["features"]:
                names.append(feature["properties"]["name"])
                medians.append(feature["properties"]["median"])
            organized_values = pd.DataFrame([medians], columns=names)
        elif reducer_type == "min":
            img_stats = image.reduceRegions(
                collection=features,
                reducer=ee.Reducer.min(),
                scale=scale,
                tileScale=tileScale,
            )
            min_values = img_stats.getInfo()
            mins = []
            names = []
            for feature in min_values["features"]:
                names.append(feature["properties"]["name"])
                mins.append(feature["properties"]["min"])
            organized_values = pd.DataFrame([mins], columns=names)
        elif reducer_type == "max":
            img_stats = image.reduceRegions(
                collection=features,
                reducer=ee.Reducer.max(),
                scale=scale,
                tileScale=tileScale,
            )
            max_values = img_stats.getInfo()
            maxs = []
            names = []
            for feature in max_values["features"]:
                names.append(feature["properties"]["name"])
                maxs.append(feature["properties"]["max"])
            organized_values = pd.DataFrame([maxs], columns=names)
        else:
            raise ValueError(
                "reducer_type must be one of 'mean', 'median', 'min', or 'max'."
            )
        return organized_values

    def iterate_zonal_stats(
        self,
        coordinates,
        buffer_size=1,
        reducer_type="mean",
        scale=30,
        tileScale=1,
        coordinate_names=None,
        file_path=None,
        dates=None,
    ):
        """
        Function to iterate over a collection of images and extract spatial statistics for a list of coordinates (defaults to mean). Individual statistics are provided for each location.
        A radial buffer is applied around each coordinate to extract the statistics, which defaults to 1 meter.
        The function returns a pandas DataFrame with the statistics for each coordinate and date, or optionally exports the data to a table in .csv format.

        Args:
            coordinates (list): Single tuple or a list of tuples with the coordinates as decimal degrees in the format of (longitude, latitude) for which to extract the statistics. NOTE the format needs to be [(x1, y1), (x2, y2), ...].
            buffer_size (int, optional): The radial buffer size in meters around the coordinates. Defaults to 1.
            reducer_type (str, optional): The reducer type to use. Defaults to 'mean'. Options are 'mean', 'median', 'min', and 'max'.
            scale (int, optional): The scale (pixel size) to use in meters. Defaults to 30.
            tileScale (int, optional): The tile scale to use. Defaults to 1.
            coordinate_names (list, optional): A list of strings with the names of the coordinates. Defaults to None.
            file_path (str, optional): The file path to export the data to. Defaults to None. Ensure ".csv" is NOT included in the file name path.
            dates (list, optional): A list of dates for which to extract the statistics. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame with the statistics for each coordinate and date, each row corresponds to a date and each column to a coordinate.
            .csv file: Optionally exports the data to a table in .csv format. If file_path is None, the function returns the DataFrame - otherwise the function will only export the csv file.
        """
        img_collection = self
        # Create empty DataFrame to accumulate results
        accumulated_df = pd.DataFrame()
        # Check if dates is None, if not use the dates provided
        if dates is None:
            dates = img_collection.dates
        else:
            dates = dates
        # Iterate over the dates and extract the zonal statistics for each date
        for date in dates:
            image = img_collection.collection.filter(
                ee.Filter.eq("Date_Filter", date)
            ).first()
            single_df = LandsatCollection.extract_zonal_stats_from_buffer(
                image,
                coordinates,
                buffer_size=buffer_size,
                reducer_type=reducer_type,
                scale=scale,
                tileScale=tileScale,
                coordinate_names=coordinate_names,
            )
            single_df["Date"] = date
            single_df.set_index("Date", inplace=True)
            accumulated_df = pd.concat([accumulated_df, single_df])
        # Return the DataFrame or export the data to a .csv file
        if file_path is None:
            return accumulated_df
        else:
            return accumulated_df.to_csv(f"{file_path}.csv")

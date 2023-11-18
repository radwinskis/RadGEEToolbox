import geemap
import ee
class LandsatCollection:
    """
    Class object representing a combined collection of NASA/USGS Landsat 5, 8, and 9 TM & OLI surface reflectance satellite images at 30 m/px

    This class provides methods to filter, process, and analyze Landsat satellite imagery for a given period and region

    Arguments:
        start_date (str): Start date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)

        end_date (str): End date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)

        tile_row (int or list): WRS-2 tile row of Landsat image (required unless boundary or collection is provided) - see https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files

        tile_path (int or list): WRS-2 tile path of Landsat image (required unless boundary or collection is provided) - see https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files

        cloud_percentage_threshold (int): Integer percentage threshold where only imagery with cloud % less than threshold will be provided (defaults to 100)

        boundary (ee.Geometry): Boundary for filtering images to images that intersect with the boundary shape (optional) - will override tile specifications
        
        collection (ee.ImageCollection): Optional argument to provide an ee.ImageCollection object to be converted to a LandsatCollection object - will override other arguments!

    Attributes:
        collection: Returns an ee.ImageCollection object from any LandsatCollection image collection object
        
        _dates_list: Cache storage for dates_list property attribute

        _dates: Cahce storgage for dates property attribute
        
        ndwi_threshold: Default threshold for masking ndwi imagery
        
        ndvi_threshold: Default threshold for masking ndvi imagery
        
        halite_threshold: Default threshold for masking halite imagery
        
        gypsum_threshold: Default threshold for masking gypsum imagery

        turbidity_threshold: Default threshold for masking turbidity imagery

        chlorophyll_threshold: Default threshold for masking chlorophyll imagery

        _masked_clouds_collection: Cache storage for masked_clouds_collection property attribute

        _masked_water_collection: Cache storage for masked_water_collection property attribute

        _masked_to_water_collection: Cache storage for masked_to_water_collection property attribute

        _geometry_masked_collection: Cache storage for mask_to_polygon method

        _geometry_masked_out_collection: Cache storage for mask_out_polygon method

        _median: Cache storage for median property attribute

        _mean: Cache storage for mean property attribute
        
        _max: Cache storage for max property attribute

        _min: Cache storage for min property attribute

        _ndwi: Cache storage for ndwi property attribute

        _ndvi: Cache storage for ndvi property attribute

        _halite: Cache storage for halite property attribute

        _gypsum: Cache storage for gypsum property attribute

        _turbidity: Cache storage for turbidity property attribute

        _chlorophyll: Cache storage for chlorophyll property attribute

        _LST: Cache storage for LST property attribute

        _MosaicByDate: Cache storage for MosaicByDate property attribute

    Property attributes:
        dates_list (returns: Server-Side List): Unreadable Earth Engine list of image dates (server-side)
        
        dates (returns: Client-Side List): Readable pythonic list of image dates (client-side)
        
        masked_clouds_collection (returns: LandsatCollection image collection): Returns collection with clouds masked (transparent) for each image

        masked_water_collection (returns: LandsatCollection image collection): Returns collection with water pixels masked (transparent) for each image

        masked_to_water_collection (returns: LandsatCollection image collection): Returns collection with pixels masked to water (transparent) for each image (masks land and cloud pixels)

        max (returns: ee.Image): Returns a temporally reduced max image (calculates max at each pixel)
        
        median (returns: ee.Image): Returns a temporally reduced median image (calculates median at each pixel)
        
        mean (returns: ee.Image): Returns a temporally reduced mean image (calculates mean at each pixel)
        
        min (returns: ee.Image): Returns a temporally reduced min image (calculates min at each pixel)
        
        MosaicByDate (returns: LandsatCollection image collection): Mosaics image collection where images with the same date are mosaiced into the same image. Calculates total cloud percentage for subsequent filtering of cloudy mosaics.
        
        gypsum (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband gypsum index rasters
        
        halite (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband halite index rasters
        
        LST (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband land-surface-temperature rasters (Celcius)
        
        ndwi (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband NDWI (water) rasters
        
        ndvi (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband NDVI (vegetation) rasters

        turbidity (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband NDTI (turbidity) rasters

        chlorophyll (returns: ee.ImageCollection): Returns LandsatCollection image collection of singleband KIVU (relative chlorophyll-a) rasters

    Methods:
        get_filtered_collection(self)

        get_boundary_filtered_collection(self)
        
        ndwi_collection(self, threshold, ng_threshold=None)
        
        ndvi_collection(self, threshold, ng_threshold=None)
        
        halite_collection(self, threshold, ng_threshold=None)
        
        gypsum_collection(self, threshold, ng_threshold=None)

        turbidity_collection(self, threshold, ng_threshold=None)

        chlorophyll_collection(self, threshold, ng_threshold=None)

        masked_water_collection_NDWI(self, threshold)

        masked_to_water_collection_NDWI(self, threshold)
        
        surface_temperature_collection(self)
        
        mask_to_polygon(self, polygon)

        mask_out_polygon(self, polygon)
        
        mask_halite(self, threshold, ng_threshold=None)
        
        mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None)
        
        image_grab(self, img_selector)
        
        custom_image_grab(self, img_col, img_selector)
        
        image_pick(self, img_date)
        
        CollectionStitch(self, img_col2)

    Static Methods:
        image_dater(image)
        
        landsat5bandrename(img)
        
        landsat_ndwi_fn(image, threshold, ng_threshold=None)
        
        landsat_ndvi_fn(image, threshold, ng_threshold=None)
        
        landsat_halite_fn(image, threshold, ng_threshold=None)
        
        landsat_gypsum_fn(image, threshold, ng_threshold=None)

        landsat_ndti_fn(image, threshold, ng_threshold=None)

        landsat_kivu_chla_fn(image, threshold, ng_threshold=None)
        
        MaskWaterLandsat(image)

        MaskToWaterLandsat(image)

        MaskWaterLandsatByNDWI(image, threshold, ng_threshold=None)

        MaskToWaterLandsatByNDWI(image, threshold, ng_threshold=None)
        
        halite_mask(image, threshold, ng_threshold=None)
        
        gypsum_and_halite_mask(image, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None)
        
        maskL8clouds(image)
        
        temperature_bands(img)
        
        landsat_LST(image)
        
        PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12)
        
        dNDWIPixelAreaSum(image, geometry, band_name='ndwi', scale=30, maxPixels=1e12)

    Usage:
        The LandsatCollection object alone acts as a base object for which to further filter or process to indices or spatial reductions
        
        To use the LandsatCollection functionality, use any of the built in class attributes or method functions. For example, using class attributes:
       
        image_collection = LandsatCollection(start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

        ee_image_collection = image_collection.collection #returns ee.ImageCollection from provided argument filters

        latest_image = image_collection.image_grab(-1) #returns latest image in collection as ee.Image

        cloud_masked_collection = image_collection.masked_clouds_collection #returns cloud-masked LandsatCollection image collection

        NDWI_collection = image_collection.ndwi #returns NDWI LandsatCollection image collection

        latest_NDWI_image = NDWI_collection.image_grab(-1) #Example showing how class functions work with any LandsatCollection image collection object, returning latest ndwi image
    """
    def __init__(self, start_date=None, end_date=None, tile_row=None, tile_path=None, cloud_percentage_threshold=None, boundary=None,  collection=None):
        if collection is None and (start_date is None or end_date is None):
            raise ValueError("Either provide all required fields (start_date, end_date, tile_row, tile_path ; or boundary in place of tiles) or provide a collection.")
        if tile_row is None and tile_path is None and boundary is None and collection is None:
            raise ValueError("Provide either tile or boundary/geometry specifications to filter the image collection")
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

        
        self._dates_list = None
        self._dates = None
        self.ndwi_threshold = -1
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
        self._ndvi = None
        self._halite = None
        self._gypsum = None
        self._turbidity = None
        self._chlorophyll = None
        self._LST = None
        self._MosaicByDate = None

    @staticmethod
    def image_dater(image):
        """
        Adds date to image properties as 'Date_Filter'.

        Args: 
        image (ee.Image): Input image

        Returns: 
        image (ee.Image): Image with date in properties.
        """
        date = ee.Number(image.date().format('YYYY-MM-dd'))
        return image.set({'Date_Filter': date})
    
    @staticmethod
    def landsat5bandrename(img):
        """
        Function to rename Landsat 5 bands to match Landsat 8 & 9.

        Args: 
        image (ee.Image): input image
        
        Returns: 
        image (ee.Image): image with renamed bands
        """
        return img.select('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL').rename('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL')
    
    @staticmethod
    def landsat_ndwi_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate ndwi for Landsat imagery and mask image based on threshold. 
        Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold 
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): ndwi image
        """
        ndwi_calc = image.normalizedDifference(['SR_B3', 'SR_B5']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        if ng_threshold != None:
            water = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image).set('threshold', threshold), \
                                        ndwi_calc.updateMask(ndwi_calc.gte(ng_threshold)).rename('ndwi').copyProperties(image).set('threshold', ng_threshold))
        else:
            water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image)
        return water

    @staticmethod
    def landsat_ndvi_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate ndvi for Landsat imagery and mask image based on threshold.
        Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): ndvi ee.Image
        """
        ndvi_calc = image.normalizedDifference(['SR_B5', 'SR_B4']) #NIR-RED/NIR+RED -- full NDVI image
        if ng_threshold != None:
            vegetation = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      ndvi_calc.updateMask(ndvi_calc.gte(threshold)).rename('ndvi').copyProperties(image).set('threshold', threshold), \
                                        ndvi_calc.updateMask(ndvi_calc.gte(ng_threshold)).rename('ndvi').copyProperties(image).set('threshold', ng_threshold))
        else:
            vegetation = ndvi_calc.updateMask(ndvi_calc.gte(threshold)).rename('ndvi').copyProperties(image)
        return vegetation
    
    @staticmethod
    def landsat_halite_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate multispectral halite index for Landsat imagery and mask image based on threshold.
        Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): halite ee.Image
        """
        halite_index = image.normalizedDifference(['SR_B4', 'SR_B6'])
        if ng_threshold != None:
            halite = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      halite_index.updateMask(halite_index.gte(threshold)).rename('halite').copyProperties(image).set('threshold', threshold), \
                                        halite_index.updateMask(halite_index.gte(ng_threshold)).rename('halite').copyProperties(image).set('threshold', ng_threshold))
        else:
            halite = halite_index.updateMask(halite_index.gte(threshold)).rename('halite').copyProperties(image)
        return halite 
      
    @staticmethod
    def landsat_gypsum_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate multispectral gypsum index for Landsat imagery and mask image based on threshold.
        Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): gypsum ee.Image
        """
        gypsum_index = image.normalizedDifference(['SR_B6', 'SR_B7'])
        if ng_threshold != None:
            gypsum = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      gypsum_index.updateMask(gypsum_index.gte(threshold)).rename('gypsum').copyProperties(image).set('threshold', threshold), \
                                        gypsum_index.updateMask(gypsum_index.gte(ng_threshold)).rename('gypsum').copyProperties(image).set('threshold', ng_threshold))
        else:
            gypsum = gypsum_index.updateMask(gypsum_index.gte(threshold)).rename('gypsum').copyProperties(image)
        return gypsum
    
    @staticmethod
    def landsat_ndti_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate turbidity of water pixels using Normalized Difference Turbidity Index (NDTI; Lacaux et al., 2007) 
        and mask image based on threshold. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): turbidity ee.Image
        """
        NDTI = image.normalizedDifference(['SR_B4', 'SR_B3'])
        if ng_threshold != None:
            turbidity = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      NDTI.updateMask(NDTI.gte(threshold)).rename('ndti').copyProperties(image).set('threshold', threshold), \
                                        NDTI.updateMask(NDTI.gte(ng_threshold)).rename('ndti').copyProperties(image).set('threshold', ng_threshold))
        else:
            turbidity = NDTI.updateMask(NDTI.gte(threshold)).rename('ndti').copyProperties(image)
        return turbidity
    
    @staticmethod
    def landsat_kivu_chla_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate relative chlorophyll-a concentrations of water pixels using 3BDA/KIVU index 
        (see Boucher et al., 2018 for review) and mask image based on threshold. Can specify separate thresholds 
        for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 and the ng_threshold
        argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image (ee.Image): chlorophyll-a ee.Image
        """
        KIVU = image.expression('(BLUE - RED) / GREEN', {'BLUE':image.select('SR_B2'), 'RED':image.select('SR_B4'), 'GREEN':image.select('SR_B3')})
        if ng_threshold != None:
            chlorophyll = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      KIVU.updateMask(KIVU.gte(threshold)).rename('kivu').copyProperties(image).set('threshold', threshold), \
                                        KIVU.updateMask(KIVU.gte(ng_threshold)).rename('kivu').copyProperties(image).set('threshold', ng_threshold))
        else:
            chlorophyll = KIVU.updateMask(KIVU.gte(threshold)).rename('kivu').copyProperties(image)
        return chlorophyll


    @staticmethod
    def MaskWaterLandsat(image):
        """
        Function to mask water pixels based on Landsat image QA band.

        Args:
        image (ee.Image): input ee.Image

        Returns:
        image (ee.Image): ee.Image with water pixels masked.
        """
        WaterBitMask = ee.Number(2).pow(7).int()
        qa = image.select('QA_PIXEL')
        water_extract = qa.bitwiseAnd(WaterBitMask).eq(0)
        masked_image = image.updateMask(water_extract).copyProperties(image)
        return masked_image
    
    @staticmethod
    def MaskWaterLandsatByNDWI(image, threshold, ng_threshold=None):
        """
        Function to mask water pixels (mask land and cloud pixels) for all bands based on NDWI and a set threshold where
        all pixels less than NDWI threshold are masked out. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where NDWI pixels greater than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where NDWI pixels greater than threshold are masked
        
        Returns:
        image (ee.Image): ee.Image with water pixels masked
        """
        ndwi_calc = image.normalizedDifference(['SR_B3', 'SR_B5']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        if ng_threshold != None:
            water = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      image.updateMask(ndwi_calc.lt(threshold)).set('threshold', threshold), \
                                        image.updateMask(ndwi_calc.lt(ng_threshold)).set('threshold', ng_threshold))
        else:
            water = image.updateMask(ndwi_calc.lt(threshold)).set('threshold', threshold)
        return water
    
    @staticmethod
    def MaskToWaterLandsat(image):
        """
        Function to mask to water pixels by masking land and cloud pixels based on Landsat image QA band.

        Args:
        image (ee.Image): input ee.Image

        Returns:
        image (ee.Image): ee.Imagewith water pixels masked.
        """
        WaterBitMask = ee.Number(2).pow(7).int()
        qa = image.select('QA_PIXEL')
        water_extract = qa.bitwiseAnd(WaterBitMask).neq(0)
        masked_image = image.updateMask(water_extract).copyProperties(image)
        return masked_image
    
    @staticmethod
    def MaskToWaterLandsatByNDWI(image, threshold, ng_threshold=None):
        """
        Function to mask water pixels using NDWI based on threshold. Can specify separate thresholds for Landsat 5 vs 8&9 images, where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where NDWI pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where NDWI pixels less than threshold are masked
        
        Returns:
        image (ee.Image): ee.Image with water pixels masked.
        """
        ndwi_calc = image.normalizedDifference(['SR_B3', 'SR_B5']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        if ng_threshold != None:
            water = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      image.updateMask(ndwi_calc.gte(threshold)).set('threshold', threshold), \
                                        image.updateMask(ndwi_calc.gte(ng_threshold)).set('threshold', ng_threshold))
        else:
            water = image.updateMask(ndwi_calc.gte(threshold)).set('threshold', threshold)
        return water

    @staticmethod
    def halite_mask(image, threshold, ng_threshold=None):
        """
        Function to mask halite pixels after specifying index to isolate/mask-to halite pixels. 
        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold
        argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked, applies to landsat 5 when ng_threshold is also set.
        ng_threshold (int | optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked 

        Returns:
        image (ee.Image): masked ee.Image
        """
        halite_index = image.normalizedDifference(['SR_B4', 'SR_B6']) # red-swir1 / red+swir1
        if ng_threshold != None:
            mask = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      image.updateMask(halite_index.lt(threshold)).copyProperties(image), \
                                        image.updateMask(halite_index.lt(ng_threshold)).copyProperties(image)) 
        else:
            mask = image.updateMask(halite_index.lt(threshold)).copyProperties(image)
        return mask 
    
    @staticmethod
    def gypsum_and_halite_mask(image, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None):
        """
        Function to mask both gypsum and halite pixels. Must specify threshold for isolating halite and gypsum pixels. 
        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold argument applies to Landsat 5 
        and the ng_threshold argument applies to Landsat 8&9.

        Args:
        image (ee.Image): input ee.Image
        halite_threshold (int): integer threshold for halite where pixels less than threshold are masked, applies to landsat 5 when ng_threshold is also set.
        gypsum_threshold (int): integer threshold for gypsum where pixels less than threshold are masked, applies to landsat 5 when ng_threshold is also set.
        halite_ng_threshold (int | optional): integer threshold for halite to be applied to landsat 8 or 9 where pixels less than threshold are masked 
        gypsum_ng_threshold (int | optional): integer threshold for gypsum to be applied to landsat 8 or 9 where pixels less than threshold are masked 

        Returns:
        image (ee.Image): masked ee.Image
        """
        halite_index = image.normalizedDifference(['SR_B4', 'SR_B6']) # red-swir1 / red+swir1
        gypsum_index = image.normalizedDifference(['SR_B6', 'SR_B7'])
        if halite_ng_threshold and gypsum_ng_threshold != None:
            mask = ee.Algorithms.If(ee.String(image.get('SPACECRAFT_ID')).equals('LANDSAT_5'), \
                                      gypsum_index.updateMask(halite_index.lt(halite_threshold)).updateMask(gypsum_index.lt(gypsum_threshold)).rename('carbonate_muds').copyProperties(image), \
                                        gypsum_index.updateMask(halite_index.lt(halite_ng_threshold)).updateMask(gypsum_index.lt(gypsum_ng_threshold)).rename('carbonate_muds').copyProperties(image))
        else:
            mask = gypsum_index.updateMask(halite_index.lt(halite_threshold)).updateMask(gypsum_index.lt(gypsum_threshold)).rename('carbonate_muds').copyProperties(image)
        return mask

    @staticmethod
    def maskL8clouds(image):
        """
        Function to mask clouds baseed on Landsat 8 QA band.

        Args:
        image (ee.Image): input ee.Image

        Returns:
        image (ee.Image): ee.Image
        """
        cloudBitMask = ee.Number(2).pow(3).int()
        CirrusBitMask = ee.Number(2).pow(2).int()
        qa = image.select('QA_PIXEL')
        cloud_mask = qa.bitwiseAnd(cloudBitMask).eq(0)
        cirrus_mask = qa.bitwiseAnd(CirrusBitMask).eq(0)
        return image.updateMask(cloud_mask).updateMask(cirrus_mask)
    
    @staticmethod
    def temperature_bands(img):
        """
        Function to rename bands for temperature calculations.

        Args:
        img: input ee.Image

        Returns:
        image (ee.Image): ee.Image
        """
        #date = ee.Number(img.date().format('YYYY-MM-dd'))
        scale1 = ['ST_ATRAN', 'ST_EMIS']
        scale2 = ['ST_DRAD', 'ST_TRAD', 'ST_URAD']
        scale1_names = ['transmittance', 'emissivity']
        scale2_names = ['downwelling', 'B10_radiance', 'upwelling']
        scale1_bands = img.select(scale1).multiply(0.0001).rename(scale1_names) #Scaled to new L8 collection
        scale2_bands = img.select(scale2).multiply(0.001).rename(scale2_names) #Scaled to new L8 collection
        return img.addBands(scale1_bands).addBands(scale2_bands).copyProperties(img)
    
    @staticmethod
    def landsat_LST(image):
        """
        Function to calculate land surface temperature (LST) from landsat TIR bands. 
        Based on Sekertekin, A., & Bonafoni, S. (2020) https://doi.org/10.3390/rs12020294

        Args:
        image (ee.Image): input ee.Image

        Returns:
        image (ee.Image): LST ee.Image 
        """
        # Based on Sekertekin, A., & Bonafoni, S. (2020) https://doi.org/10.3390/rs12020294
        
        k1 = 774.89
        k2 = 1321.08
        LST = image.expression(
            '(k2/log((k1/((B10_rad - upwelling - transmittance*(1 - emissivity)*downwelling)/(transmittance*emissivity)))+1)) - 273.15',
            {'k1': k1,
            'k2': k2,
            'B10_rad': image.select('B10_radiance'),
            'upwelling': image.select('upwelling'),
            'transmittance': image.select('transmittance'),
            'emissivity': image.select('emissivity'),
            'downwelling': image.select('downwelling')}).rename('LST')
        return image.addBands(LST).copyProperties(image) #Outputs temperature in C
    
    @staticmethod
    def PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) in a geometry
        and store the value as image property (matching name of chosen band).

        Args:
        image (ee.Image): input ee.Image
        band_name (string): name of band (string) for calculating area
        geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
        threshold (int): integer threshold to specify masking of pixels below threshold (defaults to -1)
        scale (int): integer scale of image resolution (meters) (defaults to 30)
        maxPixels (int): integer denoting maximum number of pixels for calculations
        
        Returns:
        image (ee.Image): ee.Image with area calculation stored as property matching name of band
        """
        area_image = ee.Image.pixelArea()
        mask = image.select(band_name).gte(threshold)
        final = image.addBands(area_image)
        stats = final.select('area').updateMask(mask).rename(band_name).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry= geometry,
            scale=scale,
            maxPixels = maxPixels)
        return image.set(band_name, stats.get(band_name))

    @staticmethod
    def dNDWIPixelAreaSum(image, geometry, band_name='ndwi', scale=30, maxPixels=1e12):
        """
        Function to dynamically calulate the summation of area for water pixels of interest and store the value as image property named 'ndwi'
        Uses Otsu thresholding to dynamically choose the best threshold rather than needing to specify threshold.
        Note: An offset of 0.15 is added to the Otsu threshold.

        Args:
        image (ee.Image): input ee.Image
        geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation
        band_name (string): name of ndwi band (string) for calculating area (defaults to 'ndwi')
        scale (int): integer scale of image resolution (meters) (defaults to 30)
        maxPixels (int): integer denoting maximum number of pixels for calculations

        Returns:
        image (ee.Image): ee.Image with area calculation stored as property matching name of band
        """
        def OtsuThreshold(histogram):
            counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
            means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
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
                    bCount.multiply(bMean.subtract(mean).pow(2)))

            bss = indices.map(func_xxx)
            return means.sort(bss).get([-1])

        area_image = ee.Image.pixelArea()
        histogram = image.select(band_name).reduceRegion(
            reducer = ee.Reducer.histogram(255, 2),
            geometry = geometry.geometry().buffer(6000),
            scale = scale,
            bestEffort= True,)
        threshold = OtsuThreshold(histogram.get(band_name)).add(0.15)
        mask = image.select(band_name).gte(threshold)
        final = image.addBands(area_image)
        stats = final.select('area').updateMask(mask).rename(band_name).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry= geometry,
            scale=scale,
            maxPixels = maxPixels)
        return image.set(band_name, stats.get(band_name))
    
    @property
    def dates_list(self):
        """
        Property attribute to retrieve list of dates as server-side (GEE) object.

        Args:
        self: self is passed into argument.

        Returns:
        ee.List: Server-side ee.List of dates.
        """
        if self._dates_list is None:
            dates = self.collection.aggregate_array('Date_Filter')
            self._dates_list = dates
        return self._dates_list

    @property
    def dates(self):
        """
        Property attribute to retrieve list of dates as readable and indexable client-side list object.

        Args:
        self: self is passed into argument.

        Returns:
        list: list of date strings.
        """
        if self._dates_list is None:
            dates = self.collection.aggregate_array('Date_Filter')
            self._dates_list = dates
        if self._dates is None:
            dates = self._dates_list.getInfo()
            self._dates = dates
        return self._dates

    def get_filtered_collection(self):
        """
        Function to filter image collection based on LandsatCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Args:
        self: self is passed into argument

        Returns:
        ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from LandsatCollection object
        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(LandsatCollection.landsat5bandrename)  # Replace with the correct Landsat 5 collection ID
        filtered_collection = landsat8.merge(landsat9).merge(landsat5).filterDate(self.start_date, self.end_date).filter(ee.Filter.And(ee.Filter.inList('WRS_PATH', self.tile_path),
                                ee.Filter.inList('WRS_ROW', self.tile_row))).filter(ee.Filter.lte('CLOUD_COVER', self.cloud_percentage_threshold)).map(LandsatCollection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_boundary_filtered_collection(self):
        """
        Function to filter and mask image collection based on LandsatCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Args:
        self: self is passed into argument

        Returns:
        ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from LandsatCollection object

        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(LandsatCollection.landsat5bandrename)  # Replace with the correct Landsat 5 collection ID
        filtered_collection = landsat8.merge(landsat9).merge(landsat5).filterDate(self.start_date, self.end_date).filterBounds(self.boundary).filter(ee.Filter.lte('CLOUD_COVER', self.cloud_percentage_threshold)).map(LandsatCollection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    @property
    def median(self):
        """
        Property attribute function to calculate median image from image collection. Results are calculated once per class object then cached for future use.

        Args:
        self: self is passed into argument

        Returns:
        image (ee.Image): median image from entire collection.
        """
        if self._median is None:
            col = self.collection.median()
            self._median = col
        return self._median
    
    @property
    def mean(self):
        """
        Property attribute function to calculate mean image from image collection. Results are calculated once per class object then cached for future use.

        Args:
        self: self is passed into argument

        Returns:
        image (ee.Image): mean image from entire collection.

        """
        if self._mean is None:
            col = self.collection.mean()
            self._mean = col
        return self._mean
    
    @property
    def max(self):
        """
        Property attribute function to calculate max image from image collection. Results are calculated once per class object then cached for future use.

        Args:
        self: self is passed into argument

        Returns:
        image (ee.Image): max image from entire collection.
        """
        if self._max is None:
            col = self.collection.max()
            self._max = col
        return self._max
    
    @property
    def min(self):
        """
        Property attribute function to calculate min image from image collection. Results are calculated once per class object then cached for future use.
        
        Args:
        self: self is passed into argument

        Returns:
        image (ee.Image): min image from entire collection.
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

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        if self._ndwi is None:
            self._ndwi = self.ndwi_collection(self.ndwi_threshold)
        return self._ndwi

    def ndwi_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate ndwi and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called 
        by default when using the ndwi property attribute. 

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for NDWI function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()

        if available_bands.contains('SR_B3') and available_bands.contains('SR_B5'):
            pass
        else:
            raise ValueError("Insufficient Bands for ndwi calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_ndwi_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)
    
    @property
    def ndvi(self):
        """
        Property attribute to calculate and access the NDVI (Normalized Difference Vegetation Index) imagery of the LandsatCollection. 
        This property initiates the calculation of NDVI using a default threshold of -1 (or a previously set threshold of self.ndvi_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
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
        self: self is passed into argument
        threshold (int): specify threshold for NDVI function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('SR_B4') and available_bands.contains('SR_B5'):
            pass
        else:
            raise ValueError("Insufficient Bands for ndwi calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_ndvi_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)

    @property
    def halite(self):
        """
        Property attribute to calculate and access the halite index (see Radwin & Bowen, 2021) imagery of the LandsatCollection. 
        This property initiates the calculation of halite using a default threshold of -1 (or a previously set threshold of self.halite_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
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
        self: self is passed into argument
        threshold (int): specify threshold for halite function (values less than threshold are masked)
        ng_threshold (int): (optional) specify threshold for Landsat 8&9 halite function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('SR_B4') and available_bands.contains('SR_B6'):
            pass
        else:
            raise ValueError("Insufficient Bands for halite calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_halite_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)

    @property
    def gypsum(self):
        """
        Property attribute to calculate and access the gypsum/sulfate index (see Radwin & Bowen, 2021) imagery of the LandsatCollection. 
        This property initiates the calculation of gypsum using a default threshold of -1 (or a previously set threshold of self.gypsum_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
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
        self: self is passed into argument
        threshold (int): specify threshold for gypsum function (values less than threshold are masked)
        ng_threshold (int): (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('SR_B6') and available_bands.contains('SR_B7'):
            pass
        else:
            raise ValueError("Insufficient Bands for gypsum calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_gypsum_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)
    
    @property
    def turbidity(self):
        """
        Property attribute to calculate and access the turbidity (NDTI) imagery of the LandsatCollection. 
        This property initiates the calculation of turbidity using a default threshold of -1 (or a previously set threshold of self.turbidity_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        if self._turbidity is None:
            self._turbidity = self.turbidity_collection(self.turbidity_threshold)
        return self._turbidity

    def turbidity_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the turbidity (NDTI) index and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called 
        by default when using the ndwi property attribute.

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for the turbidity function (values less than threshold are masked)
        ng_threshold (int): (optional) specify threshold for Landsat 8&9 turbidity function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('SR_B4') and available_bands.contains('SR_B3'):
            pass
        else:
            raise ValueError("Insufficient Bands for turbidity calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_ndti_fn(image, threshold=threshold, ng_threshold=ng_threshold))

        return LandsatCollection(collection=col)
    
    @property
    def chlorophyll(self):
        """
        Property attribute to calculate and access the chlorophyll (NDTI) imagery of the LandsatCollection. 
        This property initiates the calculation of chlorophyll using a default threshold of -1 (or a previously set threshold of self.chlorophyll_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection 
        """
        if self._chlorophyll is None:
            self._chlorophyll = self.chlorophyll_collection(self.chlorophyll_threshold)
        return self._chlorophyll

    def chlorophyll_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate the KIVU chlorophyll index and return collection as class object, allows specifying threshold(s) for masking.
        Thresholds can be specified for Landsat 5 vs 8&9 images, where the threshold argument applies to Landsat 5 
        and the ng_threshold argument applies to Landsat 8&9. This function can be called as a method but is called 
        by default when using the ndwi property attribute.

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for the turbidity function (values less than threshold are masked)
        ng_threshold (int): (optional) specify threshold for Landsat 8&9 turbidity function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('SR_B4') and available_bands.contains('SR_B3') and available_bands.contains('SR_B2'):
            pass
        else:
            raise ValueError("Insufficient Bands for chlorophyll calculation")
        col = self.collection.map(lambda image: LandsatCollection.landsat_kivu_chla_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)
    
    @property
    def masked_water_collection(self):
        """
        Property attribute to mask water and return collection as class object.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection 
        """
        if self._masked_water_collection is None:
            col = self.collection.map(LandsatCollection.MaskWaterLandsat)
            self._masked_water_collection = LandsatCollection(collection=col)
        return self._masked_water_collection
    
    def masked_water_collection_NDWI(self, threshold):
        """
        Function to mask water pixels based on NDWI and user set threshold.

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for NDWI function (values greater than threshold are masked)

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection 
        """
        col = self.collection.map(lambda image: LandsatCollection.MaskWaterLandsatByNDWI(image, threshold=threshold))
        return LandsatCollection(collection=col)
    
    @property
    def masked_to_water_collection(self):
        """
        Property attribute to mask image to water, removing land and cloud pixels, and return collection as class object.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection
        """
        if self._masked_to_water_collection is None:
            col = self.collection.map(LandsatCollection.MaskToWaterLandsat)
            self._masked_to_water_collection = LandsatCollection(collection=col)
        return self._masked_to_water_collection
    
    def masked_to_water_collection_NDWI(self, threshold):
        """
        Function to mask all but water pixels based on NDWI and user set threshold.

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for NDWI function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection 
        """
        col = self.collection.map(lambda image: LandsatCollection.MaskToWaterLandsatByNDWI(image, threshold=threshold))
        return LandsatCollection(collection=col)
    
    @property
    def masked_clouds_collection(self):
        """
        Property attribute to mask clouds and return collection as class object.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection 
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

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection object containing LST imagery (temperature in Celcius).
        """
        if self._LST is None:
            self._LST = self.surface_temperature_collection()
        return self._LST
    
    def surface_temperature_collection(self):
        """
        Function to calculate LST (Land Surface Temperature - in Celcius) and return collection as class object.

        Args:
        self: self is passed into argument

        Returns:
        image collection (LandsatCollection): A LandsatCollection image collection
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('ST_ATRAN') and available_bands.contains('ST_EMIS') and available_bands.contains('ST_DRAD') and available_bands.contains('ST_TRAD') and available_bands.contains('ST_URAD') :
            pass
        else:
            raise ValueError("Insufficient Bands for temperature calculation")
        col = self.collection.map(LandsatCollection.temperature_bands).map(LandsatCollection.landsat_LST).map(LandsatCollection.image_dater)
        return LandsatCollection(collection=col)
    
    def mask_to_polygon(self, polygon):
        """
        Function to mask LandsatCollection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args:
        self: self is passed into argument (image collection)
        polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
        image collection (LandsatCollection): masked LandsatCollection image collection
        
        """
        if self._geometry_masked_collection is None:
            # Convert the polygon to a mask
            mask = ee.Image.constant(1).clip(polygon)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(mask))
            
            # Update the internal collection state
            self._geometry_masked_collection = LandsatCollection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_collection
    
    def mask_out_polygon(self, polygon):
        """
        Function to mask LandsatCollection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args:
        self: self is passed into argument (image collection)
        polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
        image collection (LandsatCollection): masked LandsatCollection image collection
        
        """
        if self._geometry_masked_out_collection is None:
            # Convert the polygon to a mask
            full_mask = ee.Image.constant(1)

            # Use paint to set pixels inside polygon as 0
            area = full_mask.paint(polygon, 0)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(area))
            
            # Update the internal collection state
            self._geometry_masked_out_collection = LandsatCollection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_out_collection

    def mask_halite(self, threshold, ng_threshold=None):
        """
        Function to mask halite and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images 
        where the threshold argument applies to Landsat 5 and the ng_threshold argument applies to Landsat 8&9.

        Args:
        self: self is passed into argument
        threshold (int): specify threshold for gypsum function (values less than threshold are masked)
        ng_threshold (int): (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked).

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection
        """
        col = self.collection.map(lambda image: LandsatCollection.halite_mask(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatCollection(collection=col)
    
    def mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None):
        """
        Function to mask halite and gypsum and return collection as class object. 
        Can specify separate thresholds for Landsat 5 vs 8&9 images where the threshold argument applies to Landsat 5
        and the ng_threshold argument applies to Landsat 8&9.

        Args:
        self: self is passed into argument
        halite_threshold (int): specify threshold for halite function (values less than threshold are masked)
        halite_ng_threshold (int): (optional) specify threshold for Landsat 8&9 halite function (values less than threshold are masked)
        gypsum_threshold (int): specify threshold for gypsum function (values less than threshold are masked)
        gypsum_ng_threshold (int): (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection 
        """
        col = self.collection.map(lambda image: LandsatCollection.gypsum_and_halite_mask(image, halite_threshold=halite_threshold, gypsum_threshold=gypsum_threshold, halite_ng_threshold=halite_ng_threshold, gypsum_ng_threshold=gypsum_ng_threshold))
        return LandsatCollection(collection=col)

    def image_grab(self, img_selector):
        """
        Function to select ("grab") an image by index from the collection. Easy way to get latest image or browse imagery one-by-one.

        Args:
        self: self is passed into argument
        img_selector: index of image in the collection for which user seeks to select/"grab".
        
        Returns:
        image (ee.Image): ee.Image of selected image
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
        self: self is passed into argument
        img_col: ee.ImageCollection with same dates as another LandsatCollection image collection object.
        img_selector: index of image in list of dates for which user seeks to "select".
        
        Returns:
        image (ee.Image): ee.Image of selected image
        """
        # Convert the collection to a list
        image_list = img_col.toList(img_col.size())

        # Get the image at the specified index
        image = ee.Image(image_list.get(img_selector))

        return image
    
    def image_pick(self, img_date):
        """
        Function to select ("grab") image of a specific date in format of 'YYYY-MM-DD' - will not work correctly if collection is composed of multiple images of the same date.

        Args:
        self: self is passed into argument
        img_date: date (str) of image to select in format of 'YYYY-MM-DD'

        Returns:
        image (ee.Image): ee.Image of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq('Date_Filter', img_date))
        return new_col.first()

    def CollectionStitch(self, img_col2):
        """
        Function to mosaic two LandsatCollection objects which share image dates. 
        Mosaics are only formed for dates where both image collections have images. 
        Image properties are copied from the primary collection. Server-side friendly.

        Args:
        self: self is passed into argument, which is a LandsatCollection image collection
        img_col2: secondary LandsatCollection image collection to be mosaiced with the primary image collection

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection
        """
        dates_list = ee.List(self._dates_list).cat(ee.List(img_col2.dates_list)).distinct()
        filtered_dates1 = self._dates_list
        filtered_dates2 = img_col2._dates_list

        filtered_col2 = img_col2.collection.filter(ee.Filter.inList('Date_Filter', filtered_dates1))
        filtered_col1 = self.collection.filter(ee.Filter.inList('Date_Filter', filtered_col2.aggregate_array('Date_Filter')))

        # Create a function that will be mapped over filtered_col1
        def mosaic_images(img):
            # Get the date of the image
            date = img.get('Date_Filter')
            
            # Get the corresponding image from filtered_col2
            img2 = filtered_col2.filter(ee.Filter.equals('Date_Filter', date)).first()

            # Create a mosaic of the two images
            mosaic = ee.ImageCollection.fromImages([img, img2]).mosaic()

            # Copy properties from the first image and set the 'Date_Filter' property
            mosaic = mosaic.copyProperties(img).set('Date_Filter', date).set('system:time_start', img.get('system:time_start'))

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
        Server-side friendly. NOTE: if images are removed from the collection from cloud filtering, you may have mosaics composed of only one image.

        Args:
        self: self is passed into argument, which is a LandsatCollection image collection

        Returns:
        image collection (LandsatCollection): LandsatCollection image collection with mosaiced imagery and mean CLOUD_COVER as a property
        """
        if self._MosaicByDate is None:
            input_collection = self.collection
            # Function to mosaic images of the same date and accumulate them
            def mosaic_and_accumulate(date, list_accumulator):
                # date = ee.Date(date)
                list_accumulator = ee.List(list_accumulator)
                date_filter = ee.Filter.eq('Date_Filter', date)
                date_collection = input_collection.filter(date_filter)
                # Convert the collection to a list
                image_list = date_collection.toList(date_collection.size())

                # Get the image at the specified index
                first_image = ee.Image(image_list.get(0))
                # Create mosaic
                mosaic = date_collection.mosaic().set('Date_Filter', date)

                # Calculate cumulative cloud and no data percentages
                cloud_percentage = date_collection.aggregate_mean('CLOUD_COVER')

                props_of_interest = ['SPACECRAFT_ID', 'SENSOR_ID', 'PROCESSING_LEVEL', 'ACQUISITION_DATE', 'system:time_start']

                # mosaic = mosaic.copyProperties(self.image_grab(0), props_of_interest).set({
                #     'CLOUD_COVER': cloud_percentage
                # })
                mosaic = mosaic.copyProperties(first_image, props_of_interest).set({
                    'CLOUD_COVER': cloud_percentage
                })

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array('Date_Filter').distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = LandsatCollection(collection=new_col)
            self._MosaicByDate = col

        # Convert the list of mosaics to an ImageCollection
        return self._MosaicByDate
    

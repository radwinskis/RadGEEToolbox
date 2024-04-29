import ee
import pandas as pd
import numpy as np
class Sentinel2Collection:
    """
    Class object representing a collection of ESA Sentinel-2 MSIsurface reflectance satellite images at 10 m/px resolution

    This class provides methods to filter, process, and analyze Sentinel-2 satellite imagery for a given period and region

    Arguments:
        start_date (str): Start date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        
        end_date (str): End date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        
        tile (str or list): MGRS tile(s) of Sentinel image (required unless boundary, relative_orbit_number, or collection is provided) | user is allowed to provide multiple tiles as list (note tile specifications will override boundary or orbits) - see https://hls.gsfc.nasa.gov/products-description/tiling-system/
        
        cloud_percentage_threshold (int): Integer percentage threshold where only imagery with cloud % less than threshold will be provided (defaults to 100)

        nodata_threshold (int): Integer percentage threshold where only imagery with nodata pixels encompassing a % less than the threshold will be provided (defaults to 100)

        boundary (ee.Geometry): Boundary for filtering images to images that intersect with the boundary shape (optional) - can be used in conjunction with relative_orbit_number
        
        relative_orbit_number (int or list): Relative orbit number(s) to filter collection (optional) - can be used in conjunction with boundary | provide multiple values as list
        
        collection (ee.ImageCollection): Optional argument to convert an ee.ImageCollection object to a Sentinel2Collection object - will override other arguments!

    Attributes:
        collection (returns: ee.ImageCollection): Returns an ee.ImageCollection object from any Sentinel2Collection image collection object
        
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

        _MosaicByDate: Cache storage for MosaicByDate property attribute

    Property Attributes:
        dates_list (returns: Server-Side List): Unreadable Earth Engine list of image dates (server-side)
        
        dates (returns: Client-Side List): Readable pythonic list of image dates (client-side)
        
        masked_clouds_collection (returns: Sentinel2Collection image collection): Returns collection with clouds masked (transparent) for each image

        masked_to_water_collection (returns: Sentinel2Collection image collection): Returns collection masked to just water pixels

        masked_water_collection (returns: Sentinel2Collection image collection): Returns collection with water pixels masked
        
        max (returns: ee.Image): Returns a temporally reduced max image (calculates max at each pixel)
        
        median (returns: ee.Image): Returns a temporally reduced median image (calculates median at each pixel)
        
        mean (returns: ee.Image): Returns a temporally reduced mean image (calculates mean at each pixel)
        
        min (returns: ee.Image): Returns a temporally reduced min image (calculates min at each pixel)
        
        ndwi (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband NDWI (water) rasters
        
        ndvi (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband NDVI (vegetation) rasters

        halite (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband halite index rasters

        gypsum (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband gypsum index rasters

        turbidity (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband NDTI (turbidity) rasters

        chlorophyll (returns: ee.ImageCollection): Returns Sentinel2Collection image collection of singleband 2BDA (relative chlorophyll-a) rasters

        MosaicByDate (returns: Sentinel2Collection image collection): Mosaics image collection where images with the same date are mosaiced into the same image. Calculates total cloud percentage for subsequent filtering of cloudy mosaics.

    Methods:        
        ndwi_collection(self, threshold)
        
        ndvi_collection(self, threshold)
        
        halite_collection(self, threshold)
        
        gypsum_collection(self, threshold)

        turbidity_collection(self, threshold)

        chlorophyll_collection(self, threshold)

        get_filtered_collection(self)

        get_boundary_filtered_collection(self)

        get_orbit_filtered_collection(self)

        get_orbit_and_boundary_filtered_collection(self)
        
        mask_to_polygon(self, polygon)

        mask_out_polygon(self, polygon)

        masked_water_collection_NDWI(self, threshold)

        masked_to_water_collection_NDWI(self, threshold)
        
        mask_halite(self, threshold)
        
        mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold)

        PixelAreaSumCollection(self, band_name, geometry, threshold, scale, maxPixels)
        
        image_grab(self, img_selector)
        
        custom_image_grab(self, img_col, img_selector)
        
        image_pick(self, img_date)
        
        CollectionStitch(self, img_col2)
        

    Static Methods:
        image_dater(image)
        
        sentinel_ndwi_fn(image, threshold)
        
        sentinel_ndvi_fn(image, threshold)
        
        sentinel_halite_fn(image, threshold)
        
        sentinel_gypsum_fn(image, threshold)

        sentinel_turbidity_fn(image, threshold)
        
        sentinel_chlorophyll_fn(image, threshold)
        
        MaskWaterS2(image)

        MaskToWaterS2(image)
        
        MaskWaterS2ByNDWI(image, threshold)

        MaskToWaterS2ByNDWI(image, threshold)

        halite_mask(image, threshold)
        
        gypsum_and_halite_mask(image, halite_threshold, gypsum_threshold)
        
        MaskCloudsS2(image)
        
        PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12)
        

    Usage:
        The Sentinel2Collection object alone acts as a base object for which to further filter or process to indices or spatial reductions
        
        To use the Sentinel2Collection functionality, use any of the built in class attributes or method functions. For example, using class attributes:
        
        image_collection = Sentinel2Collection(start_date, end_date, tile, cloud_percentage_threshold)

        ee_image_collection = image_collection.collection #returns ee.ImageCollection from provided argument filters

        latest_image = image_collection.image_grab(-1) #returns latest image in collection as ee.Image

        cloud_masked_collection = image_collection.masked_clouds_collection #returns cloud-masked Sentinel2Collection image collection

        NDWI_collection = image_collection.ndwi #returns NDWI Sentinel2Collection image collection

        latest_NDWI_image = NDWI_collection.image_grab(-1) #Example showing how class functions work with any Sentinel2Collection image collection object, returning latest ndwi image
    """

    def __init__(self, start_date=None, end_date=None, tile=None, cloud_percentage_threshold=None, nodata_threshold=None, boundary=None, relative_orbit_number=None, collection=None):
        if collection is None and (start_date is None or end_date is None):
            raise ValueError("Either provide all required fields (start_date, end_date, tile, cloud_percentage_threshold, nodata_threshold) or provide a collection.")
        if tile is None and boundary is None and relative_orbit_number is None and collection is None:
            raise ValueError("Provide either tile, boundary/geometry, or relative orbit number specifications to filter the image collection")
        if collection is None:
            self.start_date = start_date
            self.end_date = end_date
            self.tile = tile
            self.boundary = boundary
            self.relative_orbit_number = relative_orbit_number

            if cloud_percentage_threshold is None:
                cloud_percentage_threshold = 100
                self.cloud_percentage_threshold = cloud_percentage_threshold
            else:
                self.cloud_percentage_threshold = cloud_percentage_threshold

            if nodata_threshold is None:
                nodata_threshold = 100
                self.nodata_threshold = nodata_threshold
            else:
                self.nodata_threshold = nodata_threshold

            if isinstance(tile, list):
                pass
            else:
                self.tile = [tile]

            if isinstance(relative_orbit_number, list):
                pass
            else:
                self.relative_orbit_number = [relative_orbit_number]

            # Filter the collection
            if tile is not None:
                self.collection = self.get_filtered_collection()
            elif boundary is not None and relative_orbit_number is None:
                self.collection = self.get_boundary_filtered_collection()
            elif relative_orbit_number is not None and boundary is None:
                self.collection = self.get_orbit_filtered_collection()
            elif relative_orbit_number is not None and boundary is not None:
                self.collection = self.get_orbit_and_boundary_filtered_collection()
        else:
            self.collection = collection

        self._dates_list = None
        self._dates = None
        self.ndwi_threshold = -1
        self.ndvi_threshold = -1
        self.halite_threshold = -1
        self.gypsum_threshold = -1
        self.turbidity_threshold = -1
        self.chlorophyll_threshold = 0.5
        
        
        self._geometry_masked_collection = None
        self._geometry_masked_out_collection = None
        self._masked_clouds_collection = None
        self._masked_to_water_collection = None
        self._masked_water_collection = None
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
        self._MosaicByDate = None
        self._PixelAreaSumCollection = None

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
    def sentinel_ndwi_fn(image, threshold):
        """
        Function to calculate ndwi for Sentinel2 imagery and mask image based on threshold.

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): ndwi image
        """
        ndwi_calc = image.normalizedDifference(['B3', 'B8']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        return water

    @staticmethod
    def sentinel_ndvi_fn(image, threshold):
        """
        Function to calculate ndvi for for Sentinel2 imagery and mask image based on threshold.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): ndvi image
        """
        ndvi_calc = image.normalizedDifference(['B8', 'B4']) #NIR-RED/NIR+RED -- full NDVI image
        vegetation = ndvi_calc.updateMask(ndvi_calc.gte(threshold)).rename('ndvi').copyProperties(image) # subsets the image to just water pixels, 0.2 threshold for datasets
        return vegetation

    @staticmethod
    def sentinel_halite_fn(image, threshold):
        """
        Function to calculate multispectral halite index for Sentinel2 imagery and mask image based on threshold.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): halite ee.Image
        """
        halite_index = image.normalizedDifference(['B4', 'B11'])
        halite = halite_index.updateMask(halite_index.gte(threshold)).rename('halite').copyProperties(image)
        return halite

    @staticmethod
    def sentinel_gypsum_fn(image, threshold):
        """
        Function to calculate multispectral gypsum index for Sentinel2 imagery and mask image based on threshold.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): gypsum ee.Image
        """
        gypsum_index = image.normalizedDifference(['B11', 'B12'])
        gypsum = gypsum_index.updateMask(gypsum_index.gte(threshold)).rename('gypsum').copyProperties(image)
        return gypsum
    
    @staticmethod
    def sentinel_turbidity_fn(image, threshold):
        """
        Function to calculate Normalized Difference Turbidity Index (NDTI; Lacaux et al., 2007) for for Sentinel2 imagery and mask image based on threshold.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): turbidity ee.Image
        """
        NDTI = image.normalizedDifference(['B3', 'B2'])
        turbidity = NDTI.updateMask(NDTI.gte(threshold)).rename('ndti').copyProperties(image)
        return turbidity
    
    @staticmethod
    def sentinel_chlorophyll_fn(image, threshold):
        """
        Function to calculate relative chlorophyll-a concentrations of water pixels using 2BDA index (see Buma and Lee, 2020 for review) and mask image based on threshold. NOTE: the image is downsampled to 20 meters as the red edge 1 band is utilized.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): chlorophyll-a ee.Image
        """
        chl_index = image.normalizedDifference(['B5', 'B4'])
        chlorophyll = chl_index.updateMask(chl_index.gte(threshold)).rename('2BDA').copyProperties(image)
        return chlorophyll
    
    @staticmethod
    def MaskCloudsS2(image):
        """
        Function to map clouds using SCL band data.

        Args:
        image (ee.Image): input image

        Returns:
        image (ee.Image): output ee.Image with clouds masked
        """
        SCL = image.select('SCL')
        CloudMask = SCL.neq(9)
        return image.updateMask(CloudMask).copyProperties(image)
    
    @staticmethod
    def MaskWaterS2(image):
        """
        Function to mask water pixels using SCL band data.

        Args:
        image (ee.Image): input image

        Returns:
        image (ee.Image): output ee.Image with water pixels masked
        """
        SCL = image.select('SCL')
        WaterMask = SCL.neq(6)
        return image.updateMask(WaterMask).copyProperties(image)
    
    @staticmethod
    def MaskWaterS2ByNDWI(image, threshold):
        """
        Function to mask water pixels (mask land and cloud pixels) for all bands based on NDWI and a set threshold where
        all pixels less than NDWI threshold are masked out.

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): ee.Image
        """
        ndwi_calc = image.normalizedDifference(['B3', 'B8']) #green-NIR / green+NIR -- full NDWI image
        water = image.updateMask(ndwi_calc.lt(threshold)) 
        return water
    
    @staticmethod
    def MaskToWaterS2(image):
        """
        Function to mask to water pixels (mask land and cloud pixels) using SCL band data.

        Args:
        image (ee.Image): input image

        Returns:
        image (ee.Image): output ee.Image with all but water pixels masked
        """
        SCL = image.select('SCL')
        WaterMask = SCL.eq(6)
        return image.updateMask(WaterMask).copyProperties(image)
    
    @staticmethod
    def halite_mask(image, threshold):
        """
        Function to mask halite pixels after specifying index to isolate/mask-to halite pixels.

        Args:
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked..
        
        Returns:
        image (ee.Image): ee.Image where halite pixels are masked (image without halite pixels).
        """
        halite_index = image.normalizedDifference(['B4', 'B11'])
        mask = image.updateMask(halite_index.lt(threshold)).copyProperties(image)
        return mask 
    
    @staticmethod
    def gypsum_and_halite_mask(image, halite_threshold, gypsum_threshold):
        """
        Function to mask both gypsum and halite pixels. Must specify threshold for isolating halite and gypsum pixels. 

        Args:
        image (ee.Image): input image
        halite_threshold: integer threshold for halite where pixels less than threshold are masked.
        gypsum_threshold: integer threshold for gypsum where pixels less than threshold are masked.
       
        Returns:
        image (ee.Image): ee.Image where gypsum and halite pixels are masked (image without halite or gypsum pixels).
        """
        halite_index = image.normalizedDifference(['B4', 'B11'])
        gypsum_index = image.normalizedDifference(['B11', 'B12'])
        
        mask = gypsum_index.updateMask(halite_index.lt(halite_threshold)).updateMask(gypsum_index.lt(gypsum_threshold)).rename('carbonate_muds').copyProperties(image)
        return mask
    
    @staticmethod
    def MaskToWaterS2ByNDWI(image, threshold):
        """
        Function to mask all bands to water pixels (mask land and cloud pixels) based on NDWI.

        Args: 
        image (ee.Image): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked.

        Returns:
        image (ee.Image): ee.Image image
        """
        ndwi_calc = image.normalizedDifference(['B3', 'B8']) #green-NIR / green+NIR -- full NDWI image
        water = image.updateMask(ndwi_calc.gte(threshold)) 
        return water
    
    @staticmethod
    def PixelAreaSum(image, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) within a geometry and store the value as image property (matching name of chosen band).
        The resulting value has units of square meters. 

        Args:
        image (ee.Image): input ee.Image
        band_name: name of band (string) for calculating area.
        geometry: ee.Geometry object denoting area to clip to for area calculation.
        threshold: integer threshold to specify masking of pixels below threshold (defaults to -1).
        scale: integer scale of image resolution (meters) (defaults to 10).
        maxPixels: integer denoting maximum number of pixels for calculations.
        
        Returns:
        image (ee.Image): Image with area calculation stored as property matching name of band.
        """
        area_image = ee.Image.pixelArea()
        mask = image.select(band_name).gte(threshold)
        final = image.addBands(area_image)
        stats = final.select('area').updateMask(mask).rename(band_name).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry= geometry,
            scale=scale,
            maxPixels = maxPixels)
        return image.set(band_name, stats.get(band_name)) #calculates and returns summed pixel area as image property titled the same as the band name of the band used for calculation
    
    def PixelAreaSumCollection(self, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) 
        within a geometry and store the value as image property (matching name of chosen band) for an entire
        image collection.
        The resulting value has units of square meters. 

        Args:
        self: self is the input image collection
        band_name: name of band (string) for calculating area.
        geometry: ee.Geometry object denoting area to clip to for area calculation.
        threshold: integer threshold to specify masking of pixels below threshold (defaults to -1).
        scale: integer scale of image resolution (meters) (defaults to 10).
        maxPixels: integer denoting maximum number of pixels for calculations.
        
        Returns:
        image (ee.Image): Image with area calculation stored as property matching name of band.
        """
        if self._PixelAreaSumCollection is None:
            collection = self.collection
            AreaCollection = collection.map(lambda image: Sentinel2Collection.PixelAreaSum(image, band_name=band_name, geometry=geometry, threshold=threshold, scale=scale, maxPixels=maxPixels))
            self._PixelAreaSumCollection = AreaCollection
        return self._PixelAreaSumCollection

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
        Function to filter image collection using a list of MGRS tiles (based on Sentinel2Collection class arguments).

        Args:
        self: self is passed into argument.

        Returns:
        image collection (ee.ImageCollection): Image collection objects
        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('MGRS_TILE', self.tile)).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_boundary_filtered_collection(self):
        """
        Function to filter image collection using a geometry/boundary rather than list of tiles (based on Sentinel2Collection class arguments).

        Args:
        self: self is passed into argument.

        Returns:
        image collection (ee.ImageCollection): Image collection objects

        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filterBounds(self.boundary).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_orbit_filtered_collection(self):
        """
        Function to filter image collection a list of relative orbit numbers rather than list of tiles (based on Sentinel2Collection class arguments).

        Args:
        self: self is passed into argument.

        Returns:
        image collection (ee.ImageCollection): Image collection objects
        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('SENSING_ORBIT_NUMBER', self.relative_orbit_number)).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_orbit_and_boundary_filtered_collection(self):
        """
        Function to filter image collection a list of relative orbit numbers and geometry/boundary rather than list of tiles (based on Sentinel2Collection class arguments).

        Args:
        self: self is passed into argument.

        Returns:
        image collection (ee.ImageCollection): Image collection objects
        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('SENSING_ORBIT_NUMBER', self.relative_orbit_number)).filterBounds(self.boundary).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    @property
    def median(self):
        """
        Property attribute function to calculate median image from image collection. Results are calculated once per class object then cached for future use.

        Args:
        self: self is passed into argument.

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
        self: self is passed into argument.

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
        self: self is passed into argument.

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
        self: self is passed into argument.

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
        Property attribute to calculate and access the NDWI (Normalized Difference Water Index) imagery of the Sentinel2Collection. 
        This property initiates the calculation of NDWI using a default threshold of -1 (or a previously set threshold of self.ndwi_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._ndwi is None:
            self._ndwi = self.ndwi_collection(self.ndwi_threshold)
        return self._ndwi

    def ndwi_collection(self, threshold):
        """
        Function to calculate ndwi and return collection as class object. Masks collection based on threshold which defaults to -1.

        Args:
        self: self is passed into argument
        threshold: specify threshold for NDWI function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()

        if available_bands.contains('B3') and available_bands.contains('B8'):
            pass
        else:
            raise ValueError("Insufficient Bands for ndwi calculation")
        col =  self.collection.map(lambda image: Sentinel2Collection.sentinel_ndwi_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    @property
    def ndvi(self):
        """
        Property attribute to calculate and access the NDVI (Normalized Difference Vegetation Index) imagery of the Sentinel2Collection. 
        This property initiates the calculation of NDVI using a default threshold of -1 (or a previously set threshold of self.ndvi_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._ndvi is None:
            self._ndvi = self.ndvi_collection(self.ndvi_threshold)
        return self._ndvi
    
    def ndvi_collection(self, threshold):
        """
        Function to calculate ndvi and return collection as class object. Masks collection based on threshold which defaults to -1.

        Args:
        self: self is passed into argument.
        threshold: specify threshold for NDVI function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('B4') and available_bands.contains('B8'):
            pass
        else:
            raise ValueError("Insufficient Bands for ndvi calculation")
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_ndvi_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)

    @property
    def halite(self):
        """
        Property attribute to calculate and access the halite index (see Radwin & Bowen, 2021) imagery of the Sentinel2Collection. 
        This property initiates the calculation of halite using a default threshold of -1 (or a previously set threshold of self.halite_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._halite is None:
            self._halite = self.halite_collection(self.halite_threshold)
        return self._halite

    def halite_collection(self, threshold):
        """
        Function to calculate multispectral halite index and return collection as class object. Masks collection based on threshold which defaults to -1.

        Args:
        self: self is passed into argument.
        threshold: specify threshold for halite function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('B4') and available_bands.contains('B11'):
            pass
        else:
            raise ValueError("Insufficient Bands for halite calculation")
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_halite_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)

    @property
    def gypsum(self):
        """
        Property attribute to calculate and access the gypsum/sulfate index (see Radwin & Bowen, 2021) imagery of the Sentinel2Collection. 
        This property initiates the calculation of gypsum using a default threshold of -1 (or a previously set threshold of self.gypsum_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._gypsum is None:
            self._gypsum = self.gypsum_collection(self.gypsum_threshold)
        return self._gypsum

    def gypsum_collection(self, threshold):
        """
        Function to calculate multispectral gypsum index and return collection as class object.  Masks collection based on threshold which defaults to -1.

        Args:
        self: self is passed into argument.
        threshold: specify threshold for gypsum function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('B11') and available_bands.contains('B12'):
            pass
        else:
            raise ValueError("Insufficient Bands for gypsum calculation")
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_gypsum_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    @property
    def turbidity(self):
        """
        Property attribute to calculate and access the turbidity (NDTI) imagery of the Sentinel2Collection. 
        This property initiates the calculation of turbidity using a default threshold of -1 (or a previously set threshold of self.turbidity_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._turbidity is None:
            self._turbidity = self.turbidity_collection(self.turbidity_threshold)
        return self._turbidity

    def turbidity_collection(self, threshold):
        """
        Function to calculate NDTI turbidity index and return collection as class object. Masks collection based on threshold which defaults to -1.

        Args:
        self: self is passed into argument
        threshold: specify threshold for NDTI function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('B3') and available_bands.contains('B2'):
            pass
        else:
            raise ValueError("Insufficient Bands for turbidity calculation")
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_turbidity_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    @property
    def chlorophyll(self):
        """
        Property attribute to calculate and access the chlorophyll (NDTI) imagery of the Sentinel2Collection. 
        This property initiates the calculation of chlorophyll using a default threshold of -1 (or a previously set threshold of self.chlorophyll_threshold) 
        and caches the result. The calculation is performed only once when the property is first accessed, and the cached result is returned 
        on subsequent accesses.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): A Sentinel2Collection image collection.
        """
        if self._chlorophyll is None:
            self._chlorophyll = self.chlorophyll_collection(self.chlorophyll_threshold)
        return self._chlorophyll

    def chlorophyll_collection(self, threshold):
        """
        Function to calculate 2BDA chlorophyll index and return collection as class object. Masks collection based on threshold which defaults to 0.5.

        Args:
        self: self is passed into argument.
        threshold: specify threshold for 2BDA function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        first_image = self.collection.first()
        available_bands = first_image.bandNames()
        if available_bands.contains('B5') and available_bands.contains('B4'):
            pass
        else:
            raise ValueError("Insufficient Bands for chlorophyll calculation")
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_chlorophyll_fn(image, threshold=threshold))
        return Sentinel2Collection(collection=col)

    @property
    def masked_water_collection(self):
        """
        Property attribute to mask water and return collection as class object.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        if self._masked_water_collection is None:
            col = self.collection.map(Sentinel2Collection.MaskWaterS2)
            self._masked_water_collection = Sentinel2Collection(collection=col)
        return self._masked_water_collection
    
    def masked_water_collection_NDWI(self, threshold):
        """
        Function to mask water by using NDWI and return collection as class object.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        col = self.collection.map(lambda image: Sentinel2Collection.MaskWaterS2ByNDWI(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    @property
    def masked_to_water_collection(self):
        """
        Property attribute to mask to water (mask land and cloud pixels) and return collection as class object.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        if self._masked_to_water_collection is None:
            col = self.collection.map(Sentinel2Collection.MaskToWaterS2)
            self._masked_water_collection = Sentinel2Collection(collection=col)
        return self._masked_water_collection
    
    def masked_to_water_collection_NDWI(self, threshold):
        """
        Function to mask to water pixels by using NDWI and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection.
        """
        col = self.collection.map(lambda image: Sentinel2Collection.MaskToWaterS2ByNDWI(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    @property
    def masked_clouds_collection(self):
        """
        Property attribute to mask clouds and return collection as class object.

        Args:
        self: self is passed into argument.

        Returns:
        image collection (Sentinel2Collection): masked Sentinel2Collection image collection.
        """
        if self._masked_clouds_collection is None:
            col = self.collection.map(Sentinel2Collection.MaskCloudsS2)
            self._masked_clouds_collection = Sentinel2Collection(collection=col)
        return self._masked_clouds_collection
    
    def mask_to_polygon(self, polygon):
        """
        Function to mask Sentinel2Collection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args:
        self: self is passed into argument.
        polygon: ee.Geometry polygon or shape used to mask image collection.

        Returns:
        image collection (Sentinel2Collection): masked Sentinel2Collection image collection.
        
        """
        if self._geometry_masked_collection is None:
            # Convert the polygon to a mask
            mask = ee.Image.constant(1).clip(polygon)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(mask))
            
            # Update the internal collection state
            self._geometry_masked_collection = Sentinel2Collection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_collection
    
    def mask_out_polygon(self, polygon):
        """
         Function to mask Sentinel2Collection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args:
        self: self is passed into argument.
        polygon: ee.Geometry polygon or shape used to mask image collection.

        Returns:
        image collection (Sentinel2Collection): masked Sentinel2Collection image collection.
        
        """
        if self._geometry_masked_out_collection is None:
            # Convert the polygon to a mask
            full_mask = ee.Image.constant(1)

            # Use paint to set pixels inside polygon as 0
            area = full_mask.paint(polygon, 0)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(area))
            
            # Update the internal collection state
            self._geometry_masked_out_collection = Sentinel2Collection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_out_collection
    
    def mask_halite(self, threshold):
        """
        Function to mask halite and return collection as class object. 

        Args:
        self: self is passed into argument
        threshold: specify threshold for gypsum function (values less than threshold are masked).
  
        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection
        """
        col = self.collection.map(lambda image: Sentinel2Collection.halite_mask(image, threshold=threshold))
        return Sentinel2Collection(collection=col)
    
    def mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold):
        """
        Function to mask halite and gypsum and return collection as class object. 

        Args:
        self: self is passed into argument
        halite_threshold: specify threshold for halite function (values less than threshold are masked).
        gypsum_threshold: specify threshold for gypsum function (values less than threshold are masked).

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection
        """
        col = self.collection.map(lambda image: Sentinel2Collection.gypsum_and_halite_mask(image, halite_threshold=halite_threshold, gypsum_threshold=gypsum_threshold))
        return Sentinel2Collection(collection=col)

    def image_grab(self, img_selector):
        """
        Function to select ("grab") an image by index from the collection. Easy way to get latest image or browse imagery one-by-one.

        Args:
        self: self is passed into argument.
        img_selector: index of image in the collection for which user seeks to select/"grab".
        
        Returns:
        image (ee.Image): ee.Image of selected image.
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
        self: self is passed into argument.
        img_col: ee.ImageCollection with same dates as another Sentinel2Collection image collection object.
        img_selector: index of image in list of dates for which user seeks to "select".
        
        Returns:
        image (ee.Image): ee.Image of selected image.
        """
        # Convert the collection to a list
        image_list = img_col.toList(img_col.size())

        # Get the image at the specified index
        image = ee.Image(image_list.get(img_selector))

        return image
    
    def image_pick(self, img_date):
        """
        Function to select ("grab") image of a specific date in format of 'YYYY-MM-DD'.
        Will not work correctly if collection is composed of multiple images of the same date.

        Args:
        self: self is passed into argument.
        img_date: date (str) of image to select in format of 'YYYY-MM-DD'.

        Returns:
        image (ee.Image): ee.Image of selected image.
        """
        new_col = self.collection.filter(ee.Filter.eq('Date_Filter', img_date))
        return new_col.first()
    
    def CollectionStitch(self, img_col2):
        """
        Function to mosaic two Sentinel2Collection objects which share image dates. 
        Mosaics are only formed for dates where both image collections have images. 
        Image properties are copied from the primary collection.
        Server-side friendly.

        Args:
        self: self is passed into argument, which is a Sentinel2Collection image collection.
        img_col2: secondary Sentinel2Collection image collection to be mosaiced with the primary image collection.

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection
        """
        dates_list = ee.List(self.dates_list).cat(ee.List(img_col2.dates_list)).distinct()
        filtered_dates1 = self.dates_list
        filtered_dates2 = img_col2.dates_list

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

            # Copy properties from the first image and set the time properties
            mosaic = mosaic.copyProperties(img).set('Date_Filter', date).set('system:time_start', img.get('system:time_start'))

            return mosaic

        # Map the function over filtered_col1
        new_col = filtered_col1.map(mosaic_images)

        # Return a Sentinel2Collection instance
        return Sentinel2Collection(collection=new_col)
    
    @property
    def MosaicByDate(self):
        """
        Property attribute function to mosaic collection images that share the same date. The properties CLOUD_PIXEL_PERCENTAGE and NODATA_PIXEL_PERCENTAGE 
        for each image are used to calculate an overall mean, which replaces the CLOUD_PIXEL_PERCENTAGE and NODATA_PIXEL_PERCENTAGE for each mosaiced image. 
        Server-side friendly. NOTE: if images are removed from the collection from cloud filtering, you may have mosaics composed of only one image.

        Args:
        self: self is passed into argument, which is a Sentinel2Collection image collection.

        Returns:
        image collection (Sentinel2Collection): Sentinel2Collection image collection 
        """
        if self._MosaicByDate is None:
            input_collection = self.collection
            # Function to mosaic images of the same date and accumulate them
            def mosaic_and_accumulate(date, list_accumulator):
                # date = ee.Date(date)
                list_accumulator = ee.List(list_accumulator)
                date_filter = ee.Filter.eq('Date_Filter', date)
                date_collection = input_collection.filter(date_filter)
                image_list = date_collection.toList(date_collection.size())
                first_image = ee.Image(image_list.get(0))
                
                # Create mosaic
                mosaic = date_collection.mosaic().set('Date_Filter', date)

                # Calculate cumulative cloud and no data percentages
                cloud_percentage = date_collection.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE')
                no_data_percentage = date_collection.aggregate_mean('NODATA_PIXEL_PERCENTAGE')

                props_of_interest = ['SPACECRAFT_NAME', 'SENSING_ORBIT_NUMBER', 'SENSING_ORBIT_DIRECTION', 'MISSION_ID', 'PLATFORM_IDENTIFIER', 'system:time_start']

                mosaic = mosaic.copyProperties(first_image, props_of_interest).set({
                    'CLOUDY_PIXEL_PERCENTAGE': cloud_percentage,
                    'NODATA_PIXEL_PERCENTAGE': no_data_percentage
                })

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array('Date_Filter').distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = Sentinel2Collection(collection=new_col)
            self._MosaicByDate = col

        return self._MosaicByDate

    @staticmethod
    def ee_to_df(ee_object, columns=None, remove_geom=True, sort_columns=False, **kwargs):
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
    def extract_transect(image, line, reducer="mean", n_segments=100, dist_interval=None, scale=None, crs=None, crsTransform=None, tileScale=1.0, to_pandas=False, **kwargs):

        """Extracts transect from an image. Adapted from the geemap package (https://geemap.org/common/#geemap.common.extract_transect)

        Args:
            image (ee.Image): The image to extract transect from.
            line (ee.Geometry.LineString): The LineString used to extract transect from an image.
            reducer (str, optional): The ee.Reducer to use, e.g., 'mean', 'median', 'min', 'max', 'stdDev'. Defaults to "mean".
            n_segments (int, optional): The number of segments that the LineString will be split into. Defaults to 100.
            dist_interval (float, optional): The distance interval in meters used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to None.
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
                return Sentinel2Collection.ee_to_df(transect)
            return transect

        except Exception as e:
            raise Exception(e)
    
    @staticmethod
    def transect(image, lines, line_names, reducer='mean', n_segments=None, dist_interval=10, to_pandas=True):
        """Computes and stores the values along a transect for each line in a list of lines. Builds off of the extract_transect function from the geemap package
            where checks are ran to ensure that the reducer column is present in the transect data. If the reducer column is not present, a column of NaNs is created.
            An ee reducer is used to aggregate the values along the transect, depending on the number of segments or distance interval specified. Defaults to 'mean' reducer.

        Args:
            image (ee.Image): ee.Image object to use for calculating transect values.
            lines (list): List of ee.Geometry.LineString objects.
            line_names (list of strings): List of line string names.
            reducer (str): The ee reducer to use. Defaults to 'mean'.
            n_segments (int): The number of segments that the LineString will be split into. Defaults to None.
            dist_interval (float): The distance interval in meters used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 10.
            to_pandas (bool): Whether to convert the result to a pandas dataframe. Defaults to True.

        Returns:
            pd.DataFrame or ee.FeatureCollection: organized list of values along the transect(s)
        """
        #Create empty dataframe
        transects_df = pd.DataFrame()

        #Check if line is a list of lines or a single line - if single line, convert to list
        if isinstance(lines, list):
            pass
        else:
            lines = [lines]
        
        for i, line in enumerate(lines):
            if n_segments is None:
                transect_data = Sentinel2Collection.extract_transect(image=image, line=line, reducer=reducer, dist_interval=dist_interval, to_pandas=to_pandas)
                if reducer in transect_data.columns:
                    # Extract the 'mean' column and rename it
                    mean_column = transect_data[['mean']]
                else:
                    # Handle the case where 'mean' column is not present
                    print(f"{reducer} column not found in transect data for line {line_names[i]}")
                    # Create a column of NaNs with the same length as the longest column in transects_df
                    max_length = max(transects_df.shape[0], transect_data.shape[0])
                    mean_column = pd.Series([np.nan] * max_length)
            else:
                transect_data = Sentinel2Collection.extract_transect(image=image, line=line, reducer=reducer, n_segments=n_segments, to_pandas=to_pandas)
                if reducer in transect_data.columns:
                    # Extract the 'mean' column and rename it
                    mean_column = transect_data[['mean']]
                else:
                    # Handle the case where 'mean' column is not present
                    print(f"{reducer} column not found in transect data for line {line_names[i]}")
                    # Create a column of NaNs with the same length as the longest column in transects_df
                    max_length = max(transects_df.shape[0], transect_data.shape[0])
                    mean_column = pd.Series([np.nan] * max_length)
            
            transects_df = pd.concat([transects_df, mean_column], axis=1)

        transects_df.columns = line_names
                
        return transects_df
    
    def transect_iterator(self, lines, line_names, save_folder_path, reducer='mean', n_segments=None, dist_interval=10, to_pandas=True):
        """Computes and stores the values along a transect for each line in a list of lines for each image in a Sentinel2Collection image collection, then saves the data for each image to a csv file. Builds off of the extract_transect function from the geemap package
            where checks are ran to ensure that the reducer column is present in the transect data. If the reducer column is not present, a column of NaNs is created.
            An ee reducer is used to aggregate the values along the transect, depending on the number of segments or distance interval specified. Defaults to 'mean' reducer.
            Naming conventions for the csv files follows as: "image-date_transects.csv"

        Args:
            self (Sentinel2Collection image collection): Image collection object to iterate for calculating transect values for each image.
            lines (list): List of ee.Geometry.LineString objects.
            line_names (list of strings): List of line string names.
            save_folder_path (str): The path to the folder where the csv files will be saved.
            reducer (str): The ee reducer to use. Defaults to 'mean'.
            n_segments (int): The number of segments that the LineString will be split into. Defaults to None.
            dist_interval (float): The distance interval in meters used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 10.
            to_pandas (bool): Whether to convert the result to a pandas dataframe. Defaults to True.

        Raises:
            Exception: If the program fails to compute.

        Returns:
            csv file: file for each image with an organized list of values along the transect(s)
        """
        image_collection = self #.collection
        image_collection_dates = self.dates
        for i, date in enumerate(image_collection_dates):
            try:
                print(f"Processing image {i+1}/{len(image_collection_dates)}: {date}")
                image = image_collection.image_grab(i)
                transects_df = Sentinel2Collection.transect(image, lines, line_names, reducer=reducer, n_segments=n_segments, dist_interval=dist_interval, to_pandas=to_pandas)
                image_id = date
                transects_df.to_csv(f'{save_folder_path}{image_id}_transects.csv')
                print(f'{image_id}_transects saved to csv')
            except Exception as e:
                print(f"An error occurred while processing image {i+1}: {e}")

import geemap
import ee
class Sentinel2Collection:
    """
    Class object representing a collection of Sentinel satellite images

    This class provides methods to filter, process, and analyze Sentinel satellite imagery for a given period and region

    Arguments:
        start_date (str): Start date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        end_date (str): End date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        tile (str): tile of Sentinel image (required unless collection is provided)
        boundary (ee geometry): Boundary for filtering images to images that intersect with the boundary shape (optional)
        cloud_percentage_threshold (int): Integer percentage threshold where only imagery with cloud % less than threshold will be provided (optional; defaults to 15)
        collection (ee ImageCollection): Optional argument to convert an eeImageCollection object to a SentinelCollection object - will override other arguments!

    Attributes:
        collection (returns: eeImageCollection): Returns an eeImageCollection object from any SentinelCollection image collection object
        dates_list (returns: Server-Side List): Unreadable Earth Engine list of image dates (server-side)
        dates (returns: Client-Side List): Readable pythonic list of image dates (client-side)
        masked_clouds_collection (returns: SentinelCollection image collection): Returns collection with clouds masked (transparent) for each image
        max (returns: eeImage): Returns a temporally reduced max image (calculates max at each pixel)
        median (returns: eeImage): Returns a temporally reduced median image (calculates median at each pixel)
        mean (returns: eeImage): Returns a temporally reduced mean image (calculates mean at each pixel)
        min (returns: eeImage): Returns a temporally reduced min image (calculates min at each pixel)
        gypsum (returns: eeImageCollection): Returns SentinelCollection image collection of singleband gypsum index rasters
        halite (returns: eeImageCollection): Returns SentinelCollection image collection of singleband halite index rasters
        ndwi (returns: eeImageCollection): Returns SentinelCollection image collection of singleband NDWI (water) rasters
        ndvi (returns: eeImageCollection): Returns SentinelCollection image collection of singleband NDVI (vegetation) rasters

    Methods:
        median_collection(self)
        mean_collection(self)
        max_collection(self)
        min_collection(self)
        ndwi_collection(self, threshold)
        ndvi_collection(self, threshold)
        halite_collection(self, threshold)
        gypsum_collection(self, threshold)
        mask_with_polygon(self, polygon)
        masked_water_collection(self)
        masked_clouds_collection(self)
        surface_temperature_collection(self)
        mask_halite(self, threshold)
        mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold)
        list_of_dates(self)
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
        MaskWaterS2(image)
        halite_mask(image, threshold)
        gypsum_and_halite_mask(image, halite_threshold, gypsum_threshold)
        MaskCloudsS2(image)
        PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12)
        

    Usage:
        The SentinelCollection object alone acts as a base object for which to further filter or process to indices or spatial reductions
        To use the SentinelCollection functionality, use any of the built in class attributes or method functions. For example, using class attributes:

        image_collection = SentinelCollection('arguments set by user').collection

        ee_image_collection = image_collection.collection #returns eeImageCollection from provided argument filters

        latest_image = image_collection.image_grab(-1) #returns latest image in collection as eeImage

        cloud_masked_collection = image_collection.masked_clouds_collection #returns cloud-masked SentinelCollection image collection

        NDWI_collection = image_collection.ndwi #returns NDWI SentinelCollection image collection

        latest_NDWI_image = NDWI_collection.image_grab(-1) #Example showing how class functions work with any SentinelCollection image collection object, returning latest ndwi image
    
    """

    def __init__(self, start_date=None, end_date=None, tile=None, cloud_percentage_threshold=None, nodata_threshold=None, collection=None, boundary=None):
        # if collection is None and (start_date is None or end_date is None or cloud_percentage_threshold is None or nodata_threshold is None):
        if collection is None and (start_date is None or end_date is None or cloud_percentage_threshold is None or nodata_threshold is None):
            raise ValueError("Either provide all required fields (start_date, end_date, tile, cloud_percentage_threshold, nodata_threshold) or provide a collection.")
        if tile is None and boundary is None and collection is None:
            raise ValueError("Provide either tile or boundary/gemoetry specifications to filter the image collection")
        if collection is None:
            self.start_date = start_date
            self.end_date = end_date
            self.tile = tile
            self.boundary = boundary
            self.cloud_percentage_threshold = cloud_percentage_threshold
            self.nodata_threshold = nodata_threshold

            # Filter the collection
            if tile is not None:
                self.collection = self.get_filtered_collection()
            elif boundary is not None:
                self.collection = self.get_boundary_filtered_collection()
        else:
            self.collection = collection

        self.dates_list = self.list_of_dates()
        self.dates = self.dates_list.getInfo()
        self.ndwi_threshold = -1
        self.ndvi_threshold = -1
        self.halite_threshold = -1
        self.gypsum_threshold = -1
        self.masked_clouds_collection = self.masked_clouds_collection()
        self.median = self.median_collection()
        self.mean = self.mean_collection()
        self.max = self.max_collection()
        self.min = self.min_collection()

        # Check if the required bands are available
        first_image = self.collection.first()
        available_bands = first_image.bandNames()

        if available_bands.contains('B3') and available_bands.contains('B8'):
            self.ndwi = self.ndwi_collection(self.ndwi_threshold)
        else:
            self.ndwi = None
            raise ValueError("Insufficient Bands for ndwi calculation")
        
        if available_bands.contains('B4') and available_bands.contains('B8'):
            self.ndvi = self.ndvi_collection(self.ndvi_threshold)
        else:
            self.ndvi = None
            raise ValueError("Insufficient Bands for ndvi calculation")

        if available_bands.contains('B4') and available_bands.contains('B11'):
            self.halite = self.halite_collection(self.halite_threshold)
        else:
            self.halite = None
            raise ValueError("Insufficient Bands for halite calculation")

        if available_bands.contains('B11') and available_bands.contains('B12'):
            self.gypsum = self.gypsum_collection(self.gypsum_threshold)
        else:
            self.gypsum = None
            raise ValueError("Insufficient Bands for gypsum calculation")

    @staticmethod
    def image_dater(image):
        """
        Adds date to image properties as 'Date_Filter'

        Args: 
        image (eeImage): input image

        Use: to be mapped over an image collection

        Returns: 
        image (eeImage): image with date in properties
        """
        date = ee.Number(image.date().format('YYYY-MM-dd'))
        return image.set({'Date_Filter': date})
    
    
    @staticmethod
    def sentinel_ndwi_fn(image, threshold):
        """
        Function to calculate ndwi for Landsat imagery

        Args: 
        image (eeImage): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked

        Returns:
        image: ndwi image
        """
        ndwi_calc = image.normalizedDifference(['B3', 'B8']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        return water

    @staticmethod
    def sentinel_ndvi_fn(image, threshold):
        """
        Function to calculate ndvi for Landsat imagery

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked

        Returns:
        image: ndvi eeImage
        """
        ndvi_calc = image.normalizedDifference(['B8', 'B4']) #NIR-RED/NIR+RED -- full NDVI image
        vegetation = ndvi_calc.updateMask(ndvi_calc.gte(threshold)).rename('ndvi').copyProperties(image) # subsets the image to just water pixels, 0.2 threshold for datasets
        return vegetation

    @staticmethod
    def sentinel_halite_fn(image, threshold):
        """
        Function to calculate halite index for Landsat imagery. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked

        Returns:
        image: halite eeImage
        """
        halite_index = image.normalizedDifference(['B4', 'B11'])
        halite = halite_index.updateMask(halite_index.gte(threshold)).rename('halite').copyProperties(image)
        return halite

    @staticmethod
    def sentinel_gypsum_fn(image, threshold):
        """
        Function to calculate gypsum index for Landsat imagery. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked

        Returns:
        image: gypsum eeImage
        """
        gypsum_index = image.normalizedDifference(['B11', 'B12'])
        gypsum = gypsum_index.updateMask(gypsum_index.gte(threshold)).rename('gypsum').copyProperties(image)
        return gypsum
    
    @staticmethod
    def MaskCloudsS2(image):
        """
        Function to map clouds

        Args:
        image: input eeImage

        Returns:
        image: output eeImage with clouds masked
        """
        SCL = image.select('SCL')
        CloudMask = SCL.neq(9)
        return image.updateMask(CloudMask).copyProperties(image)
    
    @staticmethod
    def MaskWaterS2(image):
        """
        Function to mask water pixels

        Args:
        image: input eeImage

        Returns:
        image: output eeImage with water pixels masked
        """
        SCL = image.select('SCL')
        WaterMask = SCL.neq(6)
        return image.updateMask(WaterMask).copyProperties(image)
    
    @staticmethod
    def PixelAreaSum(image, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) and store the value as image property (matching name of chosen band)

        Args:
        image: input eeImage
        band_name: name of band (string) for calculating area
        geometry: eeGeometry object denoting area to clip to for area calculation
        threshold: integer threshold to specify masking of pixels below threshold (defaults to -1)
        scale: integer scale of image resolution (meters) (defaults to 30)
        maxPixels: integer denoting maximum number of pixels for calculations
        
        Returns:
        image: eeImage with area calculation stored as property matching name of band
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
    
    def get_filtered_collection(self):
        """
        Function to filter image collection based on LandsatCollection class arguments

        Args:
        self: self is passed into argument

        Returns:
        image collection: eeImageCollection
        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('MGRS_TILE', [self.tile])).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_boundary_filtered_collection(self):
        """
        Function to filter and mask image collection based on LandsatCollection class arguments

        Args:
        self: self is passed into argument

        Returns:
        image collection: eeImageCollection

        """
        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        filtered_collection = sentinel2.filterDate(self.start_date, self.end_date).filterBounds(self.boundary).filter(ee.Filter.lte('NODATA_PIXEL_PERCENTAGE', self.nodata_threshold)) \
                                                        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_percentage_threshold)).map(Sentinel2Collection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def median_collection(self):
        """
        Function to calculate median image from image collection

        Args:
        self: self is passed into argument

        Returns:
        image: median eeImage
        """
        return self.collection.median()
    
    def mean_collection(self):
        """
        Function to calculate mean image from image collection

        Args:
        self: self is passed into argument

        Returns:
        image: mean eeImage

        """
        return self.collection.mean()
    
    def max_collection(self):
        """
        Function to calculate max image from image collection

        Args:
        self: self is passed into argument

        Returns:
        image: max eeImage
        """
        return self.collection.max()
    
    def min_collection(self):
        """
        Function to calculate min image from image collection
        
        Args:
        self: self is passed into argument

        Returns:
        image: min eeImage
        """
        return self.collection.min()

    def ndwi_collection(self, threshold):
        """
        Function to calculate ndwi and return collection as class object

        Args:
        self: self is passed into argument
        threshold: specify threshold for NDWI function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of ndwi imagery
        """
        col =  self.collection.map(lambda image: Sentinel2Collection.sentinel_ndwi_fn(image, threshold=threshold))
        return Sentinel2SubCollection(col, self.dates_list)
    
    def ndvi_collection(self, threshold):
        """
        Function to calculate ndvi and return collection as class object

        Args:
        self: self is passed into argument
        threshold: specify threshold for NDVI function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of ndvi imagery
        """
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_ndvi_fn(image, threshold=threshold))
        return Sentinel2SubCollection(col, self.dates_list)

    def halite_collection(self, threshold):
        """
        Function to calculate halite index and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        threshold: specify threshold for halite function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of halite index imagery
        """
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_halite_fn(image, threshold=threshold))
        return Sentinel2SubCollection(col, self.dates_list)

    def gypsum_collection(self, threshold):
        """
        Function to calculate gypsum index and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        threshold: specify threshold for gypsum function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of gypsum index imagery
        """
        col = self.collection.map(lambda image: Sentinel2Collection.sentinel_gypsum_fn(image, threshold=threshold))
        return Sentinel2SubCollection(col, self.dates_list)

    def masked_water_collection(self):
        """
        Function to mask water and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection: LandsatCollection image collection object of images with water being masked
        """
        col = self.collection.map(Sentinel2Collection.MaskWaterS2)
        return Sentinel2SubCollection(col, self.dates_list)
    
    def masked_clouds_collection(self):
        """
        Function to mask clouds and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection: LandsatCollection image collection objects with clouds being masked
        """
        col = self.collection.map(Sentinel2Collection.MaskCloudsS2)
        return Sentinel2SubCollection(col, self.dates_list)
    
    def mask_with_polygon(self, polygon):
        """
        Function to mask SentinelCollection image collection by a polygon (eeGeometry)

        Args:
        self: self is passed into argument
        polygon: eeGeometry polygon or shape used to mask image collection

        Returns:
        image collection: masked SentinelCollection image collection
        
        """
        # Convert the polygon to a mask
        mask = ee.Image.constant(1).clip(polygon)
        
        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(mask))
        
        # Update the internal collection state
        self.collection = masked_collection
        
        # Return the updated object
        return self

    def list_of_dates(self):
        """
        Function to retrieve list of dates as server-side object

        Args:
        self: self is passed into argument

        Returns:
        list: server-side eeList of dates
        """
        dates = self.collection.aggregate_array('Date_Filter') #.getInfo()
        return dates
    
    def image_grab(self, img_selector):
        """
        Function to select ("grab") image of a specific index from the list of dates. Easy way to get latest image or browse imagery one-by-one.

        Args:
        self: self is passed into argument
        img_selector: index of image in list of dates for which user seeks to "select"
        
        Returns:
        image: eeImage of selected image
        """
        # Convert list to ee.List for server-side operation
        dates_list_ee = ee.List(self.dates_list)
        date = dates_list_ee.get(img_selector)
        new_col = self.collection.filter(ee.Filter.eq('Date_Filter', date))
        return new_col.first()

    def custom_image_grab(self, img_col, img_selector):
        """
        Function to select ("grab") image of a specific index from the list of dates of an eeImageCollection object

        Args:
        self: self is passed into argument
        img_col: eeImageCollection with same dates as another LandsatCollection image collection object
        img_selector: index of image in list of dates for which user seeks to "select"
        
        Returns:
        image: eeImage of selected image
        """
        # Convert list to ee.List for server-side operation
        dates_list_ee = ee.List(self.dates_list)
        date = dates_list_ee.get(img_selector)
        new_col = img_col.filter(ee.Filter.eq('Date_Filter', date))
        return new_col.first()
    
    def image_pick(self, img_date):
        """
        Function to select ("grab") image of a specific date in format of 'YYYY-MM-DD'

        Args:
        self: self is passed into argument
        img_date: date (str) of image to select in format of 'YYYY-MM-DD'

        Returns:
        image: eeImage of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq('Date_Filter', img_date))
        return new_col.first()
    
    def CollectionStitch(self, img_col2):
        """
        Function to mosaic two LandsatCollection objects which share image dates. Mosaics are only formed for dates where both image collections have images. Server-side friendly.

        Args:
        self: self is passed into argument, which is a LandsatCollection image collection
        img_col2: secondary LandsatCollection image collection to be mosaiced with the primary image collection

        Returns:
        image collection: LandsatCollection image collection with mosaiced imagery and image properties from the primary collection
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

class Sentinel2SubCollection(Sentinel2Collection):
    """
    Class to handle returning processed collections back to a class object
    """
    def __init__(self, collection, dates_list):
        self.collection = collection
        self.dates_list = dates_list

    def get_filtered_collection(self):
        return self.collection
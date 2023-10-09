import geemap
import ee
class LandsatCollection:
    """
    Class object representing a collection of Landsat satellite images

    This class provides methods to filter, process, and analyze Landsat satellite imagery for a given period and region

    Arguments:
        start_date (str): Start date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        end_date (str): End date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)
        tile_row (str): MGRS tile row of Landsat image (required unless collection is provided)
        tile_path (str): MGRS tile path of Landsat image (required unless collection is provided)
        boundary (ee geometry): Boundary for filtering images to images that intersect with the boundary shape (optional)
        cloud_percentage_threshold (int): Integer percentage threshold where only imagery with cloud % less than threshold will be provided (optional; defaults to 15)
        collection (ee ImageCollection): Optional argument to convert an eeImageCollection object to a LandsatCollection object - will override other arguments!

    Attributes:
        collection (returns: eeImageCollection): Returns an eeImageCollection object from any LandsatCollection image collection object
        dates_list (returns: Server-Side List): Unreadable Earth Engine list of image dates (server-side)
        dates (returns: Client-Side List): Readable pythonic list of image dates (client-side)
        masked_clouds_collection (returns: LandsatCollection image collection): Returns collection with clouds masked (transparent) for each image
        max (returns: eeImage): Returns a temporally reduced max image (calculates max at each pixel)
        median (returns: eeImage): Returns a temporally reduced median image (calculates median at each pixel)
        mean (returns: eeImage): Returns a temporally reduced mean image (calculates mean at each pixel)
        min (returns: eeImage): Returns a temporally reduced min image (calculates min at each pixel)
        gypsum (returns: eeImageCollection): Returns LandsatCollection image collection of singleband gypsum index rasters
        halite (returns: eeImageCollection): Returns LandsatCollection image collection of singleband halite index rasters
        LST (returns: eeImageCollection): Returns LandsatCollection image collection of singleband land-surface-temperature rasters (Celcius)
        ndwi (returns: eeImageCollection): Returns LandsatCollection image collection of singleband NDWI (water) rasters
        ndvi (returns: eeImageCollection): Returns LandsatCollection image collection of singleband NDVI (vegetation) rasters

    Methods:
        median_collection(self)
        mean_collection(self)
        max_collection(self)
        min_collection(self)
        ndwi_collection(self, threshold)
        ndvi_collection(self, threshold)
        halite_collection(self, threshold, ng_threshold=None)
        gypsum_collection(self, threshold, ng_threshold=None)
        masked_water_collection(self)
        masked_clouds_collection(self)
        surface_temperature_collection(self)
        mask_with_polygon(self, polygon)
        mask_halite(self, threshold, ng_threshold=None)
        mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None)
        list_of_dates(self)
        image_grab(self, img_selector)
        custom_image_grab(self, img_col, img_selector)
        image_pick(self, img_date)
        CollectionStitch(self, img_col2)

    Static Methods:
        image_dater(image)
        landsat5bandrename(img)
        landsat_ndwi_fn(image, threshold)
        landsat_ndvi_fn(image, threshold)
        landsat_halite_fn(image, threshold, ng_threshold=None)
        landsat_gypsum_fn(image, threshold, ng_threshold=None)
        MaskWaterLandsat(image)
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
        ```
        image_collection = LandsatCollection(start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

        ee_image_collection = image_collection.collection #returns eeImageCollection from provided argument filters

        latest_image = image_collection.image_grab(-1) #returns latest image in collection as eeImage

        cloud_masked_collection = image_collection.masked_clouds_collection #returns cloud-masked LandsatCollection image collection

        NDWI_collection = image_collection.ndwi #returns NDWI LandsatCollection image collection

        latest_NDWI_image = NDWI_collection.image_grab(-1) #Example showing how class functions work with any LandsatCollection image collection object, returning latest ndwi image
        ```
    """
    def __init__(self, start_date=None, end_date=None, tile_row=None, tile_path=None, boundary=None, cloud_percentage_threshold=None, collection=None):
        if collection is None and (start_date is None or end_date is None or cloud_percentage_threshold is None):
            raise ValueError("Either provide all required fields (start_date, end_date, tile_row, tile_path, cloud_percentage_threshold) or provide a collection.")
        if tile_row is None and tile_path is None and boundary is None and collection is None:
            raise ValueError("Provide either tile or boundary/gemoetry specifications to filter the image collection")
        if collection is None:
            self.start_date = start_date
            self.end_date = end_date
            self.tile_row = tile_row
            self.tile_path = tile_path
            self.boundary = boundary
            self.cloud_percentage_threshold = cloud_percentage_threshold

            # Filter the collection
            if tile_row and tile_path is not None:
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

        if available_bands.contains('SR_B3') and available_bands.contains('SR_B5'):
            self.ndwi = self.ndwi_collection(self.ndwi_threshold)
        else:
            self.ndwi = None
            raise ValueError("Insufficient Bands for ndwi calculation")
        
        if available_bands.contains('SR_B4') and available_bands.contains('SR_B5'):
            self.ndvi = self.ndvi_collection(self.ndvi_threshold)
        else:
            self.ndvi = None
            raise ValueError("Insufficient Bands for ndwi calculation")

        if available_bands.contains('SR_B4') and available_bands.contains('SR_B6'):
            self.halite = self.halite_collection(self.halite_threshold)
        else:
            self.halite = None
            raise ValueError("Insufficient Bands for halite calculation")

        if available_bands.contains('SR_B6') and available_bands.contains('SR_B7'):
            self.gypsum = self.gypsum_collection(self.gypsum_threshold)
        else:
            self.gypsum = None
            raise ValueError("Insufficient Bands for gypsum calculation")

        if available_bands.contains('ST_ATRAN') and available_bands.contains('ST_EMIS') and available_bands.contains('ST_DRAD') and available_bands.contains('ST_TRAD') and available_bands.contains('ST_URAD') :
            self.LST = self.surface_temperature_collection()
        else:
            self.LST = None
            raise ValueError("Insufficient Bands for temperature calculation")



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
    def landsat5bandrename(img):
        """
        Function to rename Landsat 5 bands to match Landsat 8 & 9

        Args: 
        image (eeImage): input image
        
        Returns: 
        image (eeImage): image with renamed bands
        """
        return img.select('SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL').rename('SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL')
    
    @staticmethod
    def landsat_ndwi_fn(image, threshold):
        """
        Function to calculate ndwi for Landsat imagery

        Args: 
        image (eeImage): input image
        threshold (int): value between -1 and 1 where pixels less than threshold will be masked

        Returns:
        image: ndwi image
        """
        ndwi_calc = image.normalizedDifference(['SR_B3', 'SR_B5']) #green-NIR / green+NIR -- full NDWI image
        water = ndwi_calc.updateMask(ndwi_calc.gte(threshold)).rename('ndwi').copyProperties(image) 
        return water

    @staticmethod
    def landsat_ndvi_fn(image, threshold):
        """
        Function to calculate ndvi for Landsat imagery

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked

        Returns:
        image: ndvi eeImage
        """
        ndvi_calc = image.normalizedDifference(['SR_B5', 'SR_B4']) #NIR-RED/NIR+RED -- full NDVI image
        vegetation = ndvi_calc.updateMask(ndvi_calc.gte(threshold)).rename('ndvi').copyProperties(image) # subsets the image to just water pixels, 0.2 threshold for datasets
        return vegetation
    
    @staticmethod
    def landsat_halite_fn(image, threshold, ng_threshold=None):
        """
        Function to calculate halite index for Landsat imagery. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked
        ng_threshold (optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image: halite eeImage
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
        Function to calculate gypsum index for Landsat imagery. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked
        ng_threshold (optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked

        Returns:
        image: gypsum eeImage
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
    def MaskWaterLandsat(image):
        """
        Function to mask water pixels

        Args:
        image: input eeImage

        Returns:
        image: output eeImage with water pixels masked
        """
        WaterBitMask = ee.Number(2).pow(7).int()
        qa = image.select('QA_PIXEL')
        water_extract = qa.bitwiseAnd(WaterBitMask).eq(0)
        masked_image = image.updateMask(water_extract).copyProperties(image)
        return masked_image
    
    @staticmethod
    def halite_mask(image, threshold, ng_threshold=None):
        """
        Function to mask halite pixels after specifying index to isolate/mask-to halite pixels. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        threshold: integer threshold where pixels less than threshold are masked
        ng_threshold (optional): integer threshold to be applied to landsat 8 or 9 where pixels less than threshold are masked 

        Returns:
        image: eeImage where halite pixels are masked (image without halite pixels)
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
        Function to mask both gypsum and halite pixels. Must specify threshold for isolating halite and gypsum pixels. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        image: input eeImage
        halite_threshold: integer threshold for halite where pixels less than threshold are masked
        gypsum_threshold: integer threshold for gypsum where pixels less than threshold are masked
        halite_ng_threshold (optional): integer threshold for halite to be applied to landsat 8 or 9 where pixels less than threshold are masked 
        gypsum_ng_threshold (optional): integer threshold for gypsum to be applied to landsat 8 or 9 where pixels less than threshold are masked 

        Returns:
        image: eeImage where gypsum and halite pixels are masked (image without halite or gypsum pixels)
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
        Function to map clouds

        Args:
        image: input eeImage

        Returns:
        image: output eeImage with clouds masked
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
        Function to rename bands for temperature calculations

        Args:
        img: input eeImage

        Returns:
        image: output eeImage with new bands
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
        Function to calculate land surface temperature (LST) from landsat TIR bands

        Args:
        image: input eeImage

        Returns:
        image: output LST eeImage 
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
        return image.set(band_name, stats.get(band_name))

    @staticmethod
    def dNDWIPixelAreaSum(image, geometry, band_name='ndwi', scale=30, maxPixels=1e12):
        """
        Function to dynamically calulate the summation of area for water pixels of interest and store the value as image property named 'ndwi'
        Uses Otsu thresholding to dynamically choose the best threshold rather than needing to specify threshold.
        Note: An offset of 0.15 is added to the Otsu threshold.

        Args:
        image: input eeImage
        geometry: eeGeometry object denoting area to clip to for area calculation
        band_name: name of ndwi band (string) for calculating area (defaults to 'ndwi')
        scale: integer scale of image resolution (meters) (defaults to 30)
        maxPixels: integer denoting maximum number of pixels for calculations

        Returns:
        image: eeImage with area calculation stored as property matching name of band
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

    def get_filtered_collection(self):
        """
        Function to filter image collection based on LandsatCollection class arguments

        Args:
        self: self is passed into argument

        Returns:
        image collection: eeImageCollection
        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(LandsatCollection.landsat5bandrename)  # Replace with the correct Landsat 5 collection ID
        filtered_collection = landsat8.merge(landsat9).merge(landsat5).filterDate(self.start_date, self.end_date).filter(ee.Filter.And(ee.Filter.eq('WRS_PATH', self.tile_path),
                                ee.Filter.eq('WRS_ROW', self.tile_row))).filter(ee.Filter.lte('CLOUD_COVER', self.cloud_percentage_threshold)).map(LandsatCollection.image_dater).sort('Date_Filter')
        return filtered_collection
    
    def get_boundary_filtered_collection(self):
        """
        Function to filter and mask image collection based on LandsatCollection class arguments

        Args:
        self: self is passed into argument

        Returns:
        image collection: eeImageCollection

        """
        landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(LandsatCollection.landsat5bandrename)  # Replace with the correct Landsat 5 collection ID
        filtered_collection = landsat8.merge(landsat9).merge(landsat5).filterDate(self.start_date, self.end_date).filterBounds(self.boundary).filter(ee.Filter.lte('CLOUD_COVER', self.cloud_percentage_threshold)).map(LandsatCollection.image_dater).sort('Date_Filter')
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
        col = self.collection.map(lambda image: LandsatCollection.landsat_ndwi_fn(image, threshold=threshold))
        return LandsatSubCollection(col, self.dates_list)
    
    def ndvi_collection(self, threshold):
        """
        Function to calculate ndvi and return collection as class object

        Args:
        self: self is passed into argument
        threshold: specify threshold for NDVI function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of ndvi imagery
        """
        col = self.collection.map(lambda image: LandsatCollection.landsat_ndvi_fn(image, threshold=threshold))
        return LandsatSubCollection(col, self.dates_list)

    def halite_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate halite index and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        threshold: specify threshold for halite function (values less than threshold are masked)
        ng_threshold: (optional) specify threshold for Landsat 8&9 halite function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of halite index imagery
        """
        col = self.collection.map(lambda image: LandsatCollection.landsat_halite_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatSubCollection(col, self.dates_list)

    def gypsum_collection(self, threshold, ng_threshold=None):
        """
        Function to calculate gypsum index and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        threshold: specify threshold for gypsum function (values less than threshold are masked)
        ng_threshold: (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollecion image collection object of gypsum index imagery
        """
        col = self.collection.map(lambda image: LandsatCollection.landsat_gypsum_fn(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatSubCollection(col, self.dates_list)

    def masked_water_collection(self):
        """
        Function to mask water and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection: LandsatCollection image collection object of images with water being masked
        """
        col = self.collection.map(LandsatCollection.MaskWaterLandsat)
        return LandsatSubCollection(col, self.dates_list)
    
    def masked_clouds_collection(self):
        """
        Function to mask clouds and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection: LandsatCollection image collection objects with clouds being masked
        """
        col = self.collection.map(LandsatCollection.maskL8clouds)
        return LandsatSubCollection(col, self.dates_list)
    
    def surface_temperature_collection(self):
        """
        Function to calculate LST and return collection as class object

        Args:
        self: self is passed into argument

        Returns:
        image collection: LandsatCollection image collection object of land surface temperature imagery (temperature in Celcius)
        """
        col = self.collection.map(LandsatCollection.temperature_bands).map(LandsatCollection.landsat_LST).map(LandsatCollection.image_dater)
        return LandsatSubCollection(col, self.dates_list)
    
    def mask_with_polygon(self, polygon):
        """
        Function to mask LandsatCollection image collection by a polygon (eeGeometry)

        Args:
        self: self is passed into argument
        polygon: eeGeometry polygon or shape used to mask image collection

        Returns:
        image collection: masked LandsatCollection image collection
        
        """
        # Convert the polygon to a mask
        mask = ee.Image.constant(1).clip(polygon)
        
        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(mask))
        
        # Update the internal collection state
        self.collection = masked_collection
        
        # Return the updated object
        return self

    def mask_halite(self, threshold, ng_threshold=None):
        """
        Function to mask halite and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        threshold: specify threshold for gypsum function (values less than threshold are masked)
        ng_threshold: (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollection image collection object with halite masked
        """
        col = self.collection.map(lambda image: LandsatCollection.halite_mask(image, threshold=threshold, ng_threshold=ng_threshold))
        return LandsatSubCollection(col, self.dates_list)
    
    def mask_halite_and_gypsum(self, halite_threshold, gypsum_threshold, halite_ng_threshold=None, gypsum_ng_threshold=None):
        """
        Function to mask halite and gypsum and return collection as class object. Can specify separate thresholds for Landsat 5 vs 8&9 images

        Args:
        self: self is passed into argument
        halite_threshold: specify threshold for halite function (values less than threshold are masked)
        halite_ng_threshold: (optional) specify threshold for Landsat 8&9 halite function (values less than threshold are masked)
        gypsum_threshold: specify threshold for gypsum function (values less than threshold are masked)
        gypsum_ng_threshold: (optional) specify threshold for Landsat 8&9 gypsum function (values less than threshold are masked)

        Returns:
        image collection: LandsatCollection image collection object with gypsum and halite masked
        """
        col = self.collection.map(lambda image: LandsatCollection.gypsum_and_halite_mask(image, halite_threshold=halite_threshold, gypsum_threshold=gypsum_threshold, halite_ng_threshold=halite_ng_threshold, gypsum_ng_threshold=gypsum_ng_threshold))
        return LandsatSubCollection(col, self.dates_list)

    def list_of_dates(self):
        """
        Function to retrieve list of dates as server-side object

        Args:
        self: self is passed into argument

        Returns:
        list: server-side eeList of dates
        """
        dates = self.collection.aggregate_array('Date_Filter')
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

            # Copy properties from the first image and set the 'Date_Filter' property
            mosaic = mosaic.copyProperties(img).set('Date_Filter', date).set('system:time_start', img.get('system:time_start'))

            return mosaic

        # Map the function over filtered_col1
        new_col = filtered_col1.map(mosaic_images)

        # Return a LandsatCollection instance
        return LandsatCollection(collection=new_col)
    
class LandsatSubCollection(LandsatCollection):
    """
    Class to handle returning processed collections back to a class object
    """
    def __init__(self, collection, dates_list):
        self.collection = collection
        self.dates_list = dates_list

    def get_filtered_collection(self):
        return self.collection
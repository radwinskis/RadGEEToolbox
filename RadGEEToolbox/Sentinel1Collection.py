import ee
import math
import pandas as pd
import numpy as np
class Sentinel1Collection:
    """
    Class object representing a collection of Sentinel-1 C-band Synthetic Aperture Radar (SAR) GRD data at 10 m/px resolution from Google Earth Engine (GEE). Units of backscatter are in decibels (dB) by default.

    This class module provides methods to filter, process, and analyze Sentinel-1 satellite imagery for a given period and region.

    Inspect the documentation or source code for details on the methods and properties available.

    Arguments:
        start_date (str): Start date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)

        end_date (str): End date string in format of yyyy-mm-dd for filtering collection (required unless collection is provided)

        relative_orbit_start (int or list): Relative orbit start number for filtering collection (required unless collection is provided)

        relative_orbit_stop (int or list): Relative orbit stop number for filtering collection (required unless collection is provided)

        instrument_mode (str or list): Instrument mode for filtering collection, with options of IW, EW, or SM (optional - defaults to IW)

        polarization (str or list): Polarization bands in image for filtering collection. Options: ['VV'], ['HH'], ['VV', 'VH'], or ['HH', 'HV'] (optional; default is ['VV', 'VH'])

        bands (str or list): Band(s) of interest in each image (optional, must match polarization type; default is ['VV', 'VH'])

        orbit_direction (str or list): Orbit direction for filtering collection. Options: 'ASCENDING' and/or 'DESCENDING' (required unless collection is provided). For example, ['ASCENDING', 'DESCENDING'] will include both ascending and descending images.

        boundary (ee.Geometry): Boundary for filtering images to images that intersect with the boundary shape (optional)

        resolution_meters (int): Resolution in meters for filtering collection. Options of 10, 25, or 40 (required unless collection is provided; NOTE: this is for filtering the GEE collection, not multilooking/reprojecting)
        
        collection (ee.ImageCollection): Optional argument to provide an ee.ImageCollection object to be converted to a Sentinel1Collection object - will override other arguments!

    Attributes:
        collection: Returns an ee.ImageCollection object from any Sentinel1Collection image collection object
        
        _dates_list: Cache storage for dates_list property attribute

        _dates: Cahce storgage for dates property attribute

        _geometry_masked_collection: Cache storage for mask_to_polygon method

        _geometry_masked_out_collection: Cache storage for mask_out_polygon method

        _median: Cache storage for median property attribute

        _mean: Cache storage for mean property attribute
        
        _max: Cache storage for max property attribute

        _min: Cache storage for min property attribute

        _MosaicByDate: Cache storage for MosaicByDate property attribute

        _PixelAreaSumCollection: Cache storage for PixelAreaSumCollection property attribute

        _speckle_filter: Cache storage for speckle_filter property attribute

        _Sigma0FromDb: Cache storage for Sigma0FromDb property attribute

        _DbFromSigma0: Cache storage for DbFromSigma0 property attribute

        _multilook: Cache storage for multilook property attribute

    Property attributes:
        dates_list (returns: Server-Side List): Unreadable Earth Engine list of image dates (server-side)
        
        dates (returns: Client-Side List): Readable pythonic list of image dates (client-side)

        max (returns: ee.Image): Returns a temporally reduced max image (calculates max at each pixel)
        
        median (returns: ee.Image): Returns a temporally reduced median image (calculates median at each pixel)
        
        mean (returns: ee.Image): Returns a temporally reduced mean image (calculates mean at each pixel)
        
        min (returns: ee.Image): Returns a temporally reduced min image (calculates min at each pixel)
        
        MosaicByDate (returns: Sentinel1Collection image collection): Mosaics image collection where images with the same date are mosaiced into the same image. Calculates total cloud percentage for subsequent filtering of cloudy mosaics.

        Sigma0FromDb (returns: Sentinel1Collection image collection): Converts image collection from decibels to sigma0

        DbFromSigma0 (returns: Sentinel1Collection image collection): Converts image collection from sigma0 to decibels

        multilook (returns: Sentinel1Collection image collection): Multilooks image collection by specified number of looks (1, 2, 3, or 4)

        speckle_filter (returns: Sentinel1Collection image collection): Applies speckle filter to image collection

    Methods:
        get_filtered_collection(self)

        get_boundary_filtered_collection(self)
        
        mask_to_polygon(self, polygon)

        mask_out_polygon(self, polygon)

        PixelAreaSumCollection(self, band_name, geometry, threshold, scale, maxPixels)
        
        image_grab(self, img_selector)
        
        custom_image_grab(self, img_col, img_selector)
        
        image_pick(self, img_date)
        
        CollectionStitch(self, img_col2)

        transect_iterator(self, lines, line_names, save_folder_path, reducer='mean', n_segments=None, dist_interval=30, to_pandas=True)

        iterate_zonal_stats(self, coordinates, buffer_size=1, reducer_type='mean', scale=40, tileScale=1, coordinate_names=None, file_path=None, dates=None)

    Static Methods:
        image_dater(image)
        
        PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12)

        multilook_fn(image, looks)

        leesigma(image, KERNEL_SIZE, geometry, Tk=7, sigma=0.9, looks=1)

        extract_transect(image, line, reducer="mean", n_segments=100, dist_interval=None, scale=None, crs=None, crsTransform=None, tileScale=1.0, to_pandas=False)

        transect(image, lines, line_names, reducer='mean', n_segments=None, dist_interval=30, to_pandas=True)

        extract_zonal_stats_from_buffer(image, coordinates, buffer_size=1, reducer_type='mean', scale=40, tileScale=1, coordinate_names=None)


    Usage:
        The Sentinel1Collection object alone acts as a base object for which to further filter or process to indices or spatial reductions
        
        To use the Sentinel1Collection functionality, use any of the built in class attributes or method functions. For example, using class attributes:
       
        image_collection = Sentinel1Collection(start_date, end_date, tile_row, tile_path, cloud_percentage_threshold)

        ee_image_collection = image_collection.collection #returns ee.ImageCollection from provided argument filters

        latest_image = image_collection.image_grab(-1) #returns latest image in collection as ee.Image

        cloud_masked_collection = image_collection.masked_clouds_collection #returns cloud-masked Sentinel1Collection image collection

        NDWI_collection = image_collection.ndwi #returns NDWI Sentinel1Collection image collection

        latest_NDWI_image = NDWI_collection.image_grab(-1) #Example showing how class functions work with any Sentinel1Collection image collection object, returning latest ndwi image
    """
    def __init__(self, start_date=None, end_date=None, relative_orbit_start=None, relative_orbit_stop=None, instrument_mode=None, polarization=None, bands=None, orbit_direction=None, boundary=None, resolution=None, resolution_meters=None, collection=None):
        if collection is None and (start_date is None or end_date is None):
            raise ValueError("Either provide all required fields (start_date, end_date, tile_row, tile_path ; or boundary in place of tiles) or provide a collection.")
        if relative_orbit_start is None and relative_orbit_stop is None and boundary is None is None and collection is None:
            raise ValueError("Provide either tile or boundary/geometry specifications to filter the image collection")
        if collection is None:
            self.start_date = start_date
            self.end_date = end_date
            self.instrument_mode = instrument_mode
            self.relative_orbit_start = relative_orbit_start
            self.relative_orbit_stop = relative_orbit_stop
            self.boundary = boundary
            self.polarization = polarization
            self.orbit_direction = orbit_direction
            self.resolution = resolution
            self.resolution_meters = resolution_meters
            self.bands = bands

            if resolution is None:
                self.resolution = 'H'
            elif resolution not in ['H', 'M']:
                raise ValueError("Resolution must be either 'H' or 'M'")
            else:
                pass

            if resolution_meters is None:
                self.resolution_meters = 10
            elif resolution_meters is not None:
                if resolution_meters not in [10, 25, 40]:
                    raise ValueError("Resolution meters must be either 10, 25, or 40")
                else:
                    self.resolution_meters = resolution_meters
            else:
                pass    

            if orbit_direction is None:
                self.orbit_direction = ['ASCENDING', 'DESCENDING']
            elif orbit_direction == ['ASCENDING', 'DESCENDING']:
                self.orbit_direction = orbit_direction
            elif orbit_direction not in ['ASCENDING', 'DESCENDING']:
                raise ValueError("Orbit direction must be either 'ASCENDING' or 'DESCENDING', or '['ASCENDING', 'DESCENDING']' ")
            else:
                pass

            if instrument_mode is None:
                self.instrument_mode = 'IW'
            elif instrument_mode not in ['IW', 'EW', 'SM']:
                raise ValueError("Instrument mode must be either 'IW', 'EW', or 'SM'")
            else:
                pass

            if polarization is None:
                self.polarization = ['VV', 'VH']
            elif polarization not in [['VV'], ['HH'], ['VV', 'VH'], ['HH', 'HV']]:
                raise ValueError("Polarization must be either ['VV'], ['HH'], ['VV, VH'], or ['HH, HV']")
            else:
                pass

            valid_bands = ['HH', 'HV', 'VV', 'VH', 'angle']

            if bands is not None and isinstance(bands, str):
                bands = [bands]

            if bands is None:
                self.bands = self.polarization
            elif not all(band in valid_bands for band in bands):
                raise ValueError("Band must be either 'HH', 'HV', 'VV', 'VH', or 'angle'")
            elif not all(band in self.polarization for band in bands):
                raise ValueError("Band must be associated with chosen polarization type, currently: "+str(self.polarization))
            else:
                self.bands = bands

            if isinstance(self.instrument_mode, list):
                pass
            else:
                self.instrument_mode = [self.instrument_mode]

            if isinstance(self.relative_orbit_start, list):
                pass
            else:
                self.relative_orbit_start = [self.relative_orbit_start]

            if isinstance(self.relative_orbit_stop, list):
                pass
            else:
                self.relative_orbit_stop = [self.relative_orbit_stop]

            if isinstance(self.polarization, list):
                pass
            else:
                self.polarization = [self.polarization]

            if isinstance(self.orbit_direction, list):
                pass
            else:
                self.orbit_direction = [self.orbit_direction]

            if isinstance(self.bands, list):
                pass
            else:
                self.bands = [self.bands]


            # Filter the collection
            if boundary and relative_orbit_start and relative_orbit_start is not None:
                self.collection = self.get_boundary_and_orbit_filtered_collection()
            elif relative_orbit_start and relative_orbit_start is not None:
                self.collection = self.get_filtered_collection()
            elif boundary is not None:
                self.collection = self.get_boundary_filtered_collection()
        else:
            self.collection = collection

        
        self._dates_list = None
        self._dates = None
        self._geometry_masked_collection = None
        self._geometry_masked_out_collection = None
        self._median = None
        self._mean = None
        self._max = None
        self._min = None
        self._MosaicByDate = None
        self._PixelAreaSumCollection = None
        self._speckle_filter = None
        self._Sigma0FromDb = None
        self._DbFromSigma0 = None
        self._multilook = None

    @staticmethod
    def image_dater(image):
        """
        Adds date to image properties as 'Date_Filter'.

        Args: 
            image (ee.Image): Input image

        Returns: 
            ee.Image: Image with date in properties.
        """
        date = ee.Number(image.date().format('YYYY-MM-dd'))
        return image.set({'Date_Filter': date})
    
    
    @staticmethod
    def PixelAreaSum(image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) in a geometry
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
        stats = final.select('area').updateMask(mask).rename(band_name).reduceRegion(
            reducer = ee.Reducer.sum(),
            geometry= geometry,
            scale=scale,
            maxPixels = maxPixels)
        return image.set(band_name, stats.get(band_name))
    
    def PixelAreaSumCollection(self, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12):
        """
        Function to calculate the summation of area for pixels of interest (above a specific threshold) 
        within a geometry and store the value as image property (matching name of chosen band) for an entire
        image collection.
        The resulting value has units of square meters. 

        Args:
            band_name: name of band (string) for calculating area.
            geometry: ee.Geometry object denoting area to clip to for area calculation.
            threshold: integer threshold to specify masking of pixels below threshold (defaults to -1).
            scale: integer scale of image resolution (meters) (defaults to 30).
            maxPixels: integer denoting maximum number of pixels for calculations.
        
        Returns: 
            ee.Image: Image with area calculation stored as property matching name of band.
        """
        if self._PixelAreaSumCollection is None:
            collection = self.collection
            AreaCollection = collection.map(lambda image: Sentinel1Collection.PixelAreaSum(image, band_name=band_name, geometry=geometry, threshold=threshold, scale=scale, maxPixels=maxPixels))
            self._PixelAreaSumCollection = AreaCollection
        return self._PixelAreaSumCollection
    
    @staticmethod
    def multilook_fn(image, looks):
        if looks not in [1, 2, 3, 4]:
            raise ValueError("Looks must be either 1, 2, 3, or 4, corresponding to 1x1, 2x2, 3x3, or 4x4 multilooking")

        default_projection = image.projection()
        image = image.setDefaultProjection(default_projection)
        looked_image = image.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024).reproject(crs=default_projection, scale=10*looks)

        return looked_image.copyProperties(image).set('number_of_processed_looks', looks)

    def multilook(self, looks):
        """
        Property attribute function to multilook image collection. Results are calculated once per class object then cached for future use.

        Args:
            looks: number of looks to multilook image collection by (int)

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """
        if looks not in [1, 2, 3, 4]:
            raise ValueError("Looks must be either 1, 2, 3, or 4, corresponding to 1x1, 2x2, 3x3, or 4x4 multilooking")
        else:
            pass
        if self._multilook is None:
            collection = self.collection
            looks = looks
            multilook_collection = collection.map(lambda image: Sentinel1Collection.multilook_fn(image, looks=looks))
            self._multilook = multilook_collection
        return Sentinel1Collection(collection=self._multilook)

    @staticmethod
    def leesigma(image, KERNEL_SIZE, geometry=None, Tk=7, sigma=0.9, looks=1):
        """
        Implements the improved lee sigma filter for speckle filtering, adapted from https://github.com/adugnag/gee_s1_ard (by Dr. Adugna Mullissa). 
        See: Lee, J.-S. Wen, J.-H. Ainsworth, T.L. Chen, K.-S. Chen, A.J. Improved sigma filter for speckle filtering of SAR imagery. 
        IEEE Trans. Geosci. Remote Sens. 2009, 47, 202–213.

        Args:
            image (ee.Image): Image for speckle filtering
            KERNEL_SIZE (int): positive odd integer (neighbourhood window size - suggested to use between 3-9)
            geometry (ee.Geometry): Geometry to use for speckle filtering (optional). Defaults to footprint of input image.
            Tk (int): number of bright pixels in a 3x3 window (default is 7)
            sigma (float): noise standard deviation (default is 0.9)
            looks (int): number of looks (1, 2, 3, or 4) corresponding to the input image (default is 1). This does NOT perform multilooking, but rather is used to determine the sigma range for filtering.

        Returns: 
            ee.Image: Speckle filtered image

        """

        #parameters
        Tk = ee.Image.constant(Tk) #number of bright pixels in a 3x3 window
        sigma = 0.9
        enl = 4
        target_kernel = 3
        bandNames = image.bandNames().remove('angle')

        # Use image bounds as default geometry
        if geometry is None:
            geometry = image.geometry()
    
        #compute the 98 percentile intensity 
        z98 = ee.Dictionary(image.select(bandNames).reduceRegion(
                    reducer= ee.Reducer.percentile([98]),
                    geometry= geometry,
                    scale=10,
                    maxPixels=1e13
                )).toImage()
    

        #select the strong scatterers to retain
        brightPixel = image.select(bandNames).gte(z98)
        K = brightPixel.reduceNeighborhood(ee.Reducer.countDistinctNonNull()
                ,ee.Kernel.square(target_kernel/2)) 
        retainPixel = K.gte(Tk)
    
    
        #compute the a-priori mean within a 3x3 local window
        #original noise standard deviation since the data is 5 look
        eta = 1.0/math.sqrt(enl) 
        eta = ee.Image.constant(eta)
        #MMSE applied to estimate the apriori mean
        reducers = ee.Reducer.mean().combine( \
                        reducer2= ee.Reducer.variance(), \
                        sharedInputs= True
                        )
        stats = image.select(bandNames).reduceNeighborhood( \
                        reducer= reducers, \
                            kernel= ee.Kernel.square(target_kernel/2,'pixels'), \
                                optimization= 'window')
        meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat('_mean'))
        varBand = bandNames.map(lambda bandName:  ee.String(bandName).cat('_variance'))
            
        z_bar = stats.select(meanBand)
        varz = stats.select(varBand)
        
        oneImg = ee.Image.constant(1)
        varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(oneImg.add(eta.pow(2)))
        b = varx.divide(varz)
        xTilde = oneImg.subtract(b).multiply(z_bar.abs()).add(b.multiply(image.select(bandNames)))
    
        #step 3: compute the sigma range using lookup tables (J.S.Lee et al 2009) for range and eta values for intensity
        if looks == 1:
            LUT = ee.Dictionary({
                0.5: ee.Dictionary({'I1': 0.436, 'I2': 1.92, 'eta': 0.4057}),
                0.6: ee.Dictionary({'I1': 0.343, 'I2': 2.21, 'eta': 0.4954}),
                0.7: ee.Dictionary({'I1': 0.254, 'I2': 2.582, 'eta': 0.5911}),
                0.8: ee.Dictionary({'I1': 0.168, 'I2': 3.094, 'eta': 0.6966}),
                0.9: ee.Dictionary({'I1': 0.084, 'I2': 3.941, 'eta': 0.8191}),
                0.95: ee.Dictionary({'I1': 0.043, 'I2': 4.840, 'eta': 0.8599})
            })
        elif looks == 2:
            LUT = ee.Dictionary({
                0.5: ee.Dictionary({'I1': 0.582, 'I2': 1.584, 'eta': 0.2763}),
                0.6: ee.Dictionary({'I1': 0.501, 'I2': 1.755, 'eta': 0.3388}),
                0.7: ee.Dictionary({'I1': 0.418, 'I2': 1.972, 'eta': 0.4062}),
                0.8: ee.Dictionary({'I1': 0.327, 'I2': 2.260, 'eta': 0.4810}),
                0.9: ee.Dictionary({'I1': 0.221, 'I2': 2.744, 'eta': 0.5699}),
                0.95: ee.Dictionary({'I1': 0.152, 'I2': 3.206, 'eta': 0.6254})
            })
        elif looks == 3:
            LUT = ee.Dictionary({
                0.5: ee.Dictionary({'I1': 0.652, 'I2': 1.458, 'eta': 0.2222}),
                0.6: ee.Dictionary({'I1': 0.580, 'I2': 1.586, 'eta': 0.2736}),
                0.7: ee.Dictionary({'I1': 0.505, 'I2': 1.751, 'eta': 0.3280}),
                0.8: ee.Dictionary({'I1': 0.419, 'I2': 1.965, 'eta': 0.3892}),
                0.9: ee.Dictionary({'I1': 0.313, 'I2': 2.320, 'eta': 0.4624}),
                0.95: ee.Dictionary({'I1': 0.238, 'I2': 2.656, 'eta': 0.5084})
            })
        elif looks == 4:
            LUT = ee.Dictionary({
                0.5: ee.Dictionary({'I1': 0.694, 'I2': 1.385, 'eta': 0.1921}),
                0.6: ee.Dictionary({'I1': 0.630, 'I2': 1.495, 'eta': 0.2348}),
                0.7: ee.Dictionary({'I1': 0.560, 'I2': 1.627, 'eta': 0.2825}),
                0.8: ee.Dictionary({'I1': 0.480, 'I2': 1.804, 'eta': 0.3354}),
                0.9: ee.Dictionary({'I1': 0.378, 'I2': 2.094, 'eta': 0.3991}),
                0.95: ee.Dictionary({'I1': 0.302, 'I2': 2.360, 'eta': 0.4391})
            })
        else:
            raise ValueError("Invalid number of looks. Please choose from 1, 2, 3, or 4.")

    
        #extract data from lookup
        sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
        I1 = sigmaImage.select('I1')
        I2 = sigmaImage.select('I2')
        #new speckle sigma
        nEta = sigmaImage.select('eta')
        #establish the sigma ranges
        I1 = I1.multiply(xTilde)
        I2 = I2.multiply(xTilde)
    
        #step 3: apply MMSE filter for pixels in the sigma range
        #MMSE estimator
        mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
        z = image.select(bandNames).updateMask(mask)
    
        stats = z.reduceNeighborhood(reducer= reducers, kernel= ee.Kernel.square(KERNEL_SIZE/2,'pixels'), optimization= 'window')
            
        z_bar = stats.select(meanBand)
        varz = stats.select(varBand)
        
        
        varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(oneImg.add(nEta.pow(2)))
        b = varx.divide(varz)
        #if b is negative set it to zero
        new_b = b.where(b.lt(0), 0)
        xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))
    
        #remove the applied masks and merge the retained pixels and the filtered pixels
        xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
        output = ee.Image(xHat).rename(bandNames)
        # return image.addBands(output, None, True)
        return output.copyProperties(image)
    
    
    def speckle_filter(self, KERNEL_SIZE, geometry=None, Tk=7, sigma=0.9, looks=1):
        """
        Property attribute function to apply speckle filter to entire image collection. Results are calculated once per class object then cached for future use.

        Args:
            KERNEL_SIZE (int): positive odd integer (neighbourhood window size - suggested to use between 3-9)
            geometry (ee.Geometry): Geometry to use for speckle filtering (optional). Defaults to footprint of input image.
            Tk (int): number of bright pixels in a 3x3 window (default is 7)
            sigma (float): noise standard deviation (default is 0.9)
            looks (int): number of looks (1, 2, 3, or 4) corresponding to the input image (default is 1). This does NOT perform multilooking, but rather is used to determine the sigma range for filtering.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """
        if self._speckle_filter is None:
            collection = self.collection
            speckle_filtered_collection = collection.map(lambda image: Sentinel1Collection.leesigma(image, KERNEL_SIZE, geometry=geometry, Tk=Tk, sigma=sigma, looks=looks))
            self._speckle_filter = speckle_filtered_collection
        return Sentinel1Collection(collection=self._speckle_filter)
    
    @property
    def Sigma0FromDb(self):
        """
        Property attribute function to convert image collection from decibels to sigma0. Results are calculated once per class object then cached for future use.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """
        def conversion(image):
            image = ee.Image(image)
            band_names = image.bandNames()
            sigma_nought = ee.Image(10).pow(image.divide(ee.Image(10))).rename(band_names).copyProperties(image)
            return sigma_nought

        if self._Sigma0FromDb is None:
            collection = self.collection
            sigma0_collection = collection.map(conversion)
            self._Sigma0FromDb = sigma0_collection
        return Sentinel1Collection(collection=self._Sigma0FromDb)
    
    @property
    def DbFromSigma0(self):
        """
        Property attribute function to convert image collection from decibels to sigma0. Results are calculated once per class object then cached for future use.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """
        def conversion(image):
            image = ee.Image(image)
            band_names = image.bandNames()
            dB = ee.Image(10).multiply(image.log10()).rename(band_names).copyProperties(image)
            return dB

        if self._Sigma0FromDb is None:
            collection = self.collection
            dB_collection = collection.map(conversion)
            self._DbFromSigma0 = dB_collection
        return Sentinel1Collection(collection=self._DbFromSigma0)
    
    @property
    def dates_list(self):
        """
        Property attribute to retrieve list of dates as server-side (GEE) object.

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
        Function to filter image collection based on Sentinel1Collection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from Sentinel1Collection object
        """
        # filtered_collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('instrumentMode', self.instrument_mode)).filter(ee.Filter.And(ee.Filter.inList('relativeOrbitNumber_start', self.relative_orbit_stop),
        #                         ee.Filter.inList('relativeOrbitNumber_stop', self.relative_orbit_stop))).filter(ee.Filter.inList('orbitProperties_pass', self.orbit_direction)).filter(ee.Filter.inList('transmitterReceiverPolarisation', 
        #                         self.polarization)).filter(ee.Filter.eq('resolution', self.resolution)).map(self.image_dater).select(self.band)

        filtered_collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('instrumentMode', self.instrument_mode)).filter(ee.Filter.And(ee.Filter.inList('relativeOrbitNumber_start', self.relative_orbit_start),
                                ee.Filter.inList('relativeOrbitNumber_stop', self.relative_orbit_stop))).filter(ee.Filter.inList('orbitProperties_pass', self.orbit_direction)).filter(ee.Filter.eq('transmitterReceiverPolarisation', 
                                self.polarization)).filter(ee.Filter.eq('resolution_meters', self.resolution_meters)).map(self.image_dater).select(self.bands)
        return filtered_collection
    
    def get_boundary_filtered_collection(self):
        """
        Function to filter and mask image collection based on Sentinel1Collection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from Sentinel1Collection object

        """
        filtered_collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(self.start_date, self.end_date).filterBounds(self.boundary).filter(ee.Filter.inList('instrumentMode', self.instrument_mode)).filter(ee.Filter.inList('orbitProperties_pass', self.orbit_direction)).filter(ee.Filter.eq('transmitterReceiverPolarisation', 
                                self.polarization)).filter(ee.Filter.eq('resolution_meters', self.resolution_meters)).map(self.image_dater).select(self.bands)
        return filtered_collection
    
    def get_boundary_and_orbit_filtered_collection(self):
        """
        Function to filter image collection based on Sentinel1Collection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from Sentinel1Collection object
        """
        # filtered_collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('instrumentMode', self.instrument_mode)).filter(ee.Filter.And(ee.Filter.inList('relativeOrbitNumber_start', self.relative_orbit_stop),
        #                         ee.Filter.inList('relativeOrbitNumber_stop', self.relative_orbit_stop))).filter(ee.Filter.inList('orbitProperties_pass', self.orbit_direction)).filter(ee.Filter.inList('transmitterReceiverPolarisation', 
        #                         self.polarization)).filter(ee.Filter.eq('resolution', self.resolution)).map(self.image_dater).select(self.band)

        filtered_collection = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(self.start_date, self.end_date).filter(ee.Filter.inList('instrumentMode', self.instrument_mode)).filterBounds(self.boundary).filter(ee.Filter.And(ee.Filter.inList('relativeOrbitNumber_start', self.relative_orbit_start),
                                ee.Filter.inList('relativeOrbitNumber_stop', self.relative_orbit_stop))).filter(ee.Filter.inList('orbitProperties_pass', self.orbit_direction)).filter(ee.Filter.eq('transmitterReceiverPolarisation', 
                                self.polarization)).filter(ee.Filter.eq('resolution_meters', self.resolution_meters)).map(self.image_dater).select(self.bands)
        return filtered_collection
    
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
    
    def mask_to_polygon(self, polygon):
        """
        Function to mask Sentinel1Collection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args: (image collection)
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            Sentinel1Collection: masked Sentinel1Collection image collection
        
        """
        if self._geometry_masked_collection is None:
            # Convert the polygon to a mask
            mask = ee.Image.constant(1).clip(polygon)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(mask))
            
            # Update the internal collection state
            self._geometry_masked_collection = Sentinel1Collection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_collection
    
    def mask_out_polygon(self, polygon):
        """
        Function to mask Sentinel1Collection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args: (image collection)
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            Sentinel1Collection: masked Sentinel1Collection image collection
        
        """
        if self._geometry_masked_out_collection is None:
            # Convert the polygon to a mask
            full_mask = ee.Image.constant(1)

            # Use paint to set pixels inside polygon as 0
            area = full_mask.paint(polygon, 0)
            
            # Update the mask of each image in the collection
            masked_collection = self.collection.map(lambda img: img.updateMask(area))
            
            # Update the internal collection state
            self._geometry_masked_out_collection = Sentinel1Collection(collection=masked_collection)
        
        # Return the updated object
        return self._geometry_masked_out_collection
    
    

    def image_grab(self, img_selector):
        """
        Function to select ("grab") an image by index from the collection. Easy way to get latest image or browse imagery one-by-one.

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
            img_col: ee.ImageCollection with same dates as another Sentinel1Collection image collection object.
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
        Function to select ("grab") image of a specific date in format of 'YYYY-MM-DD' - will not work correctly if collection is composed of multiple images of the same date.

        Args:
            img_date: date (str) of image to select in format of 'YYYY-MM-DD'

        Returns: 
            ee.Image: ee.Image of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq('Date_Filter', img_date))
        return new_col.first()

    def CollectionStitch(self, img_col2):
        """
        Function to mosaic two Sentinel1Collection objects which share image dates. 
        Mosaics are only formed for dates where both image collections have images. 
        Image properties are copied from the primary collection. Server-side friendly.

        Args:
            img_col2: secondary Sentinel1Collection image collection to be mosaiced with the primary image collection

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
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

        # Return a Sentinel1Collection instance
        return Sentinel1Collection(collection=new_col)
    
    @property
    def MosaicByDate(self):
        """
        Property attribute function to mosaic collection images that share the same date. 
        The property CLOUD_COVER for each image is used to calculate an overall mean, 
        which replaces the CLOUD_COVER property for each mosaiced image. 
        Server-side friendly. NOTE: if images are removed from the collection from cloud filtering, you may have mosaics composed of only one image.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection with mosaiced imagery and mean CLOUD_COVER as a property
        """
        if self._MosaicByDate is None:
            input_collection = self.collection
            # Function to mosaic images of the same date and accumulate them
            def mosaic_and_accumulate(date, list_accumulator):

                list_accumulator = ee.List(list_accumulator)
                date_filter = ee.Filter.eq('Date_Filter', date)
                date_collection = input_collection.filter(date_filter)
                # Convert the collection to a list
                image_list = date_collection.toList(date_collection.size())

                # Get the image at the specified index
                first_image = ee.Image(image_list.get(0))
                # Create mosaic
                mosaic = date_collection.mosaic().set('Date_Filter', date)

                props_of_interest = ['platform_number', 'instrument', 'instrumentMode', 'orbitNumber_start', 'orbitNumber_stop', 'orbitProperties_pass', 'resolution_meters', 'transmitterReceiverPolarisation','system:time_start', 'crs']

                mosaic = mosaic.setDefaultProjection(first_image.projection()).copyProperties(first_image, props_of_interest)

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array('Date_Filter').distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = Sentinel1Collection(collection=new_col)
            self._MosaicByDate = col

        # Convert the list of mosaics to an ImageCollection
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
                return Sentinel1Collection.ee_to_df(transect)
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
            dist_interval (float): The distance interval used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 10.
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
                transect_data = Sentinel1Collection.extract_transect(image=image, line=line, reducer=reducer, dist_interval=dist_interval, to_pandas=to_pandas)
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
                transect_data = Sentinel1Collection.extract_transect(image=image, line=line, reducer=reducer, n_segments=n_segments, to_pandas=to_pandas)
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
        """Computes and stores the values along a transect for each line in a list of lines for each image in a Sentinel1Collection image collection, then saves the data for each image to a csv file. Builds off of the extract_transect function from the geemap package
            where checks are ran to ensure that the reducer column is present in the transect data. If the reducer column is not present, a column of NaNs is created.
            An ee reducer is used to aggregate the values along the transect, depending on the number of segments or distance interval specified. Defaults to 'mean' reducer.
            Naming conventions for the csv files follows as: "image-date_line-name.csv"

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
        image_collection = self 
        image_collection_dates = self.dates
        for i, date in enumerate(image_collection_dates):
            try:
                print(f"Processing image {i+1}/{len(image_collection_dates)}: {date}")
                image = image_collection.image_grab(i)
                transects_df = Sentinel1Collection.transect(image, lines, line_names, reducer=reducer, n_segments=n_segments, dist_interval=dist_interval, to_pandas=to_pandas)
                image_id = date
                transects_df.to_csv(f'{save_folder_path}{image_id}_transects.csv')
                print(f'{image_id}_transects saved to csv')
            except Exception as e:
                print(f"An error occurred while processing image {i+1}: {e}")

    @staticmethod
    def extract_zonal_stats_from_buffer(image, coordinates, buffer_size=1, reducer_type='mean', scale=40, tileScale=1, coordinate_names=None):
        """
        Function to extract spatial statistics from an image for a list of coordinates, providing individual statistics for each location.
        A radial buffer is applied around each coordinate to extract the statistics, which defaults to 1 meter.
        The function returns a pandas DataFrame with the statistics for each coordinate.

        Args:
            image (ee.Image): The image from which to extract the statistics. Must be a singleband image or else resulting values will all be zero!
            coordinates (list): Single tuple or list of tuples with the decimal degrees coordinates in the format of (longitude, latitude) for which to extract the statistics. NOTE the format needs to be [(x1, y1), (x2, y2), ...].
            buffer_size (int, optional): The radial buffer size around the coordinates in meters. Defaults to 1.
            reducer_type (str, optional): The reducer type to use. Defaults to 'mean'. Options are 'mean', 'median', 'min', and 'max'.
            scale (int, optional): The scale (pixel size) to use in meters. Defaults to 40.
            tileScale (int, optional): The tile scale to use. Defaults to 1.
            coordinate_names (list, optional): A list of strings with the names of the coordinates. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame with the statistics for each coordinate, each column name corresponds to the name of the coordinate feature (which may be blank if no names are supplied).
        """

        # Check if coordinates is a single tuple and convert it to a list of tuples if necessary
        if isinstance(coordinates, tuple) and len(coordinates) == 2:
            coordinates = [coordinates]
        elif not (isinstance(coordinates, list) and all(isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates)):
            raise ValueError("Coordinates must be a list of tuples with two elements each (latitude, longitude).")
        
        # Check if coordinate_names is a list of strings
        if coordinate_names is not None:
            if not isinstance(coordinate_names, list) or not all(isinstance(name, str) for name in coordinate_names):
                raise ValueError("coordinate_names must be a list of strings.")
            if len(coordinate_names) != len(coordinates):
                raise ValueError("coordinate_names must have the same length as the coordinates list.")
        else:
            coordinate_names = [f"Location {i+1}" for i in range(len(coordinates))]

        # Check if the image is a singleband image
        def check_singleband(image):
            band_count = image.bandNames().size()
            return ee.Algorithms.If(band_count.eq(1), image, ee.Image.constant(0))

        # Check if the image is a singleband image
        image = ee.Image(check_singleband(image))

        #Convert coordinates to ee.Geometry.Point, buffer them, and add label/name to feature
        points = [ee.Feature(ee.Geometry.Point([coord[0], coord[1]]).buffer(buffer_size), {'name': str(coordinate_names[i])}) for i, coord in enumerate(coordinates)]
        # Create a feature collection from the buffered points
        features = ee.FeatureCollection(points)
        # Reduce the image to the buffered points - handle different reducer types
        if reducer_type == 'mean':
            img_stats = image.reduceRegions(
                    collection=features,
                    reducer=ee.Reducer.mean(),
                    scale=scale,
                    tileScale=tileScale)
            mean_values = img_stats.getInfo()
            means = []
            names = []
            for feature in mean_values['features']:
                names.append(feature['properties']['name'])
                means.append(feature['properties']['mean'])
            organized_values = pd.DataFrame([means], columns=names)
        elif reducer_type == 'median':
            img_stats = image.reduceRegions(
                    collection=features,
                    reducer=ee.Reducer.median(),
                    scale=scale,
                    tileScale=tileScale)
            median_values = img_stats.getInfo()
            medians = []
            names = []
            for feature in median_values['features']:
                names.append(feature['properties']['name'])
                medians.append(feature['properties']['median'])
            organized_values = pd.DataFrame([medians], columns=names)
        elif reducer_type == 'min':
            img_stats = image.reduceRegions(
                    collection=features,
                    reducer=ee.Reducer.min(),
                    scale=scale,
                    tileScale=tileScale)
            min_values = img_stats.getInfo()
            mins = []
            names = []
            for feature in min_values['features']:
                names.append(feature['properties']['name'])
                mins.append(feature['properties']['min'])
            organized_values = pd.DataFrame([mins], columns=names)
        elif reducer_type == 'max':
            img_stats = image.reduceRegions(
                    collection=features,
                    reducer=ee.Reducer.max(),
                    scale=scale,
                    tileScale=tileScale)
            max_values = img_stats.getInfo()
            maxs = []
            names = []
            for feature in max_values['features']:
                names.append(feature['properties']['name'])
                maxs.append(feature['properties']['max'])
            organized_values = pd.DataFrame([maxs], columns=names)
        else:
            raise ValueError("reducer_type must be one of 'mean', 'median', 'min', or 'max'.")
        return organized_values

    def iterate_zonal_stats(self, coordinates, buffer_size=1, reducer_type='mean', scale=40, tileScale=1, coordinate_names=None, file_path=None, dates=None):
        """
        Function to iterate over a collection of images and extract spatial statistics for a list of coordinates (defaults to mean). Individual statistics are provided for each location.
        A radial buffer is applied around each coordinate to extract the statistics, which defaults to 1 meter.
        The function returns a pandas DataFrame with the statistics for each coordinate and date, or optionally exports the data to a table in .csv format.

        NOTE: The input RadGEEToolbox class object but be a collection of singleband images or else resulting values will all be zero!
        
        Args:
            coordinates (list): Single tuple or a list of tuples with the coordinates as decimal degrees in the format of (longitude, latitude) for which to extract the statistics. NOTE the format needs to be [(x1, y1), (x2, y2), ...].
            buffer_size (int, optional): The radial buffer size in meters around the coordinates. Defaults to 1.
            reducer_type (str, optional): The reducer type to use. Defaults to 'mean'. Options are 'mean', 'median', 'min', and 'max'.
            scale (int, optional): The scale (pixel size) to use in meters. Defaults to 40.
            tileScale (int, optional): The tile scale to use. Defaults to 1.
            coordinate_names (list, optional): A list of strings with the names of the coordinates. Defaults to None.
            file_path (str, optional): The file path to export the data to. Defaults to None. Ensure ".csv" is NOT included in the file name path.
            dates (list, optional): A list of dates for which to extract the statistics. Defaults to None.

        Returns:
            pd.DataFrame: A pandas DataFrame with the statistics for each coordinate and date, each row corresponds to a date and each column to a coordinate.
            .csv file: Optionally exports the data to a table in .csv format. If file_path is None, the function returns the DataFrame - otherwise the function will only export the csv file.
        """
        img_collection = self
        #Create empty DataFrame to accumulate results
        accumulated_df = pd.DataFrame()
        #Check if dates is None, if not use the dates provided
        if dates is None:
            dates = img_collection.dates
        else:
            dates = dates
        #Iterate over the dates and extract the zonal statistics for each date
        for date in dates:
            image = img_collection.collection.filter(ee.Filter.eq('Date_Filter', date)).first()
            single_df = Sentinel1Collection.extract_zonal_stats_from_buffer(image, coordinates, buffer_size=buffer_size, reducer_type=reducer_type, scale=scale, tileScale=tileScale, coordinate_names=coordinate_names)
            single_df['Date'] = date
            single_df.set_index('Date', inplace=True)
            accumulated_df = pd.concat([accumulated_df, single_df])
        #Return the DataFrame or export the data to a .csv file
        if file_path is None:
            return accumulated_df
        else:
            return accumulated_df.to_csv(f'{file_path}.csv')

import ee
import math
import pandas as pd
import numpy as np
import warnings


class Sentinel1Collection:
    """
    Represents a user-defined collection of ESA Sentinel-1 C-band Synthetic Aperture Radar (SAR) GRD data at 10 m/px resolution from Google Earth Engine (GEE). Units of backscatter are in decibels (dB) by default.

    This class enables simplified definition, filtering, masking, and processing of Seninel-1 SAR imagery.
    It supports multiple spatial and temporal filters, multilooking and speckle filtering, caching for efficient computation, and direct conversion between log and linear backscatter scales. It also includes utilities for
    mosaicking, zonal statistics, and transect analysis.

    Initialization can be done by providing filtering parameters or directly passing in a pre-filtered GEE collection.

    Inspect the documentation or source code for details on the methods and properties available.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. Required unless `collection` is provided.
        end_date (str): End date in 'YYYY-MM-DD' format. Required unless `collection` is provided.
        relative_orbit_start (int or list): Relative orbit start number for filtering collection. Required unless `collection` is provided.
        relative_orbit_stop (int or list): Relative orbit stop number for filtering collection. Required unless `collection` is provided.
        instrument_mode (str or list, optional): Instrument mode for filtering collection, with options of 'IW', 'EW', or 'SM'. Defaults to 'IW'
        polarization (str or list, optional): Polarization bands in image for filtering collection. Options: ['VV'], ['HH'], ['VV', 'VH'], or ['HH', 'HV']. Default is ['VV', 'VH'].
        bands (str or list, optional): Desired band(s). Must match polarization type. Default is ['VV', 'VH']
        orbit_direction (str or list): Orbit direction for filtering collection. Options: 'ASCENDING' and/or 'DESCENDING'. Required unless `collection` is provided. For example, ['ASCENDING', 'DESCENDING'] will include both ascending and descending images.
        boundary (ee.Geometry, optional): A geometry for filtering to images that intersect with the boundary shape. Overrides `relative_orbit_start` and `relative_orbit_stop` if provided.
        resolution_meters (int): Resolution in meters for filtering collection. Options of 10, 25, or 40. Required unless collection is provided. NOTE: this is for filtering the GEE collection, not multilooking/reprojecting)
        collection (ee.ImageCollection, optional): A pre-filtered Sentinel-1 ee.ImageCollection object to be converted to a Sentinel1Collection object. Overrides all other filters.

    Attributes:
        collection (ee.ImageCollection): The filtered or user-supplied image collection converted to an ee.ImageCollection object.

    Raises:
        ValueError: Raised if required filter parameters are missing, or if both `collection` and other filters are provided.

    Note:
        See full usage examples in the documentation or notebooks:
        https://github.com/radwinskis/RadGEEToolbox/tree/main/Example%20Notebooks

    Examples:
        >>> from RadGEEToolbox import Sentinel1Collection
        >>> import ee
        >>> ee.Initialize()
        >>> counties = ee.FeatureCollection('TIGER/2018/Counties')
        >>> salt_lake_county = counties.filter(ee.Filter.And(
        ...    ee.Filter.eq('NAME', 'Salt Lake'),
        ...    ee.Filter.eq('STATEFP', '49')))
        >>> salt_lake_geometry = salt_lake_county.geometry()
        >>> SAR_collection = Sentinel1Collection(
        ...    start_date='2024-05-01',
        ...    end_date='2024-05-31',
        ...    instrument_mode='IW',
        ...    polarization=['VV', 'VH'],
        ...    orbit_direction='DESCENDING',
        ...    boundary=salt_lake_geometry,
        ...    resolution_meters=10
        ... )
        >>> latest_image = SAR_collection.image_grab(-1)
        >>> mean_SAR_backscatter = SAR_collection.mean
    """

    def __init__(
        self,
        start_date=None,
        end_date=None,
        relative_orbit_start=None,
        relative_orbit_stop=None,
        instrument_mode=None,
        polarization=None,
        bands=None,
        orbit_direction=None,
        boundary=None,
        resolution=None,
        resolution_meters=None,
        collection=None,
    ):
        if collection is None and (start_date is None or end_date is None):
            raise ValueError(
                "Either provide all required fields (start_date, end_date, tile_row, tile_path ; or boundary in place of tiles) or provide a collection."
            )
        if (
            relative_orbit_start is None
            and relative_orbit_stop is None
            and boundary is None
            and collection is None
        ):
            raise ValueError(
                "Provide either tile or boundary/geometry specifications to filter the image collection"
            )
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
                self.resolution = "H"
            elif resolution not in ["H", "M"]:
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
                self.orbit_direction = ["ASCENDING", "DESCENDING"]
            elif orbit_direction == ["ASCENDING", "DESCENDING"]:
                self.orbit_direction = orbit_direction
            elif orbit_direction not in ["ASCENDING", "DESCENDING"]:
                raise ValueError(
                    "Orbit direction must be either 'ASCENDING' or 'DESCENDING', or '['ASCENDING', 'DESCENDING']' "
                )
            else:
                pass

            if instrument_mode is None:
                self.instrument_mode = "IW"
            elif instrument_mode not in ["IW", "EW", "SM"]:
                raise ValueError("Instrument mode must be either 'IW', 'EW', or 'SM'")
            else:
                pass

            if polarization is None:
                self.polarization = ["VV", "VH"]
            elif polarization not in [["VV"], ["HH"], ["VV", "VH"], ["HH", "HV"]]:
                raise ValueError(
                    "Polarization must be either ['VV'], ['HH'], ['VV, VH'], or ['HH, HV']"
                )
            else:
                pass

            valid_bands = ["HH", "HV", "VV", "VH", "angle"]

            if bands is not None and isinstance(bands, str):
                bands = [bands]

            if bands is None:
                self.bands = self.polarization
            elif not all(band in valid_bands for band in bands):
                raise ValueError(
                    "Band must be either 'HH', 'HV', 'VV', 'VH', or 'angle'"
                )
            elif not all(band in self.polarization for band in bands):
                raise ValueError(
                    "Band must be associated with chosen polarization type, currently: "
                    + str(self.polarization)
                )
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
        self._monthly_median = None
        self._monthly_mean = None
        self._monthly_max = None
        self._monthly_min = None
        self._monthly_sum = None
        self._yearly_median = None
        self._yearly_mean = None
        self._yearly_max = None
        self._yearly_min = None
        self._yearly_sum = None
        self._MosaicByDate = None
        self._PixelAreaSumCollection = None
        self._speckle_filter = None
        self._Sigma0FromDb = None
        self._DbFromSigma0 = None
        self._multilook = None

    def __call__(self):
        """
        Allows the object to be called as a function, returning itself. 
        This enables property-like methods to be accessed with or without parentheses 
        (e.g., .mosaicByDate or .mosaicByDate()).
        """
        return self

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
    def pixelAreaSum(
        image, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12
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
            scale (int): integer scale of image resolution (meters) (defaults to 10)
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
    
    @staticmethod
    def PixelAreaSum(
        image, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12
    ):
        warnings.warn(
            "The 'PixelAreaSum' method is deprecated. Please use 'pixelAreaSum' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return Sentinel1Collection.pixelAreaSum(
            image=image,
            band_name=band_name,
            geometry=geometry,
            threshold=threshold,
            scale=scale,
            maxPixels=maxPixels,
        )

    def pixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12, output_type='ImageCollection', area_data_export_path=None
    ):
        """
        Calculates the geodesic summation of area for pixels of interest (above a specific threshold)
        within a geometry and stores the value as an image property (matching name of chosen band) for an entire
        image collection. Optionally exports the area data to a CSV file.

        NOTE: The resulting value has units of square meters.

        Args:
            band_name (string or list of strings): name of band(s) (string) for calculating area. If providing multiple band names, pass as a list of strings.
            geometry (ee.Geometry): ee.Geometry object denoting area to clip to for area calculation.
            threshold (float): integer threshold to specify masking of pixels below threshold (defaults to -1). If providing multiple band names, the same threshold will be applied to all bands. Best practice in this case is to mask the bands prior to passing to this function and leave threshold at default of -1.
            scale (int): integer scale of image resolution (meters) (defaults to 30).
            maxPixels (int): integer denoting maximum number of pixels for calculations.
            output_type (str): 'ImageCollection' or 'ee.ImageCollection' to return an ee.ImageCollection, 'Sentinel1Collection' to return a Sentinel1Collection object, or 'DataFrame', 'Pandas', 'pd', 'dataframe', 'df' to return a pandas DataFrame (defaults to 'ImageCollection').
            area_data_export_path (str, optional): If provided, the function will save the resulting area data to a CSV file at the specified path.

        Returns:
            ee.ImageCollection or Sentinel1Collection: Image collection of images with area calculation (square meters) stored as property matching name of band. Type of output depends on output_type argument.
        """
        # If the area calculation has not been computed for this Sentinel1Collection instance, the area will be calculated for the provided bands
        if self._PixelAreaSumCollection is None:
            collection = self.collection
            # Area calculation for each image in the collection, using the PixelAreaSum function
            AreaCollection = collection.map(
                lambda image: Sentinel1Collection.pixelAreaSum(
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

        prop_names = band_name if isinstance(band_name, list) else [band_name]

        # If an export path is provided, the area data will be exported to a CSV file
        if area_data_export_path:
            Sentinel1Collection(collection=self._PixelAreaSumCollection).exportProperties(property_names=prop_names, file_path=area_data_export_path+'.csv')
        # Returning the result in the desired format based on output_type argument or raising an error for invalid input
        if output_type == 'ImageCollection' or output_type == 'ee.ImageCollection':
            return self._PixelAreaSumCollection
        elif output_type == 'Sentinel1Collection':
            return Sentinel1Collection(collection=self._PixelAreaSumCollection)
        elif output_type == 'DataFrame' or output_type == 'Pandas' or output_type == 'pd' or output_type == 'dataframe' or output_type == 'df':
            return Sentinel1Collection(collection=self._PixelAreaSumCollection).exportProperties(property_names=prop_names)
        else:
            raise ValueError("Incorrect `output_type`. The `output_type` argument must be one of the following: 'ImageCollection', 'ee.ImageCollection', 'Sentinel1Collection', 'DataFrame', 'Pandas', 'pd', 'dataframe', or 'df'.")

    def PixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=10, maxPixels=1e12, output_type='ImageCollection', area_data_export_path=None
    ):
        warnings.warn(
            "The 'PixelAreaSumCollection' method is deprecated. Please use 'pixelAreaSumCollection' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.pixelAreaSumCollection(
            band_name=band_name,
            geometry=geometry,
            threshold=threshold,
            scale=scale,
            maxPixels=maxPixels,
            output_type=output_type,
            area_data_export_path=area_data_export_path
        )

    @staticmethod
    def add_month_property_fn(image):
        """
        Adds a numeric 'month' property to the image based on its date.

        Args:
            image (ee.Image): Input image.

        Returns:
            ee.Image: Image with the 'month' property added.
        """
        return image.set('month', image.date().get('month'))

    @property
    def add_month_property(self):
        """
        Adds a numeric 'month' property to each image in the collection.

        Returns:
            Sentinel1Collection: A Sentinel1Collection image collection with the 'month' property added to each image.
        """
        col = self.collection.map(Sentinel1Collection.add_month_property_fn)
        return Sentinel1Collection(collection=col)


    def combine(self, other):
        """
        Combines the current Sentinel1Collection with another Sentinel1Collection, using the `combine` method.

        Args:
            other (Sentinel1Collection): Another Sentinel1Collection to combine with current collection.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection containing images from both collections.
        """
        # Checking if 'other' is an instance of Sentinel1Collection
        if not isinstance(other, Sentinel1Collection):
            raise ValueError("The 'other' parameter must be an instance of Sentinel1Collection.")
        
        # Merging the collections using the .combine() method
        merged_collection = self.collection.combine(other.collection)
        return Sentinel1Collection(collection=merged_collection)

    def merge(self, collections=None, multiband_collection=None, date_key='Date_Filter'):
        """
        Merge many singleband Sentinel1Collection products into the parent collection, 
        or merge a single multiband collection with parent collection,
        pairing images by exact Date_Filter and returning one multiband image per date.

        NOTE: if you want to merge two multiband collections, use the `combine` method instead.

        Args:
            collections (list): List of singleband collections to merge with parent collection, effectively adds one band per collection to each image in parent
            multiband_collection (Sentinel1Collection, optional): A multiband collection to merge with parent. Specifying a collection here will override `collections`.
            date_key (str): image property key for exact pairing (default 'Date_Filter')

        Returns:
            Sentinel1Collection: parent with extra single bands attached (one image per date)
        """

        if collections is None and multiband_collection is not None:
            # Exact-date inner-join merge of two collections (adds ALL bands from 'other').
            join = ee.Join.inner()
            flt  = ee.Filter.equals(leftField=date_key, rightField=date_key)
            paired = join.apply(self.collection, multiband_collection.collection, flt)

            def _pair_two(f):
                f = ee.Feature(f)
                a = ee.Image(f.get('primary'))
                b = ee.Image(f.get('secondary'))
                # Overwrite on name collision
                merged = a.addBands(b, None, True)
                # Keep parent props + date key
                merged = merged.copyProperties(a, a.propertyNames()).set('system:time_start', a.get('system:time_start'))
                merged = merged.set(date_key, a.get(date_key))
                return ee.Image(merged)

            return Sentinel1Collection(collection=ee.ImageCollection(paired.map(_pair_two)))

        # Preferred path: merge many singleband products into the parent
        if not isinstance(collections, list) or len(collections) == 0:
            raise ValueError("Provide a non-empty list of Sentinel1Collection objects in `collections`.")

        result = self.collection
        for extra in collections:
            if not isinstance(extra, Sentinel1Collection):
                raise ValueError("All items in `collections` must be Sentinel1Collection objects.")

            join = ee.Join.inner()
            flt  = ee.Filter.equals(leftField=date_key, rightField=date_key)
            paired = join.apply(result, extra.collection, flt)

            def _attach_one(f):
                f = ee.Feature(f)
                parent = ee.Image(f.get('primary'))
                sb     = ee.Image(f.get('secondary'))
                # Assume singleband product; grab its first band name server-side
                bname  = ee.String(sb.bandNames().get(0))
                # Add the single band; overwrite if the name already exists in parent
                merged = parent.addBands(sb.select([bname]).rename([bname]), None, True)
                # Preserve parent props + date key
                merged = merged.copyProperties(parent, parent.propertyNames()).set('system:time_start', parent.get('system:time_start'))
                merged = merged.set(date_key, parent.get(date_key))
                return ee.Image(merged)

            result = ee.ImageCollection(paired.map(_attach_one))

        return Sentinel1Collection(collection=result)


    @staticmethod
    def multilook_fn(image, looks):
        if looks not in [1, 2, 3, 4]:
            raise ValueError(
                "Looks must be either 1, 2, 3, or 4, corresponding to 1x1, 2x2, 3x3, or 4x4 multilooking"
            )

        default_projection = image.projection()
        image = image.setDefaultProjection(default_projection)
        looked_image = image.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=1024
        ).reproject(crs=default_projection, scale=10 * looks)

        return looked_image.copyProperties(image).set(
            "number_of_processed_looks", looks
        )

    def multilook(self, looks):
        """
        Multilooks a Sentinel-1 SAR image collection. Results are calculated once per class object then cached for future use.

        Args:
            looks (int): number of looks to multilook image collection by (int). A looks value of 1 will not multilook the image collection, while a value of 2, 3, or 4 will multilook the image collection by 2x2, 3x3, or 4x4 respectively.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """
        if looks not in [1, 2, 3, 4]:
            raise ValueError(
                "Looks must be either 1, 2, 3, or 4, corresponding to 1x1, 2x2, 3x3, or 4x4 multilooking"
            )
        else:
            pass
        if self._multilook is None:
            collection = self.collection
            looks = looks
            multilook_collection = collection.map(
                lambda image: Sentinel1Collection.multilook_fn(image, looks=looks)
            )
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

        # parameters
        Tk = ee.Image.constant(Tk)  # number of bright pixels in a 3x3 window
        sigma = 0.9
        enl = 4
        target_kernel = 3
        bandNames = image.bandNames().remove("angle")

        # Use image bounds as default geometry
        if geometry is None:
            geometry = image.geometry()

        # compute the 98 percentile intensity
        z98 = ee.Dictionary(
            image.select(bandNames).reduceRegion(
                reducer=ee.Reducer.percentile([98]),
                geometry=geometry,
                scale=10,
                maxPixels=1e13,
            )
        ).toImage()

        # select the strong scatterers to retain
        brightPixel = image.select(bandNames).gte(z98)
        K = brightPixel.reduceNeighborhood(
            ee.Reducer.countDistinctNonNull(), ee.Kernel.square(target_kernel / 2)
        )
        retainPixel = K.gte(Tk)

        # compute the a-priori mean within a 3x3 local window
        # original noise standard deviation since the data is 5 look
        eta = 1.0 / math.sqrt(enl)
        eta = ee.Image.constant(eta)
        # MMSE applied to estimate the apriori mean
        reducers = ee.Reducer.mean().combine(
            reducer2=ee.Reducer.variance(), sharedInputs=True
        )
        stats = image.select(bandNames).reduceNeighborhood(
            reducer=reducers,
            kernel=ee.Kernel.square(target_kernel / 2, "pixels"),
            optimization="window",
        )
        meanBand = bandNames.map(lambda bandName: ee.String(bandName).cat("_mean"))
        varBand = bandNames.map(lambda bandName: ee.String(bandName).cat("_variance"))

        z_bar = stats.select(meanBand)
        varz = stats.select(varBand)

        oneImg = ee.Image.constant(1)
        varx = (varz.subtract(z_bar.abs().pow(2).multiply(eta.pow(2)))).divide(
            oneImg.add(eta.pow(2))
        )
        b = varx.divide(varz)
        xTilde = (
            oneImg.subtract(b)
            .multiply(z_bar.abs())
            .add(b.multiply(image.select(bandNames)))
        )

        # step 3: compute the sigma range using lookup tables (J.S.Lee et al 2009) for range and eta values for intensity
        if looks == 1:
            LUT = ee.Dictionary(
                {
                    0.5: ee.Dictionary({"I1": 0.436, "I2": 1.92, "eta": 0.4057}),
                    0.6: ee.Dictionary({"I1": 0.343, "I2": 2.21, "eta": 0.4954}),
                    0.7: ee.Dictionary({"I1": 0.254, "I2": 2.582, "eta": 0.5911}),
                    0.8: ee.Dictionary({"I1": 0.168, "I2": 3.094, "eta": 0.6966}),
                    0.9: ee.Dictionary({"I1": 0.084, "I2": 3.941, "eta": 0.8191}),
                    0.95: ee.Dictionary({"I1": 0.043, "I2": 4.840, "eta": 0.8599}),
                }
            )
        elif looks == 2:
            LUT = ee.Dictionary(
                {
                    0.5: ee.Dictionary({"I1": 0.582, "I2": 1.584, "eta": 0.2763}),
                    0.6: ee.Dictionary({"I1": 0.501, "I2": 1.755, "eta": 0.3388}),
                    0.7: ee.Dictionary({"I1": 0.418, "I2": 1.972, "eta": 0.4062}),
                    0.8: ee.Dictionary({"I1": 0.327, "I2": 2.260, "eta": 0.4810}),
                    0.9: ee.Dictionary({"I1": 0.221, "I2": 2.744, "eta": 0.5699}),
                    0.95: ee.Dictionary({"I1": 0.152, "I2": 3.206, "eta": 0.6254}),
                }
            )
        elif looks == 3:
            LUT = ee.Dictionary(
                {
                    0.5: ee.Dictionary({"I1": 0.652, "I2": 1.458, "eta": 0.2222}),
                    0.6: ee.Dictionary({"I1": 0.580, "I2": 1.586, "eta": 0.2736}),
                    0.7: ee.Dictionary({"I1": 0.505, "I2": 1.751, "eta": 0.3280}),
                    0.8: ee.Dictionary({"I1": 0.419, "I2": 1.965, "eta": 0.3892}),
                    0.9: ee.Dictionary({"I1": 0.313, "I2": 2.320, "eta": 0.4624}),
                    0.95: ee.Dictionary({"I1": 0.238, "I2": 2.656, "eta": 0.5084}),
                }
            )
        elif looks == 4:
            LUT = ee.Dictionary(
                {
                    0.5: ee.Dictionary({"I1": 0.694, "I2": 1.385, "eta": 0.1921}),
                    0.6: ee.Dictionary({"I1": 0.630, "I2": 1.495, "eta": 0.2348}),
                    0.7: ee.Dictionary({"I1": 0.560, "I2": 1.627, "eta": 0.2825}),
                    0.8: ee.Dictionary({"I1": 0.480, "I2": 1.804, "eta": 0.3354}),
                    0.9: ee.Dictionary({"I1": 0.378, "I2": 2.094, "eta": 0.3991}),
                    0.95: ee.Dictionary({"I1": 0.302, "I2": 2.360, "eta": 0.4391}),
                }
            )
        else:
            raise ValueError(
                "Invalid number of looks. Please choose from 1, 2, 3, or 4."
            )

        # extract data from lookup
        sigmaImage = ee.Dictionary(LUT.get(str(sigma))).toImage()
        I1 = sigmaImage.select("I1")
        I2 = sigmaImage.select("I2")
        # new speckle sigma
        nEta = sigmaImage.select("eta")
        # establish the sigma ranges
        I1 = I1.multiply(xTilde)
        I2 = I2.multiply(xTilde)

        # step 3: apply MMSE filter for pixels in the sigma range
        # MMSE estimator
        mask = image.select(bandNames).gte(I1).Or(image.select(bandNames).lte(I2))
        z = image.select(bandNames).updateMask(mask)

        stats = z.reduceNeighborhood(
            reducer=reducers,
            kernel=ee.Kernel.square(KERNEL_SIZE / 2, "pixels"),
            optimization="window",
        )

        z_bar = stats.select(meanBand)
        varz = stats.select(varBand)

        varx = (varz.subtract(z_bar.abs().pow(2).multiply(nEta.pow(2)))).divide(
            oneImg.add(nEta.pow(2))
        )
        b = varx.divide(varz)
        # if b is negative set it to zero
        new_b = b.where(b.lt(0), 0)
        xHat = oneImg.subtract(new_b).multiply(z_bar.abs()).add(new_b.multiply(z))

        # remove the applied masks and merge the retained pixels and the filtered pixels
        xHat = image.select(bandNames).updateMask(retainPixel).unmask(xHat)
        output = ee.Image(xHat).rename(bandNames)
        # return image.addBands(output, None, True)
        return output.copyProperties(image).set('system:time_start', image.get('system:time_start'))

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
            speckle_filtered_collection = collection.map(
                lambda image: Sentinel1Collection.leesigma(
                    image,
                    KERNEL_SIZE,
                    geometry=geometry,
                    Tk=Tk,
                    sigma=sigma,
                    looks=looks,
                )
            )
            self._speckle_filter = speckle_filtered_collection
        return Sentinel1Collection(collection=self._speckle_filter)

    @property
    def sigma0FromDb(self):
        """
        Property attribute function to convert image collection from decibels to sigma0. Results are calculated once per class object then cached for future use.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """

        def conversion(image):
            image = ee.Image(image)
            band_names = image.bandNames()
            sigma_nought = (
                ee.Image(10)
                .pow(image.divide(ee.Image(10)))
                .rename(band_names)
                .copyProperties(image)
                .set('system:time_start', image.get('system:time_start'))
            )
            return sigma_nought

        if self._Sigma0FromDb is None:
            collection = self.collection
            sigma0_collection = collection.map(conversion)
            self._Sigma0FromDb = sigma0_collection
        return Sentinel1Collection(collection=self._Sigma0FromDb)
    
    @property
    def Sigma0FromDb(self):
        warnings.warn(
            "The 'Sigma0FromDb' property is deprecated. Please use 'sigma0FromDb' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.sigma0FromDb

    @property
    def dbFromSigma0(self):
        """
        Property attribute function to convert image collection from decibels to sigma0. Results are calculated once per class object then cached for future use.

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
        """

        def conversion(image):
            image = ee.Image(image)
            band_names = image.bandNames()
            dB = (
                ee.Image(10)
                .multiply(image.log10())
                .rename(band_names)
                .copyProperties(image)
                .set('system:time_start', image.get('system:time_start'))
            )
            return dB

        if self._DbFromSigma0 is None:
            collection = self.collection
            dB_collection = collection.map(conversion)
            self._DbFromSigma0 = dB_collection
        return Sentinel1Collection(collection=self._DbFromSigma0)
    
    @property
    def DbFromSigma0(self):
        warnings.warn(
            "The 'DbFromSigma0' property is deprecated. Please use 'dbFromSigma0' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.dbFromSigma0
    
    @staticmethod
    def anomaly_fn(image, geometry, band_name=None, anomaly_band_name=None, replace=True, scale=10):
        """
        Calculates the anomaly of a singleband image compared to the mean of the singleband image.

        This function computes the anomaly for each band in the input image by
        subtracting the mean value of that band from a provided image.
        The anomaly is a measure of how much the pixel values deviate from the
        average conditions represented by the mean of the image.

        Args:
            image (ee.Image): An ee.Image for which the anomaly is to be calculated.
                It is assumed that this image is a singleband image.
            geometry (ee.Geometry): The geometry for image reduction to define the mean value to be used for anomaly calculation.
            band_name (str, optional): A string representing the band name to be used for the output anomaly image. If not provided, the band name of the first band of the input image will be used.
            anomaly_band_name (str, optional): A string representing the band name to be used for the output anomaly image. If not provided, the band name of the first band of the input image will be used.
            replace (bool, optional): A boolean indicating whether to replace the original band with the anomaly band in the output image. If True, the output image will contain only the anomaly band. If False, the output image will contain both the original band and the anomaly band. Default is True.
            scale (int, optional): The scale (in meters) to use for the image reduction. Default is 10.

        Returns:
            ee.Image: An ee.Image where each band represents the anomaly (deviation from
                        the mean) for that band. The output image retains the same band name.
        """
        if band_name:
            band_name = band_name
        else:
            band_name = ee.String(image.bandNames().get(0))

        image_to_process = image.select([band_name])

        # Calculate the mean image of the provided collection.
        mean_image = image_to_process.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e13
        ).toImage()

        # Compute the anomaly by subtracting the mean image from the input image.
        if scale == 10:
            anomaly_image = image_to_process.subtract(mean_image)
        else:
            anomaly_image = image_to_process.reproject(crs=image_to_process.projection(), scale=scale).subtract(mean_image)

        if anomaly_band_name is None:
            if band_name:
                anomaly_image = anomaly_image.rename(band_name) 
            else:
                # Preserve original properties from the input image.
                anomaly_image = anomaly_image.rename(ee.String(image.bandNames().get(0))) 
        else:
            anomaly_image = anomaly_image.rename(anomaly_band_name) 
        # return anomaly_image
        if replace:
            return anomaly_image.copyProperties(image).set('system:time_start', image.get('system:time_start'))
        else:
            return image.addBands(anomaly_image, overwrite=True).copyProperties(image).set('system:time_start', image.get('system:time_start'))

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
    
    def exportProperties(self, property_names, file_path=None):
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
        elif isinstance(property_names, list):
            property_names = property_names

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
        df = Sentinel1Collection.ee_to_df(feature_collection, columns=all_properties_to_fetch)
        
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
    
    def ExportProperties(self, property_names, file_path=None):
        warnings.warn(
            "The 'ExportProperties' method is deprecated. Please use 'exportProperties' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.exportProperties(property_names=property_names, file_path=file_path)

    def get_filtered_collection(self):
        """
        Function to filter image collection based on Sentinel1Collection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from Sentinel1Collection object
        """

        filtered_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.inList("instrumentMode", self.instrument_mode))
            .filter(
                ee.Filter.And(
                    ee.Filter.inList(
                        "relativeOrbitNumber_start", self.relative_orbit_start
                    ),
                    ee.Filter.inList(
                        "relativeOrbitNumber_stop", self.relative_orbit_stop
                    ),
                )
            )
            .filter(ee.Filter.inList("orbitProperties_pass", self.orbit_direction))
            .filter(ee.Filter.eq("transmitterReceiverPolarisation", self.polarization))
            .filter(ee.Filter.eq("resolution_meters", self.resolution_meters))
            .map(self.image_dater)
            .select(self.bands)
        )
        return filtered_collection

    def get_boundary_filtered_collection(self):
        """
        Function to filter and mask image collection based on Sentinel1Collection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from Sentinel1Collection object

        """
        filtered_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(self.start_date, self.end_date)
            .filterBounds(self.boundary)
            .filter(ee.Filter.inList("instrumentMode", self.instrument_mode))
            .filter(ee.Filter.inList("orbitProperties_pass", self.orbit_direction))
            .filter(ee.Filter.eq("transmitterReceiverPolarisation", self.polarization))
            .filter(ee.Filter.eq("resolution_meters", self.resolution_meters))
            .map(self.image_dater)
            .select(self.bands)
        )
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

        filtered_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.inList("instrumentMode", self.instrument_mode))
            .filterBounds(self.boundary)
            .filter(
                ee.Filter.And(
                    ee.Filter.inList(
                        "relativeOrbitNumber_start", self.relative_orbit_start
                    ),
                    ee.Filter.inList(
                        "relativeOrbitNumber_stop", self.relative_orbit_stop
                    ),
                )
            )
            .filter(ee.Filter.inList("orbitProperties_pass", self.orbit_direction))
            .filter(ee.Filter.eq("transmitterReceiverPolarisation", self.polarization))
            .filter(ee.Filter.eq("resolution_meters", self.resolution_meters))
            .map(self.image_dater)
            .select(self.bands)
        )
        return filtered_collection
    
    def remove_duplicate_dates(self, sort_by='system:time_start', ascending=True):
        """
        Removes duplicate images that share the same date, keeping only the first one encountered.
        Useful for handling duplicate Sentinel-1A/1B acquisitions or overlapping tiles.
        
        Args:
            sort_by (str): Property to sort by before filtering distinct dates. 
                           Defaults to 'system:time_start'. Take care to provide a property that exists in all images if using a custom property.
            ascending (bool): Sort order. Defaults to True.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with distinct dates.
        """
        # Sort the collection to ensure the "best" image comes first (e.g. least cloudy)
        sorted_col = self.collection.sort(sort_by, ascending)
        
        # distinct() retains the first image for each unique value of the specified property
        distinct_col = sorted_col.distinct('Date_Filter')
        
        return Sentinel1Collection(collection=distinct_col)

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
    def monthly_median_collection(self):
        """Creates a monthly median composite from a Sentinel1Collection image collection.

        This function computes the median for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with monthly median composites.
        """
        if self._monthly_median is None:
            collection = self.collection
            # Get the start and end dates of the entire collection.
            date_range = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date = ee.Date(date_range.get('min'))
            end_date = ee.Date(date_range.get('max'))

            # Calculate the total number of months in the date range.
            # The .round() is important for ensuring we get an integer.
            num_months = end_date.difference(start_date, 'month').round()

            # Generate a list of starting dates for each month.
            # This uses a sequence and advances the start date by 'i' months.
            def get_month_start(i):
                return start_date.advance(i, 'month')
            
            month_starts = ee.List.sequence(0, num_months).map(get_month_start)

            # Define a function to map over the list of month start dates.
            def create_monthly_composite(date):
                # Cast the input to an ee.Date object.
                start_of_month = ee.Date(date)
                # The end date is exclusive, so we advance by 1 month.
                end_of_month = start_of_month.advance(1, 'month')

                # Filter the original collection to get images for the current month.
                monthly_subset = collection.filterDate(start_of_month, end_of_month)

                # Count the number of images in the monthly subset.
                image_count = monthly_subset.size()

                # Compute the median. This is robust to outliers like clouds.
                monthly_median = monthly_subset.median()

                # Set essential properties on the resulting composite image.
                # The timestamp is crucial for time-series analysis and charting.
                # The image_count is useful metadata for quality assessment.
                return monthly_median.set({
                    'system:time_start': start_of_month.millis(),
                    'month': start_of_month.get('month'),
                    'year': start_of_month.get('year'),
                    'Date_Filter': start_of_month.format('YYYY-MM-dd'),
                    'image_count': image_count
                })

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = Sentinel1Collection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_median = final_collection
        else:
            pass

        return self._monthly_median
    
    @property
    def monthly_mean_collection(self):
        """Creates a monthly mean composite from a Sentinel1Collection image collection.

        This function computes the mean for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with monthly mean composites.
        """
        if self._monthly_mean is None:
            collection = self.collection
            target_proj = collection.first().projection()
            # Get the start and end dates of the entire collection.
            date_range = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            original_start_date = ee.Date(date_range.get('min'))
            end_date = ee.Date(date_range.get('max'))

            start_year = original_start_date.get('year')
            start_month = original_start_date.get('month')
            start_date = ee.Date.fromYMD(start_year, start_month, 1)

            # Calculate the total number of months in the date range.
            # The .round() is important for ensuring we get an integer.
            num_months = end_date.difference(start_date, 'month').round()

            # Generate a list of starting dates for each month.
            # This uses a sequence and advances the start date by 'i' months.
            def get_month_start(i):
                return start_date.advance(i, 'month')
            
            month_starts = ee.List.sequence(0, num_months).map(get_month_start)

            # Define a function to map over the list of month start dates.
            def create_monthly_composite(date):
                # Cast the input to an ee.Date object.
                start_of_month = ee.Date(date)
                # The end date is exclusive, so we advance by 1 month.
                end_of_month = start_of_month.advance(1, 'month')

                # Filter the original collection to get images for the current month.
                monthly_subset = collection.filterDate(start_of_month, end_of_month)

                # Count the number of images in the monthly subset.
                image_count = monthly_subset.size()

                # Compute the mean. This is robust to outliers like clouds.
                monthly_mean = monthly_subset.mean()

                # Set essential properties on the resulting composite image.
                # The timestamp is crucial for time-series analysis and charting.
                # The image_count is useful metadata for quality assessment.
                return monthly_mean.set({
                    'system:time_start': start_of_month.millis(),
                    'month': start_of_month.get('month'),
                    'year': start_of_month.get('year'),
                    'Date_Filter': start_of_month.format('YYYY-MM-dd'),
                    'image_count': image_count
                }).reproject(target_proj)

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = Sentinel1Collection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_mean = final_collection
        else:
            pass

        return self._monthly_mean
    
    @property
    def monthly_sum_collection(self):
        """Creates a monthly sum composite from a Sentinel1Collection image collection.

        This function computes the sum for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with monthly sum composites.
        """
        if self._monthly_sum is None:
            collection = self.collection
            target_proj = collection.first().projection()
            # Get the start and end dates of the entire collection.
            date_range = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            original_start_date = ee.Date(date_range.get('min'))
            end_date = ee.Date(date_range.get('max'))

            start_year = original_start_date.get('year')
            start_month = original_start_date.get('month')
            start_date = ee.Date.fromYMD(start_year, start_month, 1)

            # Calculate the total number of months in the date range.
            # The .round() is important for ensuring we get an integer.
            num_months = end_date.difference(start_date, 'month').round()

            # Generate a list of starting dates for each month.
            # This uses a sequence and advances the start date by 'i' months.
            def get_month_start(i):
                return start_date.advance(i, 'month')
            
            month_starts = ee.List.sequence(0, num_months).map(get_month_start)

            # Define a function to map over the list of month start dates.
            def create_monthly_composite(date):
                # Cast the input to an ee.Date object.
                start_of_month = ee.Date(date)
                # The end date is exclusive, so we advance by 1 month.
                end_of_month = start_of_month.advance(1, 'month')

                # Filter the original collection to get images for the current month.
                monthly_subset = collection.filterDate(start_of_month, end_of_month)

                # Count the number of images in the monthly subset.
                image_count = monthly_subset.size()

                # Compute the sum. This is robust to outliers like clouds.
                monthly_sum = monthly_subset.sum()

                # Set essential properties on the resulting composite image.
                # The timestamp is crucial for time-series analysis and charting.
                # The image_count is useful metadata for quality assessment.
                return monthly_sum.set({
                    'system:time_start': start_of_month.millis(),
                    'month': start_of_month.get('month'),
                    'year': start_of_month.get('year'),
                    'Date_Filter': start_of_month.format('YYYY-MM-dd'),
                    'image_count': image_count
                }).reproject(target_proj)

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = Sentinel1Collection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_sum = final_collection
        else:
            pass

        return self._monthly_sum
    
    @property
    def monthly_max_collection(self):
        """Creates a monthly max composite from a Sentinel1Collection image collection.

        This function computes the max for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with monthly max composites.
        """
        if self._monthly_max is None:
            collection = self.collection
            target_proj = collection.first().projection()
            # Get the start and end dates of the entire collection.
            date_range = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            original_start_date = ee.Date(date_range.get('min'))
            end_date = ee.Date(date_range.get('max'))

            start_year = original_start_date.get('year')
            start_month = original_start_date.get('month')
            start_date = ee.Date.fromYMD(start_year, start_month, 1)

            # Calculate the total number of months in the date range.
            # The .round() is important for ensuring we get an integer.
            num_months = end_date.difference(start_date, 'month').round()

            # Generate a list of starting dates for each month.
            # This uses a sequence and advances the start date by 'i' months.
            def get_month_start(i):
                return start_date.advance(i, 'month')
            
            month_starts = ee.List.sequence(0, num_months).map(get_month_start)

            # Define a function to map over the list of month start dates.
            def create_monthly_composite(date):
                # Cast the input to an ee.Date object.
                start_of_month = ee.Date(date)
                # The end date is exclusive, so we advance by 1 month.
                end_of_month = start_of_month.advance(1, 'month')

                # Filter the original collection to get images for the current month.
                monthly_subset = collection.filterDate(start_of_month, end_of_month)

                # Count the number of images in the monthly subset.
                image_count = monthly_subset.size()

                # Compute the max. This is robust to outliers like clouds.
                monthly_max = monthly_subset.max()

                # Set essential properties on the resulting composite image.
                # The timestamp is crucial for time-series analysis and charting.
                # The image_count is useful metadata for quality assessment.
                return monthly_max.set({
                    'system:time_start': start_of_month.millis(),
                    'month': start_of_month.get('month'),
                    'year': start_of_month.get('year'),
                    'Date_Filter': start_of_month.format('YYYY-MM-dd'),
                    'image_count': image_count
                }).reproject(target_proj)

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = Sentinel1Collection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_max = final_collection
        else:
            pass

        return self._monthly_max
    
    @property
    def monthly_min_collection(self):
        """Creates a monthly min composite from a Sentinel1Collection image collection.

        This function computes the min for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            Sentinel1Collection: A new Sentinel1Collection object with monthly min composites.
        """
        if self._monthly_min is None:
            collection = self.collection
            target_proj = collection.first().projection()
            # Get the start and end dates of the entire collection.
            date_range = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            original_start_date = ee.Date(date_range.get('min'))
            end_date = ee.Date(date_range.get('max'))

            start_year = original_start_date.get('year')
            start_month = original_start_date.get('month')
            start_date = ee.Date.fromYMD(start_year, start_month, 1)

            # Calculate the total number of months in the date range.
            # The .round() is important for ensuring we get an integer.
            num_months = end_date.difference(start_date, 'month').round()

            # Generate a list of starting dates for each month.
            # This uses a sequence and advances the start date by 'i' months.
            def get_month_start(i):
                return start_date.advance(i, 'month')
            
            month_starts = ee.List.sequence(0, num_months).map(get_month_start)

            # Define a function to map over the list of month start dates.
            def create_monthly_composite(date):
                # Cast the input to an ee.Date object.
                start_of_month = ee.Date(date)
                # The end date is exclusive, so we advance by 1 month.
                end_of_month = start_of_month.advance(1, 'month')

                # Filter the original collection to get images for the current month.
                monthly_subset = collection.filterDate(start_of_month, end_of_month)

                # Count the number of images in the monthly subset.
                image_count = monthly_subset.size()

                # Compute the min. This is robust to outliers like clouds.
                monthly_min = monthly_subset.min()

                # Set essential properties on the resulting composite image.
                # The timestamp is crucial for time-series analysis and charting.
                # The image_count is useful metadata for quality assessment.
                return monthly_min.set({
                    'system:time_start': start_of_month.millis(),
                    'month': start_of_month.get('month'),
                    'year': start_of_month.get('year'),
                    'Date_Filter': start_of_month.format('YYYY-MM-dd'),
                    'image_count': image_count
                }).reproject(target_proj)

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = Sentinel1Collection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_min = final_collection
        else:
            pass

        return self._monthly_min
    
    def yearly_mean_collection(self, start_month=1, end_month=12):
        """
        Creates a yearly mean composite from the collection, with optional monthly filtering.

        This function computes the mean for each year within the collection's date range.
        You can specify a range of months (e.g., start_month=6, end_month=10 for June-October)
        to calculate the mean only using imagery from that specific season for each year.

        The resulting images have 'system:time_start', 'year', 'image_count', 'season_start',
        'season_end', and 'Date_Filter' properties. Years with no images (after filtering) are excluded.

        Args:
            start_month (int): The starting month (1-12) for the filter. Defaults to 1 (January).
            end_month (int): The ending month (1-12) for the filter. Defaults to 12 (December).

        Returns:
            Object: A new instance of the same class (e.g., Sentinel1Collection) containing the yearly mean composites.
        """
        if self._yearly_mean is None:

            date_range = self.collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date_full = ee.Date(date_range.get('min'))
            end_date_full = ee.Date(date_range.get('max'))
            
            start_year = start_date_full.get('year')
            end_year = end_date_full.get('year')

            if start_month != 1 or end_month != 12:
                processing_collection = self.collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
            else:
                processing_collection = self.collection

            # Capture projection from the first image to restore it after reduction
            target_proj = self.collection.first().projection()

            years = ee.List.sequence(start_year, end_year)

            def create_yearly_composite(year):
                year = ee.Number(year)
                # Define the full calendar year range
                start_of_year = ee.Date.fromYMD(year, 1, 1)
                end_of_year = start_of_year.advance(1, 'year')

                yearly_subset = processing_collection.filterDate(start_of_year, end_of_year)

                # Calculate stats
                image_count = yearly_subset.size()
                yearly_reduction = yearly_subset.mean()

                # Define the timestamp for the composite.
                # We use the start_month of that year to accurately reflect the data start time.
                composite_date = ee.Date.fromYMD(year, start_month, 1)

                return yearly_reduction.set({
                    'system:time_start': composite_date.millis(),
                    'year': year,
                    'month': start_month,  
                    'Date_Filter': composite_date.format('YYYY-MM-dd'), 
                    'image_count': image_count,
                    'season_start': start_month,
                    'season_end': end_month
                }).reproject(target_proj)

            # Map the function over the years list
            yearly_composites_list = years.map(create_yearly_composite)
            
            # Convert to Collection
            yearly_collection = ee.ImageCollection.fromImages(yearly_composites_list)

            # Filter out any composites that were created from zero images.
            final_collection = yearly_collection.filter(ee.Filter.gt('image_count', 0))

            self._yearly_mean = Sentinel1Collection(collection=final_collection)
        else: 
            pass
        return self._yearly_mean

    def yearly_median_collection(self, start_month=1, end_month=12):
        """
        Creates a yearly median composite from the collection, with optional monthly filtering.

        This function computes the median for each year within the collection's date range.
        You can specify a range of months (e.g., start_month=6, end_month=10 for June-October)
        to calculate the median only using imagery from that specific season for each year.
        The resulting images have 'system:time_start', 'year', 'image_count', 'season_start',
        'season_end', and 'Date_Filter' properties. Years with no images (after filtering) are excluded.

        Args:
            start_month (int): The starting month (1-12) for the filter. Defaults to 1 (January).
            end_month (int): The ending month (1-12) for the filter. Defaults to 12 (December).

        Returns:
            Object: A new instance of the same class (e.g., Sentinel1Collection) containing the yearly median composites.
        """
        if self._yearly_median is None:

            date_range = self.collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date_full = ee.Date(date_range.get('min'))
            end_date_full = ee.Date(date_range.get('max'))
            
            start_year = start_date_full.get('year')
            end_year = end_date_full.get('year')

            if start_month != 1 or end_month != 12:
                processing_collection = self.collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
            else:
                processing_collection = self.collection

            # Capture projection from the first image to restore it after reduction
            target_proj = self.collection.first().projection()

            years = ee.List.sequence(start_year, end_year)

            def create_yearly_composite(year):
                year = ee.Number(year)
                # Define the full calendar year range
                start_of_year = ee.Date.fromYMD(year, 1, 1)
                end_of_year = start_of_year.advance(1, 'year')

                # Filter to the specific year using the PRE-FILTERED seasonal collection
                yearly_subset = processing_collection.filterDate(start_of_year, end_of_year)

                # Calculate stats
                image_count = yearly_subset.size()
                yearly_reduction = yearly_subset.median()

                # Define the timestamp for the composite.
                # We use the start_month of that year to accurately reflect the data start time.
                composite_date = ee.Date.fromYMD(year, start_month, 1)

                return yearly_reduction.set({
                    'system:time_start': composite_date.millis(),
                    'year': year,
                    'month': start_month,  
                    'Date_Filter': composite_date.format('YYYY-MM-dd'), 
                    'image_count': image_count,
                    'season_start': start_month,
                    'season_end': end_month
                }).reproject(target_proj)

            # Map the function over the years list
            yearly_composites_list = years.map(create_yearly_composite)
            
            # Convert to Collection
            yearly_collection = ee.ImageCollection.fromImages(yearly_composites_list)

            # Filter out any composites that were created from zero images.
            final_collection = yearly_collection.filter(ee.Filter.gt('image_count', 0))

            self._yearly_median = Sentinel1Collection(collection=final_collection)
        else: 
            pass
        return self._yearly_median

    def yearly_max_collection(self, start_month=1, end_month=12):
        """
        Creates a yearly max composite from the collection, with optional monthly filtering.

        This function computes the max for each year within the collection's date range.
        You can specify a range of months (e.g., start_month=6, end_month=10 for June-October)
        to calculate the max only using imagery from that specific season for each year.
        The resulting images have 'system:time_start', 'year', 'image_count', 'season_start',
        'season_end', and 'Date_Filter' properties. Years with no images (after filtering) are excluded.

        Args:
            start_month (int): The starting month (1-12) for the filter. Defaults to 1 (January).
            end_month (int): The ending month (1-12) for the filter. Defaults to 12 (December).

        Returns:
            Object: A new instance of the same class (e.g., Sentinel1Collection) containing the yearly max composites.
        """
        if self._yearly_max is None:

            date_range = self.collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date_full = ee.Date(date_range.get('min'))
            end_date_full = ee.Date(date_range.get('max'))
            
            start_year = start_date_full.get('year')
            end_year = end_date_full.get('year')

            if start_month != 1 or end_month != 12:
                processing_collection = self.collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
            else:
                processing_collection = self.collection

            # Capture projection from the first image to restore it after reduction
            target_proj = self.collection.first().projection()

            years = ee.List.sequence(start_year, end_year)

            def create_yearly_composite(year):
                year = ee.Number(year)
                # Define the full calendar year range
                start_of_year = ee.Date.fromYMD(year, 1, 1)
                end_of_year = start_of_year.advance(1, 'year')

                # Filter to the specific year using the PRE-FILTERED seasonal collection
                yearly_subset = processing_collection.filterDate(start_of_year, end_of_year)

                # Calculate stats
                image_count = yearly_subset.size()
                yearly_reduction = yearly_subset.max()

                # Define the timestamp for the composite.
                # We use the start_month of that year to accurately reflect the data start time.
                composite_date = ee.Date.fromYMD(year, start_month, 1)

                return yearly_reduction.set({
                    'system:time_start': composite_date.millis(),
                    'year': year,
                    'month': start_month,  
                    'Date_Filter': composite_date.format('YYYY-MM-dd'), 
                    'image_count': image_count,
                    'season_start': start_month,
                    'season_end': end_month
                }).reproject(target_proj)

            # Map the function over the years list
            yearly_composites_list = years.map(create_yearly_composite)
            
            # Convert to Collection
            yearly_collection = ee.ImageCollection.fromImages(yearly_composites_list)

            # Filter out any composites that were created from zero images.
            final_collection = yearly_collection.filter(ee.Filter.gt('image_count', 0))

            self._yearly_max = Sentinel1Collection(collection=final_collection)
        else: 
            pass
        return self._yearly_max
    
    def yearly_min_collection(self, start_month=1, end_month=12):
        """
        Creates a yearly min composite from the collection, with optional monthly filtering.

        This function computes the min for each year within the collection's date range.
        You can specify a range of months (e.g., start_month=6, end_month=10 for June-October)
        to calculate the min only using imagery from that specific season for each year.
        The resulting images have 'system:time_start', 'year', 'image_count', 'season_start',
        'season_end', and 'Date_Filter' properties. Years with no images (after filtering) are excluded.

        Args:
            start_month (int): The starting month (1-12) for the filter. Defaults to 1 (January).
            end_month (int): The ending month (1-12) for the filter. Defaults to 12 (December).

        Returns:
            Object: A new instance of the same class (e.g., Sentinel1Collection) containing the yearly min composites.
        """
        if self._yearly_min is None:

            date_range = self.collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date_full = ee.Date(date_range.get('min'))
            end_date_full = ee.Date(date_range.get('max'))
            
            start_year = start_date_full.get('year')
            end_year = end_date_full.get('year')

            if start_month != 1 or end_month != 12:
                processing_collection = self.collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
            else:
                processing_collection = self.collection

            # Capture projection from the first image to restore it after reduction
            target_proj = self.collection.first().projection()

            years = ee.List.sequence(start_year, end_year)

            def create_yearly_composite(year):
                year = ee.Number(year)
                # Define the full calendar year range
                start_of_year = ee.Date.fromYMD(year, 1, 1)
                end_of_year = start_of_year.advance(1, 'year')

                # Filter to the specific year using the PRE-FILTERED seasonal collection
                yearly_subset = processing_collection.filterDate(start_of_year, end_of_year)

                # Calculate stats
                image_count = yearly_subset.size()
                yearly_reduction = yearly_subset.min()

                # Define the timestamp for the composite.
                # We use the start_month of that year to accurately reflect the data start time.
                composite_date = ee.Date.fromYMD(year, start_month, 1)

                return yearly_reduction.set({
                    'system:time_start': composite_date.millis(),
                    'year': year,
                    'month': start_month,  
                    'Date_Filter': composite_date.format('YYYY-MM-dd'), 
                    'image_count': image_count,
                    'season_start': start_month,
                    'season_end': end_month
                }).reproject(target_proj)

            # Map the function over the years list
            yearly_composites_list = years.map(create_yearly_composite)
            
            # Convert to Collection
            yearly_collection = ee.ImageCollection.fromImages(yearly_composites_list)

            # Filter out any composites that were created from zero images.
            final_collection = yearly_collection.filter(ee.Filter.gt('image_count', 0))

            self._yearly_min = Sentinel1Collection(collection=final_collection)
        else: 
            pass
        return self._yearly_min

    def yearly_sum_collection(self, start_month=1, end_month=12):
        """
        Creates a yearly sum composite from the collection, with optional monthly filtering.

        This function computes the sum for each year within the collection's date range.
        You can specify a range of months (e.g., start_month=6, end_month=10 for June-October)
        to calculate the sum only using imagery from that specific season for each year.
        The resulting images have 'system:time_start', 'year', 'image_count', 'season_start',
        'season_end', and 'Date_Filter' properties. Years with no images (after filtering) are excluded.

        Args:
            start_month (int): The starting month (1-12) for the filter. Defaults to 1 (January).
            end_month (int): The ending month (1-12) for the filter. Defaults to 12 (December).

        Returns:
            Object: A new instance of the same class (e.g., Sentinel1Collection) containing the yearly sum composites.
        """
        if self._yearly_sum is None:

            date_range = self.collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"])
            start_date_full = ee.Date(date_range.get('min'))
            end_date_full = ee.Date(date_range.get('max'))
            
            start_year = start_date_full.get('year')
            end_year = end_date_full.get('year')

            if start_month != 1 or end_month != 12:
                processing_collection = self.collection.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
            else:
                processing_collection = self.collection

            # Capture projection from the first image to restore it after reduction
            target_proj = self.collection.first().projection()

            years = ee.List.sequence(start_year, end_year)

            def create_yearly_composite(year):
                year = ee.Number(year)
                # Define the full calendar year range
                start_of_year = ee.Date.fromYMD(year, 1, 1)
                end_of_year = start_of_year.advance(1, 'year')

                # Filter to the specific year using the PRE-FILTERED seasonal collection
                yearly_subset = processing_collection.filterDate(start_of_year, end_of_year)

                # Calculate stats
                image_count = yearly_subset.size()
                yearly_reduction = yearly_subset.sum()

                # Define the timestamp for the composite.
                # We use the start_month of that year to accurately reflect the data start time.
                composite_date = ee.Date.fromYMD(year, start_month, 1)

                return yearly_reduction.set({
                    'system:time_start': composite_date.millis(),
                    'year': year,
                    'month': start_month,  
                    'Date_Filter': composite_date.format('YYYY-MM-dd'), 
                    'image_count': image_count,
                    'season_start': start_month,
                    'season_end': end_month
                }).reproject(target_proj)

            # Map the function over the years list
            yearly_composites_list = years.map(create_yearly_composite)
            
            # Convert to Collection
            yearly_collection = ee.ImageCollection.fromImages(yearly_composites_list)

            # Filter out any composites that were created from zero images.
            final_collection = yearly_collection.filter(ee.Filter.gt('image_count', 0))

            self._yearly_sum = Sentinel1Collection(collection=final_collection)
        else: 
            pass
        return self._yearly_sum

    def anomaly(self, geometry, band_name=None, anomaly_band_name=None, replace=True, scale=10):
        """
        Calculates the anomaly of each image in a collection compared to the mean of each image.

        This function computes the anomaly for each band in the input image by
        subtracting the mean value of that band from a provided ImageCollection.
        The anomaly is a measure of how much the pixel values deviate from the
        average conditions represented by the collection.

        Args:
            geometry (ee.Geometry): The geometry for image reduction to define the mean value to be used for anomaly calculation.
            band_name (str, optional): A string representing the band name to be used for the output anomaly image. If not provided, the band name of the first band of the input image will be used.
            anomaly_band_name (str, optional): A string representing the band name to be used for the output anomaly image. If not provided, the band name of the first band of the input image will be used.
            replace (bool, optional): A boolean indicating whether to replace the original band with the anomaly band. If True, the output image will only contain the anomaly band. If False, the output image will retain all original bands and add the anomaly band. Default is True.
            scale (int, optional): The scale (in meters) to use for the image reduction. Default is 10 meters.

        Returns:
            Sentinel1Collection: A Sentinel1Collection where each image represents the anomaly (deviation from
                        the mean) for the chosen band. The output images retain the same band name.
        """
        if self.collection.size().eq(0).getInfo():
            raise ValueError("The collection is empty.")
        if band_name is None:
            first_image = self.collection.first()
            band_names = first_image.bandNames()
            if band_names.size().getInfo() == 0:
                raise ValueError("No bands available in the collection.")
            elif band_names.size().getInfo() > 1:
                band_name = band_names.get(0).getInfo()
                print("Multiple bands available, will be using the first band in the collection for anomaly calculation. Please specify a band name if you wish to use a different band.")
            else:
                band_name = band_names.get(0).getInfo()

        col = self.collection.map(lambda image: Sentinel1Collection.anomaly_fn(image, geometry=geometry, band_name=band_name, anomaly_band_name=anomaly_band_name, replace=replace, scale=scale))
        return Sentinel1Collection(collection=col)

    def binary_mask(self, threshold=None, band_name=None):
        """
        Creates a binary mask (value of 1 for pixels above set threshold and value of 0 for all other pixels) of the Sentinel1Collection image collection based on a specified band.
        If a singleband image is provided, the band name is automatically determined.
        If multiple bands are available, the user must specify the band name to use for masking.

        Args:
            band_name (str, optional): The name of the band to use for masking. Defaults to None.

        Returns:
            Sentinel1Collection: Sentinel1Collection singleband image collection with binary masks applied.
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
            lambda image: image.select(band_name).gte(threshold).rename(band_name).copyProperties(image).set('system:time_start', image.get('system:time_start'))
        )
        return Sentinel1Collection(collection=col)
    
    def mann_kendall_trend(self, target_band=None, join_method='system:time_start', geometry=None):
        """
        Calculates the Mann-Kendall S-value, Variance, Z-Score, and Confidence Level for each pixel in the image collection, in addition to calculating
        the Sen's slope for each pixel in the image collection. The output is an image with the following bands: 's_statistic', 'variance', 'z_score', 'confidence', and 'slope'.

        This function can be used to identify trends in the image collection over time, such as increasing or decreasing values in the target band, and can be used to assess the significance of these trends.
        Note that this function is computationally intensive and may take a long time to run for large image collections or high-resolution images.

        The 's_statistic' band represents the Mann-Kendall S-value, which is a measure of the strength and direction of the trend.
        The 'variance' band represents the variance of the S-value, which is a measure of the variability of the S-value.
        The 'z_score' band represents the Z-Score, which is a measure of the significance of the trend.
        The 'confidence' band represents the confidence level of the trend based on the z_score, which is a probabilistic measure of the confidence in the trend (percentage).
        The 'slope' band represents the Sen's slope, which is a measure of the rate of change in the target band over time. This value can be small as multispectral indices commonly range from -1 to 1, so a slope may have values of <0.2 for most cases.

        Be sure to select the correct band for the `target_band` parameter, as this will be used to calculate the trend statistics.
        You may optionally provide an ee.Geometry object for the `geometry` parameter to limit the area over which the trend statistics are calculated.
        The `geometry` parameter is optional and defaults to None, which means that the trend statistics will be calculated over the entire footprint of the image collection.

        Args: 
            image_collection (Sentinel1Collection or ee.ImageCollection): The input image collection for which the Mann-Kendall and Sen's slope trend statistics will be calculated.
            target_band (str): The band name to be used for the output anomaly image. e.g. 'ndvi'
            join_method (str, optional): The method used to join images in the collection. Options are 'system:time_start' or 'Date_Filter'. Default is 'system:time_start'.
            geometry (ee.Geometry, optional): An ee.Geometry object to limit the area over which the trend statistics are calculated and mask the output image. Default is None.

        Returns:
            ee.Image: An image with the following bands: 's_statistic', 'variance', 'z_score', 'confidence', and 'slope'.
        """
        ########## PART 1 - S-VALUE CALCULATION ##########
        ##### https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm #####
        image_collection = self
        if isinstance(image_collection, Sentinel1Collection):
            image_collection = image_collection.collection
        elif isinstance(image_collection, ee.ImageCollection):
            pass
        else:
            raise ValueError(f'The chosen `image_collection`: {image_collection} is not a valid Sentinel1Collection or ee.ImageCollection object.')
        
        if target_band is None:
            raise ValueError('The `target_band` parameter must be specified.')
        if not isinstance(target_band, str):
            raise ValueError(f'The chosen `target_band`: {target_band} is not a valid string.')

        if geometry is not None and not isinstance(geometry, ee.Geometry):
            raise ValueError(f'The chosen `geometry`: {geometry} is not a valid ee.Geometry object.')
        
        native_projection = image_collection.first().select(target_band).projection()

        # define the join, which will join all images newer than the current image
        # use system:time_start if the image does not have a Date_Filter property
        if join_method == 'system:time_start':
            # get all images where the leftField value is less than (before) the rightField value
            time_filter = ee.Filter.lessThan(leftField='system:time_start', 
                                            rightField='system:time_start')
        elif join_method == 'Date_Filter':
            # get all images where the leftField value is less than (before) the rightField value
            time_filter = ee.Filter.lessThan(leftField='Date_Filter', 
                                            rightField='Date_Filter')
        else:
            raise ValueError(f'The chosen `join_method`: {join_method} does not match the options of "system:time_start" or "Date_Filter".')

        # for any matches during a join, set image as a property key called 'future_image'
        join = ee.Join.saveAll(matchesKey='future_image')

        # apply the join on the input collection 
        # joining all images newer than the current image with the current image
        joined_collection = ee.ImageCollection(join.apply(primary=image_collection, 
                                                secondary=image_collection, condition=time_filter))
        
        # defining a collection to calculate the partial S value for each match in the join
        # e.g. t4-t1, t3-t1, t2-1 if there are 4 images
        def calculate_partial_s(current_image):
            # select the target band for arithmetic
            current_val = current_image.select(target_band)
            # get the joined images from the current image properties and cast the joined images as a list
            future_image_list = ee.List(current_image.get('future_image'))
            # convert the joined list to an image collection
            future_image_collection = ee.ImageCollection(future_image_list)

            # define a function that will calculate the difference between the joined images and the current image, 
            # then calculate the partial S sign based on the value of the difference calculation
            def get_sign(future_image):
                # select the target band for arithmetic from the future image
                future_val = future_image.select(target_band)
                # calculate the difference, i.e. t2-t1
                difference = future_val.subtract(current_val)
                # determine the sign of the difference value (1 if diff > 0, 0 if 0, and -1 if diff < 0)
                # use .unmask(0) to set any masked pixels as 0 to avoid 
                
                sign = difference.signum().unmask(0)
                
                return sign
            
            # map the get_sign() function along the future image col 
            # then sum the values for each pixel to get the partial S value
            return future_image_collection.map(get_sign).sum()

        # calculate the partial s value for each image in the joined/input image collection
        partial_s_col = joined_collection.map(calculate_partial_s)

        # convert the image collection to an image of s_statistic values per pixel
        # where the s_statistic is the sum of partial s values
        # renaming the band as 's_statistic' for later usage
        final_s_image = partial_s_col.sum().rename('s_statistic').setDefaultProjection(native_projection)


        ########## PART 2 - VARIANCE and Z-SCORE ##########
        # to calculate variance we need to know how many pixels were involved in the partial_s calculations per pixel
        # we do this by using count() and turn the value to a float for later arithmetic
        n = image_collection.select(target_band).count().toFloat()

        ##### VARIANCE CALCULATION #####
        # as we are using floating point values with high precision, it is HIGHLY 
        # unlikely that there will be multiple pixel values with the same value.
        # Thus, we opt to use the simplified variance calculation approach as the
        # impacts to the output value are negligible and the processing benefits are HUGE
        # variance = (n * (n - 1) * (2n + 5)) / 18
        var_s = n.multiply(n.subtract(1))\
                .multiply(n.multiply(2).add(5))\
                .divide(18).rename('variance')
        
        z_score = ee.Image().expression(
                    """
                    (s > 0) ? (s - 1) / sqrt(var) :
                    (s < 0) ? (s + 1) / sqrt(var) :
                    0
                    """,
                    {'s': final_s_image, 'var': var_s}
                ).rename('z_score')
        
        confidence = z_score.abs().divide(ee.Number(2).sqrt()).erf().rename('confidence')

        stat_bands = ee.Image([var_s, z_score, confidence])

        mk_stats_image = final_s_image.addBands(stat_bands)

        ########## PART 3 - Sen's Slope ##########
        def add_year_band(image):
            if join_method == 'Date_Filter':
                # Get the string 'YYYY-MM-DD'
                date_string = image.get('Date_Filter')
                # Parse it into an ee.Date object (handles the conversion to time math)
                date = ee.Date.parse('YYYY-MM-dd', date_string)
            else:
                # Standard way: assumes system:time_start exists
                date = image.date()
            years = date.difference(ee.Date('1970-01-01'), 'year')
            return image.addBands(ee.Image(years).float().rename('year'))
        
        slope_input = image_collection.map(add_year_band).select(['year', target_band])

        sens_slope = slope_input.reduce(ee.Reducer.sensSlope())

        slope_band = sens_slope.select('slope')

        # add a mask to the final image to remove pixels with less than min_observations
        # mainly an effort to mask pixels outside of the boundary of the input image collection
        min_observations = 1
        valid_mask = n.gte(min_observations)

        final_image = mk_stats_image.addBands(slope_band).updateMask(valid_mask)

        if geometry is not None:
            mask = ee.Image(1).clip(geometry)
            final_image = final_image.updateMask(mask)

        return final_image.setDefaultProjection(native_projection)

    def sens_slope_trend(self, target_band=None, join_method='system:time_start', geometry=None):
            """
            Calculates Sen's Slope (trend magnitude) for the collection.
            This is a lighter-weight alternative to the full `mann_kendall_trend` function if only
            the direction and magnitude of the trend are needed.

            Be sure to select the correct band for the `target_band` parameter, as this will be used to calculate the trend statistics.
            You may optionally provide an ee.Geometry object for the `geometry` parameter to limit the area over which the trend statistics are calculated.
            The `geometry` parameter is optional and defaults to None, which means that the trend statistics will be calculated over the entire footprint of the image collection.

            Args:
                target_band (str): The name of the band to analyze. Defaults to 'ndvi'.
                join_method (str): Property to use for time sorting ('system:time_start' or 'Date_Filter').
                geometry (ee.Geometry, optional): Geometry to mask the final output.

            Returns:
                ee.Image: An image containing the 'slope' band.
            """
            image_collection = self
            if isinstance(image_collection, Sentinel1Collection):
                image_collection = image_collection.collection
            elif isinstance(image_collection, ee.ImageCollection):
                pass
            else:
                raise ValueError(f'The chosen `image_collection`: {image_collection} is not a valid Sentinel1Collection or ee.ImageCollection object.')
            
            if target_band is None:
                raise ValueError('The `target_band` parameter must be specified.')
            if not isinstance(target_band, str):
                raise ValueError(f'The chosen `target_band`: {target_band} is not a valid string.')

            if geometry is not None and not isinstance(geometry, ee.Geometry):
                raise ValueError(f'The chosen `geometry`: {geometry} is not a valid ee.Geometry object.')

            native_projection = image_collection.first().select(target_band).projection()

            # Add Year Band (Time X-Axis)
            def add_year_band(image):
                # Handle user-defined date strings vs system time
                if join_method == 'Date_Filter':
                    date_string = image.get('Date_Filter')
                    date = ee.Date.parse('YYYY-MM-dd', date_string)
                else:
                    date = image.date()
                    
                # Convert to fractional years relative to epoch
                years = date.difference(ee.Date('1970-01-01'), 'year')
                return image.addBands(ee.Image(years).float().rename('year'))

            # Prepare Collection: Select ONLY [Year, Target]
            # sensSlope expects Band 0 = Independent (X), Band 1 = Dependent (Y)
            slope_input = self.collection.map(add_year_band).select(['year', target_band])

            # Run the Native Reducer
            sens_result = slope_input.reduce(ee.Reducer.sensSlope())
            
            # Extract and Mask
            slope_band = sens_result.select('slope')

            if geometry is not None:
                mask = ee.Image(1).clip(geometry)
                slope_band = slope_band.updateMask(mask)

            return slope_band.setDefaultProjection(native_projection)
    

    def mask_to_polygon(self, polygon):
        """
        Function to mask Sentinel1Collection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args: (image collection)
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            Sentinel1Collection: masked Sentinel1Collection image collection

        """
        # Convert the polygon to a mask
        mask = ee.Image.constant(1).clip(polygon)

        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(mask)\
                                .copyProperties(img).set('system:time_start', img.get('system:time_start')))

        # Return the updated object
        return Sentinel1Collection(collection=masked_collection)

    def mask_out_polygon(self, polygon):
        """
        Function to mask Sentinel1Collection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args: (image collection)
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            Sentinel1Collection: masked Sentinel1Collection image collection

        """
        # Convert the polygon to a mask
        full_mask = ee.Image.constant(1)

        # Use paint to set pixels inside polygon as 0
        area = full_mask.paint(polygon, 0)

        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(area)\
                            .copyProperties(img).set('system:time_start', img.get('system:time_start')))

        # Return the updated object
        return Sentinel1Collection(collection=masked_collection)

    def image_grab(self, img_selector):
        """
        Function to select ("grab") an image by index from the collection. Easy way to get latest image or browse imagery one-by-one.

        Args:
            img_selector (int): index of image in the collection for which user seeks to select/"grab".

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
            img_col (ee.ImageCollection): ee.ImageCollection with same dates as another Sentinel1Collection image collection object.
            img_selector (int): index of image in list of dates for which user seeks to "select".

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
            img_date (str): date of image to select from collection, in format of 'YYYY-MM-DD'

        Returns:
            ee.Image: ee.Image of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq("Date_Filter", img_date))
        return new_col.first()

    def collectionStitch(self, img_col2):
        """
        Function to mosaic two Sentinel1Collection objects which share image dates.
        Mosaics are only formed for dates where both image collections have images.
        Image properties are copied from the primary collection. Server-side friendly.

        Args:
            img_col2: secondary Sentinel1Collection image collection to be mosaiced with the primary image collection

        Returns:
            Sentinel1Collection: Sentinel1Collection image collection
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

        # Return a Sentinel1Collection instance
        return Sentinel1Collection(collection=new_col)
    
    def CollectionStitch(self, img_col2):
        warnings.warn(
            "The 'CollectionStitch' method is deprecated. Please use 'collectionStitch' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.collectionStitch(img_col2)

    @property
    def mosaicByDateDepr(self):
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
                date_filter = ee.Filter.eq("Date_Filter", date)
                date_collection = input_collection.filter(date_filter)
                # Convert the collection to a list
                image_list = date_collection.toList(date_collection.size())

                # Get the image at the specified index
                first_image = ee.Image(image_list.get(0))
                # Create mosaic
                mosaic = date_collection.mosaic().set("Date_Filter", date)

                props_of_interest = [
                    "platform_number",
                    "instrument",
                    "instrumentMode",
                    "orbitNumber_start",
                    "orbitNumber_stop",
                    "orbitProperties_pass",
                    "resolution_meters",
                    "transmitterReceiverPolarisation",
                    "system:time_start",
                    "crs",
                ]

                mosaic = mosaic.setDefaultProjection(
                    first_image.projection()
                ).copyProperties(first_image, props_of_interest)

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = Sentinel1Collection(collection=new_col)
            self._MosaicByDate = col

        # Convert the list of mosaics to an ImageCollection
        return self._MosaicByDate
    
    @property
    def mosaicByDate(self):
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
            distinct_dates = self.collection.distinct("Date_Filter")

            # Define a join to link images by Date_Filter
            filter_date = ee.Filter.equals(leftField="Date_Filter", rightField="Date_Filter")
            join = ee.Join.saveAll(matchesKey="date_matches")

            # Apply the join
            # Primary: Distinct dates collection
            # Secondary: The full original collection
            joined_col = ee.ImageCollection(join.apply(distinct_dates, self.collection, filter_date))

            # Define the mosaicking function 
            def _mosaic_day(img):
                # Recover the list of images for this day
                daily_list = ee.List(img.get("date_matches"))
                daily_col = ee.ImageCollection.fromImages(daily_list)
                
                # Create the mosaic
                mosaic = daily_col.mosaic().setDefaultProjection(img.projection())

                # Properties to preserve from the representative image
                props_of_interest = [
                    "platform_number",
                    "instrument",
                    "instrumentMode",
                    "orbitNumber_start",
                    "orbitNumber_stop",
                    "orbitProperties_pass",
                    "resolution_meters",
                    "transmitterReceiverPolarisation",
                    "system:time_start",
                    "crs"
                ]
                
                # Return mosaic with properties set
                return mosaic.copyProperties(img, props_of_interest)

            # 5. Map the function and wrap the result
            mosaiced_col = joined_col.map(_mosaic_day)
            self._MosaicByDate = Sentinel1Collection(collection=mosaiced_col)

        # Convert the list of mosaics to an ImageCollection
        return self._MosaicByDate

    @property
    def MosaicByDate(self):
        warnings.warn(
            "The 'MosaicByDate' property is deprecated. Please use 'mosaicByDate' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.mosaicByDate

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
                return Sentinel1Collection.ee_to_df(transect)
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
        dist_interval=10,
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
            dist_interval (float): The distance interval used for splitting the LineString. If specified, the n_segments parameter will be ignored. Defaults to 10.
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
                transect_data = Sentinel1Collection.extract_transect(
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
                transect_data = Sentinel1Collection.extract_transect(
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
        dist_interval= 10,
        n_segments=None,
        scale=10,
        processing_mode='aggregated',
        save_folder_path=None,
        sampling_method='line',
        point_buffer_radius=5
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
                Defaults to 10. Recommended to increase this value when using the 
                'line' processing method, or else you may get blank rows.
            n_segments (int, optional): The number of equal-length segments to split
                each transect line into for sampling. This parameter overrides `dist_interval`. 
                Defaults to None.
            scale (int, optional): The nominal scale in meters for the reduction,
                which should typically match the pixel resolution of the imagery.
                Defaults to 10.
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
                when `sampling_method` is 'buffered_point'. Defaults to 5.

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
            df = Sentinel1Collection.ee_to_df(results_fc, remove_geom=True)
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
                    transects_df = Sentinel1Collection.transect(
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
        scale=10,
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
            scale (int, optional): The scale in meters for the reduction. Defaults to 10.
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

        df = Sentinel1Collection.ee_to_df(stats_fc, remove_geom=True)

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
        scale=10,
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
            scale (int, optional): Pixel scale in meters for the reduction. Defaults to 10.
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
        # Create a local reference to the collection object to allow for modifications (like band selection) without altering the original instance
        img_collection_obj = self

        # If a specific band is requested, select only that band
        if band:
            img_collection_obj = Sentinel1Collection(collection=img_collection_obj.collection.select(band))
        else: 
            # If no band is specified, default to using the first band of the first image in the collection
            first_image = img_collection_obj.image_grab(0)
            first_band = first_image.bandNames().get(0)
            img_collection_obj = Sentinel1Collection(collection=img_collection_obj.collection.select([first_band]))
        
        # If a list of dates is provided, filter the collection to include only images matching those dates
        if dates:
            img_collection_obj = Sentinel1Collection(
                collection=self.collection.filter(ee.Filter.inList('Date_Filter', dates))
            )

        # Initialize variables to hold the standardized feature collection and coordinates
        features = None
        validated_coordinates = [] 
        
        # Define a helper function to ensure every feature has a standardized 'geo_name' property
        # This handles features that might have different existing name properties or none at all
        def set_standard_name(feature):
            has_geo_name = feature.get('geo_name')
            has_name = feature.get('name')
            has_index = feature.get('system:index')
            new_name = ee.Algorithms.If(
                has_geo_name, has_geo_name,
                ee.Algorithms.If(has_name, has_name,
                ee.Algorithms.If(has_index, has_index, 'unnamed_geometry')))
            return feature.set({'geo_name': new_name})

        # Handle input: FeatureCollection or single Feature
        if isinstance(geometries, (ee.FeatureCollection, ee.Feature)):
            features = ee.FeatureCollection(geometries)
            if geometry_names:
                 print("Warning: 'geometry_names' are ignored when the input is an ee.Feature or ee.FeatureCollection.")

        # Handle input: Single ee.Geometry
        elif isinstance(geometries, ee.Geometry):
             name = geometry_names[0] if (geometry_names and geometry_names[0]) else 'unnamed_geometry'
             features = ee.FeatureCollection([ee.Feature(geometries).set('geo_name', name)])

        # Handle input: List (could be coordinates or ee.Geometry objects)
        elif isinstance(geometries, list):
            if not geometries: # Handle empty list case
                raise ValueError("'geometries' list cannot be empty.")
            
            # Case: List of tuples (coordinates)
            if all(isinstance(i, tuple) for i in geometries):
                validated_coordinates = geometries
                # Generate default names if none provided
                if geometry_names is None:
                    geometry_names = [f"Location_{i+1}" for i in range(len(validated_coordinates))]
                elif len(geometry_names) != len(validated_coordinates):
                     raise ValueError("geometry_names must have the same length as the coordinates list.")
                # Create features with buffers around the coordinates
                points = [
                    ee.Feature(ee.Geometry.Point(coord).buffer(buffer_size), {'geo_name': str(name)})
                    for coord, name in zip(validated_coordinates, geometry_names)
                ]
                features = ee.FeatureCollection(points)
            
            # Case: List of ee.Geometry objects
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

        # Handle input: Single tuple (coordinate)
        elif isinstance(geometries, tuple) and len(geometries) == 2:
            name = geometry_names[0] if geometry_names else 'Location_1'
            features = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point(geometries).buffer(buffer_size), {'geo_name': name})
            ])
        else:
            raise TypeError("Unsupported type for 'geometries'.")
        
        # Apply the naming standardization to the created FeatureCollection
        features = features.map(set_standard_name)

        # Dynamically retrieve the Earth Engine reducer based on the string name provided
        try:
            reducer = getattr(ee.Reducer, reducer_type)()
        except AttributeError:
            raise ValueError(f"Unknown reducer_type: '{reducer_type}'.")

        # Define the function to map over the image collection
        def calculate_stats_for_image(image):
            image_date = image.get('Date_Filter')
            # Calculate statistics for all geometries in 'features' for this specific image
            stats_fc = image.reduceRegions(
                collection=features, reducer=reducer, scale=scale, tileScale=tileScale
            )

            # Helper to ensure the result has the reducer property, even if masked
            # If the property is missing (e.g., all pixels masked), set it to a sentinel value (-9999)
            def guarantee_reducer_property(f):
                has_property = f.propertyNames().contains(reducer_type)
                return ee.Algorithms.If(has_property, f, f.set(reducer_type, -9999))
            
            # Apply the guarantee check
            fixed_stats_fc = stats_fc.map(guarantee_reducer_property)

            # Attach the image date to every feature in the result so we know which image it came from
            return fixed_stats_fc.map(lambda f: f.set('image_date', image_date))

        # Map the calculation over the image collection and flatten the resulting FeatureCollections into one
        results_fc = ee.FeatureCollection(img_collection_obj.collection.map(calculate_stats_for_image)).flatten()
        
        # Convert the Earth Engine FeatureCollection to a pandas DataFrame (client-side operation)
        df = Sentinel1Collection.ee_to_df(results_fc, remove_geom=True)

        # Check for empty results or missing columns
        if df.empty: 
            raise ValueError("No results found for the given parameters. Check if the geometries intersect with the images, if the dates filter is too restrictive, or if the provided bands are empty.")
        if reducer_type not in df.columns:
            print(f"Warning: Reducer '{reducer_type}' not found in results.")

        # Filter out the sentinel values (-9999) which indicate failed reductions/masked pixels
        initial_rows = len(df)
        df.dropna(subset=[reducer_type], inplace=True)
        df = df[df[reducer_type] != -9999]
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"Warning: Discarded {dropped_rows} results due to failed reductions (e.g., no valid pixels in geometry).")

        # Pivot the DataFrame so that each row represents a date and each column represents a geometry location
        pivot_df = df.pivot(index='image_date', columns='geo_name', values=reducer_type)
        # Rename the column headers (geometry names) to include the reducer type 
        pivot_df.columns = [f"{col}_{reducer_type}" for col in pivot_df.columns]
        # Rename the index axis to 'Date' so it is correctly labeled when moved to a column later
        pivot_df.index.name = 'Date'
        # Remove the name of the columns axis (which defaults to 'geo_name') so it doesn't appear as a confusing label in the final output
        pivot_df.columns.name = None
        # Reset the index to move the 'Date' index into a regular column and create a standard numerical index (0, 1, 2...)
        pivot_df = pivot_df.reset_index(drop=False)

        # If a file path is provided, save the resulting DataFrame to CSV
        if file_path:
            # Check if file_path ends with .csv and remove it if so for consistency
            if file_path.endswith('.csv'):
                file_path = file_path[:-4]
            pivot_df.to_csv(f"{file_path}.csv")
            print(f"Zonal stats saved to {file_path}.csv")
            return
        return pivot_df

    def export_to_asset_collection(
        self,
        asset_collection_path,
        region,
        scale,
        dates=None,
        filename_prefix="",
        crs=None,
        max_pixels=int(1e13),
        description_prefix="export"
    ):
        """
        Exports an image collection to a Google Earth Engine asset collection. The asset collection will be created if it does not already exist, 
        and each image exported will be named according to the provided filename prefix and date.

        Args:
            asset_collection_path (str): The path to the asset collection.
            region (ee.Geometry): The region to export.
            scale (int): The scale of the export.
            dates (list, optional): The dates to export. Defaults to None.
            filename_prefix (str, optional): The filename prefix. Defaults to "", i.e. blank.
            crs (str, optional): The coordinate reference system. Defaults to None, which will use the image's CRS.
            max_pixels (int, optional): The maximum number of pixels. Defaults to int(1e13).
            description_prefix (str, optional): The description prefix. Defaults to "export".

        Returns:
            None: (queues export tasks)
        """
        ic = self.collection
        if dates is None:
            dates = self.dates
        try:
            ee.data.createAsset({'type': 'ImageCollection'}, asset_collection_path)
        except Exception:
            pass

        for date_str in dates:
            img = ee.Image(ic.filter(ee.Filter.eq('Date_Filter', date_str)).first())
            asset_id = asset_collection_path + "/" + filename_prefix + date_str
            desc = description_prefix + "_" + filename_prefix + date_str

            params = {
                'image': img,
                'description': desc,
                'assetId': asset_id,
                'region': region,
                'scale': scale,
                'maxPixels': max_pixels
            }
            if crs:
                params['crs'] = crs

            ee.batch.Export.image.toAsset(**params).start()

        print("Queued", len(dates), "export tasks to", asset_collection_path)
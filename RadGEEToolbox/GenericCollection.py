import ee
import pandas as pd
import numpy as np
import warnings


class GenericCollection:
    """
    Represents a user-defined RadGEEToolbox class collection of any ee.ImageCollection from Google Earth Engine (GEE).

    This class enables simplified definition, filtering, masking, and processing of generic geospatial imagery.
    It supports multiple spatial and temporal filters, and caching for efficient computation. It also includes utilities for cloud masking,
    mosaicking, zonal statistics, and transect analysis.

    Initialization can be done by providing filtering parameters or directly passing in a pre-filtered GEE collection.

    Inspect the documentation or source code for details on the methods and properties available.

    Args:
        start_date (str): Start date in 'YYYY-MM-dd' format. Required unless `collection` is provided.
        end_date (str): End date in 'YYYY-MM-dd' format. Required unless `collection` is provided.
        boundary (ee.Geometry, optional): A geometry for filtering to images that intersect with the boundary shape. Overrides `tile_path` and `tile_row` if provided.
        collection (ee.ImageCollection, optional): A pre-filtered Landsat ee.ImageCollection object to be converted to a GenericCollection object. Overrides all other filters.

    Attributes:
        collection (ee.ImageCollection): The filtered or user-supplied image collection converted to an ee.ImageCollection object.

    Raises:
        ValueError: Raised if required filter parameters are missing, or if both `collection` and other filters are provided.

    Note:
        See full usage examples in the documentation or notebooks:
        https://github.com/radwinskis/RadGEEToolbox/tree/main/Example%20Notebooks

    """

    def __init__(
        self,
        collection=None,
        start_date=None,
        end_date=None,
        boundary=None,
        _dates_list=None
    ):
        if collection is None:
            raise ValueError(
                "The required `collection` argument has not been provided. Please specify an input ee.ImageCollection."
            )

        if isinstance(collection, GenericCollection):
            base_collection = collection.collection
        else:
            # Otherwise, assume it's a valid ee.ImageCollection
            base_collection = collection

        if (start_date is not None and end_date is None) or \
           (start_date is None and end_date is not None):
            raise ValueError("Please provide both start_date and end_date, or provide neither for entire collection")

        self.collection = base_collection
        self.start_date = start_date
        self.end_date = end_date
        self.boundary = boundary

        if self.start_date and self.end_date:
            if self.boundary:
                self.collection = self.get_boundary_and_date_filtered_collection()
            else:
                self.collection = self.get_filtered_collection()
        elif self.boundary:
            self.collection = self.get_boundary_filtered_collection()
        else:
            self.collection = self.get_generic_collection()

        self._dates_list = _dates_list
        self._dates = None
        self._geometry_masked_collection = None
        self._geometry_masked_out_collection = None
        self._median = None
        self._monthly_median = None
        self._monthly_mean = None
        self._monthly_sum = None
        self._monthly_max = None
        self._monthly_min = None
        self._yearly_median = None
        self._yearly_mean = None
        self._yearly_max = None
        self._yearly_min = None
        self._yearly_sum = None
        self._mean = None
        self._max = None
        self._min = None
        self._MosaicByDate = None
        self._PixelAreaSumCollection = None
        self._daily_aggregate_collection = None

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
    def anomaly_fn(image, geometry, band_name=None, anomaly_band_name=None, replace=True, scale=None):
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

        Returns:
            ee.Image: An ee.Image where each band represents the anomaly (deviation from
                        the mean) for that band. The output image retains the same band name.
        """
        if band_name:
            band_name = band_name
        else:
            band_name = ee.String(image.bandNames().get(0))

        image_to_process = image.select([band_name])

        image_scale = image_to_process.projection().nominalScale()

        # If the user supplies a numeric scale (int/float), keep it; otherwise default to image projection scale.
        scale_value = scale if scale is not None else image_scale  # Can be Python number or ee.Number

        # Compute mean over geometry at chosen scale (scale_value may be ee.Number).
        mean_image = image_to_process.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale_value,
            maxPixels=1e13
        ).toImage()

        # Compute the anomaly by subtracting the mean image from the input image.
        if scale is None:
            anomaly_image = image_to_process.subtract(mean_image)
        else:
            anomaly_image = image_to_process.reproject(crs=image_to_process.projection(), scale=scale_value).subtract(mean_image)

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

    @staticmethod
    def mask_via_band_fn(image, band_to_mask, band_for_mask, threshold, mask_above=False, add_band_to_original_image=False):
        """
        Masks pixels of interest from a specified band of a target image, based on a specified reference band and threshold.
        Designed for single image input which contains both the target and reference band.
        Example use case is masking vegetation from image when targeting land pixels. Can specify whether to mask pixels above or below the threshold.

        Args:
            image (ee.Image): input ee.Image
            band_to_mask (str): name of the band which will be masked (target image)
            band_for_mask (str): name of the band to use for the mask (band you want to remove/mask from target image)
            threshold (float): value where pixels less or more than threshold (depending on `mask_above` argument) will be masked
            mask_above (bool): if True, masks pixels above the threshold; if False, masks pixels below the threshold

        Returns:
            ee.Image: masked ee.Image
        """
    
        band_to_mask_image = image.select(band_to_mask)
        band_for_mask_image = image.select(band_for_mask)
 
        mask = band_for_mask_image.lte(threshold) if mask_above else band_for_mask_image.gte(threshold)

        if add_band_to_original_image:
            return image.addBands(band_to_mask_image.updateMask(mask).rename(band_to_mask), overwrite=True)
        else:
            return ee.Image(band_to_mask_image.updateMask(mask).rename(band_to_mask).copyProperties(image).set('system:time_start', image.get('system:time_start')))
    
    @staticmethod
    def mask_via_singleband_image_fn(image_to_mask, image_for_mask, threshold, band_name_to_mask=None, band_name_for_mask=None, mask_above=True):
        """
        Masks pixels of interest from a specified band of a target image, based on a specified reference band and threshold.
        Designed for the case where the target and reference bands are in separate images.
        Example use case is masking vegetation from image when targeting land pixels. Can specify whether to mask pixels above or below the threshold.

        Args:
            image_to_mask (ee.Image): image which will be masked (target image). If multiband, only the first band will be masked.
            image_for_mask (ee.Image): image to use for the mask (image you want to remove/mask from target image). If multiband, only the first band will be used for the masked.
            threshold (float): value where pixels less or more than threshold (depending on `mask_above` argument) will be masked
            band_name_to_mask (str, optional): name of the band in image_to_mask to be masked. If None, the first band will be used.
            band_name_for_mask (str, optional): name of the band in image_for_mask to be used for masking. If None, the first band will be used.
            mask_above (bool): if True, masks pixels above the threshold; if False, masks pixels below the threshold.

        Returns:
            ee.Image: masked ee.Image
        """
        if band_name_to_mask is None:
            band_to_mask = ee.String(image_to_mask.bandNames().get(0))
        else:
            band_to_mask = ee.String(band_name_to_mask)

        if band_name_for_mask is None:
            band_for_mask = ee.String(image_for_mask.bandNames().get(0))
        else:
            band_for_mask = ee.String(band_name_for_mask)

        band_to_mask_image = image_to_mask.select(band_to_mask)
        band_for_mask_image = image_for_mask.select(band_for_mask)
        if mask_above:
            mask = band_for_mask_image.gt(threshold)
        else:
            mask = band_for_mask_image.lt(threshold)
        return band_to_mask_image.updateMask(mask).rename(band_to_mask).copyProperties(image_to_mask).set('system:time_start', image_to_mask.get('system:time_start'))
    
    @staticmethod
    def band_rename_fn(image, current_band_name, new_band_name):
        """Renames a band in an ee.Image (single- or multi-band) in-place.

        Replaces the band named `current_band_name` with `new_band_name` without
        retaining the original band name. If the band does not exist, returns the
        image unchanged.

        Args:
            image (ee.Image): The input image (can be multiband).
            current_band_name (str): The existing band name to rename.
            new_band_name (str): The desired new band name.

        Returns:
            ee.Image: The image with the band renamed (or unchanged if not found).
        """
        img = ee.Image(image)
        current = ee.String(current_band_name)
        new = ee.String(new_band_name)

        band_names = img.bandNames()
        has_band = band_names.contains(current)

        def _rename():
            # Build a new band-name list with the target name replaced.
            new_names = band_names.map(
                lambda b: ee.String(
                    ee.Algorithms.If(ee.String(b).equals(current), new, b)
                )
            )
            # Rename the image using the updated band-name list.
            return img.rename(ee.List(new_names))

        out = ee.Image(ee.Algorithms.If(has_band, _rename(), img))
        return out.copyProperties(img).set('system:time_start', img.get('system:time_start'))

    @staticmethod
    def pixelAreaSum(
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
        return final_image #.set('system:time_start', image.get('system:time_start'))

    @staticmethod
    def PixelAreaSum(
        image, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12
    ):
        warnings.warn(
            "The `PixelAreaSum` static method is deprecated and will be removed in future versions. Please use the `pixelAreaSum` static method instead.",
            DeprecationWarning,
            stacklevel=2)
        return GenericCollection.pixelAreaSum(
            image, band_name, geometry, threshold, scale, maxPixels
        )

    def pixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12, output_type='ImageCollection', area_data_export_path=None
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
            output_type (str): 'ImageCollection' or 'ee.ImageCollection' to return an ee.ImageCollection, 'GenericCollection' to return a GenericCollection object, or 'DataFrame', 'Pandas', 'pd', 'dataframe', 'df' to return a pandas DataFrame (defaults to 'ImageCollection').
            area_data_export_path (str, optional): If provided, the function will save the resulting area data to a CSV file at the specified path.

        Returns:
            ee.ImageCollection or GenericCollection: Image collection of images with area calculation (square meters) stored as property matching name of band. Type of output depends on output_type argument.
        """
        # If the area calculation has not been computed for this GenericCollection instance, the area will be calculated for the provided bands
        if self._PixelAreaSumCollection is None:
            collection = self.collection
            # Area calculation for each image in the collection, using the PixelAreaSum function
            AreaCollection = collection.map(
                lambda image: GenericCollection.pixelAreaSum(
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
            GenericCollection(collection=self._PixelAreaSumCollection).exportProperties(property_names=prop_names, file_path=area_data_export_path+'.csv')
        # Returning the result in the desired format based on output_type argument or raising an error for invalid input
        if output_type == 'ImageCollection' or output_type == 'ee.ImageCollection':
            return self._PixelAreaSumCollection
        elif output_type == 'GenericCollection':
            return GenericCollection(collection=self._PixelAreaSumCollection)
        elif output_type == 'DataFrame' or output_type == 'Pandas' or output_type == 'pd' or output_type == 'dataframe' or output_type == 'df':
            return GenericCollection(collection=self._PixelAreaSumCollection).exportProperties(property_names=prop_names)
        else:
            raise ValueError("Incorrect `output_type`. The `output_type` argument must be one of the following: 'ImageCollection', 'ee.ImageCollection', 'GenericCollection', 'DataFrame', 'Pandas', 'pd', 'dataframe', or 'df'.")

    def PixelAreaSumCollection(
        self, band_name, geometry, threshold=-1, scale=30, maxPixels=1e12, output_type='ImageCollection', area_data_export_path=None
    ):
        warnings.warn(
            "The `PixelAreaSumCollection` method is deprecated and will be removed in future versions. Please use the `pixelAreaSumCollection` method instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.pixelAreaSumCollection(
            band_name, geometry, threshold, scale, maxPixels, output_type, area_data_export_path
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
            GenericCollection: A GenericCollection image collection with the 'month' property added to each image.
        """
        col = self.collection.map(GenericCollection.add_month_property_fn)
        return GenericCollection(collection=col)

    def combine(self, other):
        """
        Combines the current GenericCollection with another GenericCollection, using the `combine` method.

        Args:
            other (GenericCollection): Another GenericCollection to combine with current collection.

        Returns:
            GenericCollection: A new GenericCollection containing images from both collections.
        """
        # Checking if 'other' is an instance of GenericCollection
        if not isinstance(other, GenericCollection):
            raise ValueError("The 'other' parameter must be an instance of GenericCollection.")
        
        # Merging the collections using the .combine() method
        merged_collection = self.collection.combine(other.collection)
        return GenericCollection(collection=merged_collection)

    def merge(self, collections=None, multiband_collection=None, date_key='Date_Filter'):
        """
        Merge many singleband GenericCollection products into the parent collection, 
        or merge a single multiband collection with parent collection,
        pairing images by exact Date_Filter and returning one multiband image per date.

        NOTE: if you want to merge two multiband collections, use the `combine` method instead.

        Args:
            collections (list): List of singleband collections to merge with parent collection, effectively adds one band per collection to each image in parent
            multiband_collection (GenericCollection, optional): A multiband collection to merge with parent. Specifying a collection here will override `collections`.
            date_key (str): image property key for exact pairing (default 'Date_Filter')

        Returns:
            GenericCollection: parent with extra single bands attached (one image per date)
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
                merged = merged.copyProperties(a, a.propertyNames())
                merged = merged.set(date_key, a.get(date_key))
                return ee.Image(merged)

            return GenericCollection(collection=ee.ImageCollection(paired.map(_pair_two)))

        # Preferred path: merge many singleband products into the parent
        if not isinstance(collections, list) or len(collections) == 0:
            raise ValueError("Provide a non-empty list of GenericCollection objects in `collections`.")

        result = self.collection
        for extra in collections:
            if not isinstance(extra, GenericCollection):
                raise ValueError("All items in `collections` must be GenericCollection objects.")

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
                merged = merged.copyProperties(parent, parent.propertyNames())
                merged = merged.set(date_key, parent.get(date_key))
                return ee.Image(merged)

            result = ee.ImageCollection(paired.map(_attach_one))

        return GenericCollection(collection=result)

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
    
    def remove_duplicate_dates(self, sort_by='system:time_start', ascending=True):
        """
        Removes duplicate images that share the same date, keeping only the first one encountered.
        Useful for handling duplicate acquisitions or overlapping path/rows.
        
        Args:
            sort_by (str): Property to sort by before filtering distinct dates. Defaults to 'system:time_start' which is a global property. 
                    Take care to provide a property that exists in all images if using a custom property.
            ascending (bool): Sort order. Defaults to True.

        Returns:
            GenericCollection: A new GenericCollection object with distinct dates.
        """

        # Sort the collection to ensure the "best" image comes first (e.g. least cloudy)
        sorted_col = self.collection.sort(sort_by, ascending)
        
        # distinct() retains the first image for each unique value of the specified property
        distinct_col = sorted_col.distinct('Date_Filter')
        
        return GenericCollection(collection=distinct_col)

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
        df = GenericCollection.ee_to_df(feature_collection, columns=all_properties_to_fetch)
        
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
            "The `ExportProperties` method is deprecated and will be removed in future versions. Please use the `exportProperties` method instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.exportProperties(property_names, file_path)

    def get_generic_collection(self):
        """
        Filters image collection based on GenericCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from GenericCollection object
        """
        filtered_collection = (
            self.collection
            .map(GenericCollection.image_dater)
            .sort("Date_Filter")
        )
        return filtered_collection

    def get_filtered_collection(self):
        """
        Filters image collection based on GenericCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when tile info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from GenericCollection object
        """
        filtered_collection = (
            self.collection
            .filterDate(ee.Date(self.start_date), ee.Date(self.end_date).advance(1, 'day'))
            .map(GenericCollection.image_dater)
            .sort("Date_Filter")
        )
        return filtered_collection

    def get_boundary_filtered_collection(self):
        """
        Filters and masks image collection based on GenericCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from GenericCollection object

        """
        filtered_collection = (
            self.collection
            .filterBounds(self.boundary)
            .map(GenericCollection.image_dater)
            .sort("Date_Filter")
        )
        return filtered_collection

    def get_boundary_and_date_filtered_collection(self):
        """
        Filters and masks image collection based on GenericCollection class arguments. Automatically calculated when using collection method, depending on provided class arguments (when boundary info is provided).

        Returns:
            ee.ImageCollection: Filtered image collection - used for subsequent analyses or to acquire ee.ImageCollection from GenericCollection object

        """
        filtered_collection = (
            self.collection
            .filterDate(ee.Date(self.start_date), ee.Date(self.end_date).advance(1, 'day'))
            .filterBounds(self.boundary)
            .map(GenericCollection.image_dater)
            .sort("Date_Filter")
        )
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
    
    @property
    def monthly_median_collection(self):
        """Creates a monthly median composite from a GenericCollection image collection.

        This function computes the median for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            GenericCollection: A new GenericCollection object with monthly median composites.
        """
        if self._monthly_median is None:
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
                }).reproject(target_proj)

            # Map the composite function over the list of month start dates.
            monthly_composites_list = month_starts.map(create_monthly_composite)

            # Convert the list of images into an ee.ImageCollection.
            monthly_collection = ee.ImageCollection.fromImages(monthly_composites_list)

            # Filter out any composites that were created from zero images.
            # This prevents empty/masked images from being in the final collection.
            final_collection = GenericCollection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_median = final_collection
        else:
            pass

        return self._monthly_median
    
    @property
    def monthly_mean_collection(self):
        """Creates a monthly mean composite from a GenericCollection image collection.

        This function computes the mean for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            GenericCollection: A new GenericCollection object with monthly mean composites.
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
            final_collection = GenericCollection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_mean = final_collection
        else:
            pass

        return self._monthly_mean
    
    @property
    def monthly_sum_collection(self):
        """Creates a monthly sum composite from a GenericCollection image collection.

        This function computes the sum for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            GenericCollection: A new GenericCollection object with monthly sum composites.
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
            final_collection = GenericCollection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_sum = final_collection
        else:
            pass

        return self._monthly_sum
    
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
            Object: A new instance of the same class (e.g., GenericCollection) containing the yearly mean composites.
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

            self._yearly_mean = GenericCollection(collection=final_collection)
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
            Object: A new instance of the same class (e.g., GenericCollection) containing the yearly median composites.
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

            self._yearly_median = GenericCollection(collection=final_collection)
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
            Object: A new instance of the same class (e.g., GenericCollection) containing the yearly max composites.
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

            self._yearly_max = GenericCollection(collection=final_collection)
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
            Object: A new instance of the same class (e.g., GenericCollection) containing the yearly min composites.
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

            self._yearly_min = GenericCollection(collection=final_collection)
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
            Object: A new instance of the same class (e.g., GenericCollection) containing the yearly sum composites.
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

            self._yearly_sum = GenericCollection(collection=final_collection)
        else: 
            pass
        return self._yearly_sum

    @property
    def monthly_max_collection(self):
        """Creates a monthly max composite from a GenericCollection image collection.

        This function computes the max for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            GenericCollection: A new GenericCollection object with monthly max composites.
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
            final_collection = GenericCollection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_max = final_collection
        else:
            pass

        return self._monthly_max
    
    @property
    def monthly_min_collection(self):
        """Creates a monthly min composite from a GenericCollection image collection.

        This function computes the min for each
        month within the collection's date range, for each band in the collection. It automatically handles the full
        temporal extent of the input collection.

        The resulting images have a 'system:time_start' property set to the
        first day of each month and an 'image_count' property indicating how
        many images were used in the composite. Months with no images are
        automatically excluded from the final collection.

        NOTE: the day of month for the 'system:time_start' property is set to the earliest date of the first month observed and may not be the first day of the month.

        Returns:
            GenericCollection: A new GenericCollection object with monthly min composites.
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
            final_collection = GenericCollection(collection=monthly_collection.filter(ee.Filter.gt('image_count', 0)))
            self._monthly_min = final_collection
        else:
            pass

        return self._monthly_min
    
    @property
    def daily_aggregate_collection_from_properties(self):
        """
        Property attribute to aggregate (sum) collection images that share the same date.

        This is useful for collections with multiple images per day (e.g., 3-hour SMAP data)
        that need to be converted to a daily sum. It uses the 'Date_Filter' property
        to group images. The 'system:time_start' of the first image of the day
        is preserved. Server-side friendly.

        NOTE: This function sums all bands.

        Returns:
            GenericCollection: GenericCollection image collection with daily summed imagery.
        """
        if self._daily_aggregate_collection is None:
            input_collection = self.collection

            # Function to sum images of the same date and accumulate them
            def sum_and_accumulate(date, list_accumulator):
                # Cast inputs to server-side objects
                date = ee.String(date)
                list_accumulator = ee.List(list_accumulator)
                
                # Filter collection to only images from this date
                date_filter = ee.Filter.eq("Date_Filter", date)
                date_collection = input_collection.filter(date_filter)

                # Get the first image of the day to use for its metadata
                first_image = ee.Image(date_collection.first())

                # Reduce the daily collection by summing all images
                # This creates a single image where each pixel is the sum
                # of all pixels from that day.
                daily_sum = date_collection.sum()

                # --- Property Management ---
                # Copy the 'system:time_start' from the first image of the
                # day to the new daily-summed image. This is critical.
                props_to_copy = ["system:time_start"]
                daily_sum = daily_sum.copyProperties(first_image, props_to_copy)
                
                # Set the 'Date_Filter' property (since .sum() doesn't preserve it)
                daily_sum = daily_sum.set("Date_Filter", date)

                # Also add a property to know how many images were summed
                image_count = date_collection.size()
                daily_sum = daily_sum.set('images_summed', image_count)
                
                # Add the new daily image to our list
                return list_accumulator.add(daily_sum)

            # Get a server-side list of all unique dates in the collection
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create sums and accumulate them in a list
            summed_list = distinct_dates.iterate(sum_and_accumulate, initial)

            # Convert the list of summed images to an ImageCollection
            new_col = ee.ImageCollection.fromImages(summed_list)
            
            # Cache the result as a new GenericCollection
            self._daily_aggregate_collection = GenericCollection(collection=new_col)

        return self._daily_aggregate_collection
    
    def daily_aggregate_collection(self, method='algorithmic'):
        """
        Aggregates (sums) collection images that share the same date.

        This is useful for collections with multiple images per day (e.g., 3-hour SMAP data)
        that need to be converted to a daily sum. It uses the 'Date_Filter' property
        to group images. The 'system:time_start' of the first image of the day
        is preserved. Server-side friendly.

        Args:
            method (str, optional): The method for generating the list of unique dates.
                - 'algorithmic' (default): Generates dates from self.start_date and
                  self.end_date. This is highly efficient and robust for large
                  collections. Requires start/end dates to be set on the object.
                - 'aggregate': Scans the entire collection for unique 'Date_Filter'
                  properties. This can cause a 'User memory limit exceeded' error
                  on very large collections.

        Returns:
            GenericCollection: A new GenericCollection image collection with daily summed imagery.
        
        Raises:
            ValueError: If 'algorithmic' method is used but self.start_date or
                        self.end_date are not set.
        """
        input_collection = self.collection

        # --- Select the method for generating the date list ---
        if method == 'algorithmic':
            # Check that start/end dates are available on the object
            if not self.start_date or not self.end_date:
                raise ValueError(
                    "The 'algorithmic' method requires start_date and end_date to be "
                    "set on the GenericCollection object. Initialize the object "
                    "with start_date and end_date, or use method='aggregate'."
                )
            
            # 1. Get ee.Date objects from the instance properties
            start_date = ee.Date(self.start_date)
            end_date = ee.Date(self.end_date)

            # 2. Calculate the total number of days in the range
            num_days = end_date.difference(start_date, 'day').round()

            # 3. Create a server-side list of all day-starting numbers
            day_numbers = ee.List.sequence(0, num_days)

            # 4. Map over the numbers to create a list of 'YYYY-MM-DD' date strings
            def get_date_string(n):
                return start_date.advance(n, 'day').format('YYYY-MM-dd')
            
            distinct_dates = day_numbers.map(get_date_string)

        elif method == 'aggregate':
            # This is the original, memory-intensive method.
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()
        
        else:
            raise ValueError(f"Unknown method '{method}'. Must be 'algorithmic' or 'aggregate'.")
        # --- End of date list generation ---

        # Function to sum images of the same date and accumulate them
        def sum_and_accumulate(date, list_accumulator):
            # Cast inputs to server-side objects
            date = ee.String(date)
            list_accumulator = ee.List(list_accumulator)
            
            # Filter collection to only images from this date
            date_filter = ee.Filter.eq("Date_Filter", date)
            date_collection = input_collection.filter(date_filter)

            # Check if any images actually exist for this day
            image_count = date_collection.size()

            # Define the summing function to be run *only* if images exist
            def if_images_exist():
                # Get the first image of the day to use for its metadata
                first_image = ee.Image(date_collection.first())
                
                # Reduce the daily collection by summing all images
                daily_sum = date_collection.sum()
                
                # Copy 'system:time_start' from the first image
                props_to_copy = ["system:time_start"]
                daily_sum = daily_sum.copyProperties(first_image, props_to_copy)
                
                # Set the 'Date_Filter' property
                daily_sum = daily_sum.set("Date_Filter", date)
                daily_sum = daily_sum.set('images_summed', image_count)
                
                # Add the new daily image to our list
                return list_accumulator.add(daily_sum)

            # Use ee.Algorithms.If to run the sum *only* if image_count > 0
            # This avoids errors from calling .first() or .sum() on empty collections
            return ee.Algorithms.If(
                image_count.gt(0),
                if_images_exist(),  # if True
                list_accumulator    # if False (just return the list unchanged)
            )
            
        # Initialize an empty list as the accumulator
        initial = ee.List([])

        # Iterate over each date to create sums and accumulate them
        summed_list = distinct_dates.iterate(sum_and_accumulate, initial)

        # Convert the list of summed images to an ImageCollection
        new_col = ee.ImageCollection.fromImages(summed_list)
        
        # Return the new GenericCollection wrapper
        return GenericCollection(collection=new_col)
    
    def daily_aggregate_collection_via_join(self, method='algorithmic'):
        """
        Aggregates (sums) collection images that share the same date based on a join approach.
        
        Args:
            method (str): The method for which to aggregate the daily collection. Options are 'algorithmic' (default) and 'aggregate'.
                            The algorithmic method is server-side friendly while the aggregate method makes client-side calls. 
                            Algorithmic is more efficient and chosen as the default.

        Returns:
            Image Collection (GenericCollection): The daily aggregated image collection as a GenericCollection object type.
        
        """
        input_collection = self.collection

        if method == 'algorithmic':
            if not self.start_date or not self.end_date:
                raise ValueError(
                    "The 'algorithmic' method requires start_date and end_date to be "
                    "set on the GenericCollection object. Initialize the object "
                    "with start_date and end_date, or use method='aggregate'."
                )
            
            start_date = ee.Date(self.start_date)
            end_date = ee.Date(self.end_date)
            num_days = end_date.difference(start_date, 'day').round()
            day_numbers = ee.List.sequence(0, num_days)

            def get_date_string(n):
                return start_date.advance(n, 'day').format('YYYY-MM-dd')
            
            distinct_dates = day_numbers.map(get_date_string) # This is our server-side list

        elif method == 'aggregate':
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()
        
        else:
            raise ValueError(f"Unknown method '{method}'. Must be 'algorithmic' or 'aggregate'.")
        
        def create_date_feature(date_str):
            return ee.Feature(None, {'Date_Filter': ee.String(date_str)})
        
        dummy_dates_fc = ee.FeatureCollection(distinct_dates.map(create_date_feature))

        date_filter = ee.Filter.equals(leftField='Date_Filter', rightField='Date_Filter')
        join = ee.Join.saveAll(matchesKey='matches')
        joined_fc = join.apply(dummy_dates_fc, input_collection, date_filter)

        def sum_daily_images(feature_with_matches):
            images_list = ee.List(feature_with_matches.get('matches'))
            image_count = images_list.size()

            # Define a function to run *only* if the list is not empty
            def if_images_exist():
                image_collection_for_day = ee.ImageCollection.fromImages(images_list)
                first_image = ee.Image(image_collection_for_day.first())
                daily_sum = image_collection_for_day.sum()
                daily_sum = daily_sum.copyProperties(first_image, ["system:time_start"])
                daily_sum = daily_sum.set(
                    'Date_Filter', feature_with_matches.get('Date_Filter'),
                    'images_summed', image_count
                )
                return daily_sum

            # Use ee.Algorithms.If. If count is 0, return a *null* image.
            return ee.Algorithms.If(
                image_count.gt(0),
                if_images_exist(),  # if True
                None                # if False (return None)
            )

        # Map the robust function, and use dropNulls=True to filter out
        # any days that had no images (and returned None).
        image_collection = joined_fc.map(sum_daily_images, dropNulls=True)
        
        # Explicitly cast to an ImageCollection to avoid client-side confusion
        final_collection = ee.ImageCollection(image_collection)
  
        return GenericCollection(
            collection=final_collection,
            start_date=self.start_date, # Pass along other metadata
            end_date=self.end_date,
            boundary=self.boundary,
            _dates_list=distinct_dates  # <-- This is the key
        )

    def export_daily_sum_to_asset(
        self,
        asset_collection_path,
        region,
        scale,
        filename_prefix="",
        crs=None,
        max_pixels=int(1e13),
        description_prefix="export"
    ):
        """
        Exports a daily-summed (aggregated) collection to a GEE Asset Collection.

        This function is designed to be called from a collection with
        sub-daily data (e.g., 3-hourly). It efficiently creates one
        small, independent export task for each day by summing *only*
        that day's images. This avoids the re-computing of an entire collection per image task performance pitfall.

        It requires self.start_date and self.end_date to be set on the
        GenericCollection object.

        Args:
            asset_collection_path (str): The path to the asset collection.
            region (ee.Geometry): The region to export.
            scale (int): The scale of the export.
            filename_prefix (str, optional): The filename prefix. Defaults to "", i.e. blank.
            crs (str, optional): The coordinate reference system. Defaults to None.
            max_pixels (int, optional): The maximum number of pixels. Defaults to int(1e13).
            description_prefix (str, optional): The description prefix. Defaults to "export".

        Returns:
            None: (queues export tasks)
        """
        # This is the *original* 3-hourly (or sub-daily) collection
        original_collection = self.collection

        # --- 1. Algorithmic Date Generation ---
        if not self.start_date or not self.end_date:
            raise ValueError(
                "export_daily_sum_to_asset requires start_date and end_date "
                "to be set on the GenericCollection object."
            )
        
        start_date = ee.Date(self.start_date)
        end_date = ee.Date(self.end_date)
        num_days = end_date.difference(start_date, 'day').round()
        day_numbers = ee.List.sequence(0, num_days)

        def get_date_string(n):
            # Use lowercase 'dd' for day of month!
            return start_date.advance(n, 'day').format('YYYY-MM-dd')
        
        # Get a client-side list of all dates to loop over
        date_list = day_numbers.map(get_date_string).getInfo()
        # --- End of Date Generation ---

        # --- 2. Create Asset Collection (if needed) ---
        try:
            ee.data.getAsset(asset_collection_path)
        except ee.EEException:
            print(f"Creating new asset collection: {asset_collection_path}")
            ee.data.createAsset({'type': 'ImageCollection'}, asset_collection_path)

        print(f"Queuing {len(date_list)} small, daily-sum export tasks...")

        # --- 3. Loop and Create Tiny Tasks ---
        for date_str in date_list:
            
            # --- This is the simple, efficient recipe for *one* day ---
            
            # 1. Filter the *original* collection for just this one day
            daily_images = original_collection.filter(
                ee.Filter.eq('Date_Filter', date_str)
            )
            
            # 2. Get the first image for metadata
            first_image = daily_images.first()
            
            # 3. Create the daily sum
            daily_sum = daily_images.sum()
            
            # 4. Set properties
            daily_sum = ee.Image(daily_sum.copyProperties(first_image, ["system:time_start"]))
            daily_sum = daily_sum.set(
                'Date_Filter', date_str,
                'images_summed', daily_images.size()
            )
            # --- End of recipe ---

            # Define asset ID and description
            asset_id = asset_collection_path + "/" + filename_prefix + date_str
            desc = description_prefix + "_" + filename_prefix + date_str

            params = {
                'image': daily_sum,
                'description': desc,
                'assetId': asset_id,
                'region': region,
                'scale': scale,
                'maxPixels': max_pixels
            }
            if crs:
                params['crs'] = crs

            # Start the server-side export task
            ee.batch.Export.image.toAsset(**params).start()

        print("All", len(date_list), "export tasks queued to", asset_collection_path)

    def smap_flux_to_mm(self):
        """
        Converts a daily-summed SMAP flux collection (kg/m²/s) 
        to a daily total amount (mm/day).

        This works by multiplying each image by 10800 
        (3 hours * 60 min/hr * 60 sec/min).
        
        Assumes 1 kg/m² = 1 mm of water.

        Returns:
            GenericCollection: A new collection with values in mm/day.
        """
        # Define the conversion function
        def convert_to_mm(image):
            # Get the original band name(s)
            band_names = image.bandNames()
            # Multiply and rename the bands to indicate the new units
            new_band_names = band_names.map(lambda b: ee.String(b).cat('_mm'))
            
            converted_image = image.multiply(10800).rename(new_band_names)
            return converted_image.copyProperties(image, image.propertyNames()).set('system:time_start', image.get('system:time_start'))

        # Map the function over the entire collection
        converted_collection = self.collection.map(convert_to_mm)
        
        # Return a new GenericCollection object
        return GenericCollection(
            collection=converted_collection,
            start_date=self.start_date,
            end_date=self.end_date,
            boundary=self.boundary,
            _dates_list=self._dates_list # Pass along the cached dates!
        )

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
            image_collection (GenericCollection or ee.ImageCollection): The input image collection for which the Mann-Kendall and Sen's slope trend statistics will be calculated.
            target_band (str): The band name to be used for the output anomaly image. e.g. 'ndvi'
            join_method (str, optional): The method used to join images in the collection. Options are 'system:time_start' or 'Date_Filter'. Default is 'system:time_start'.
            geometry (ee.Geometry, optional): An ee.Geometry object to limit the area over which the trend statistics are calculated and mask the output image. Default is None.

        Returns:
            ee.Image: An image with the following bands: 's_statistic', 'variance', 'z_score', 'confidence', and 'slope'.
        """
        ########## PART 1 - S-VALUE CALCULATION ##########
        ##### https://vsp.pnnl.gov/help/vsample/design_trend_mann_kendall.htm #####
        image_collection = self
        if isinstance(image_collection, GenericCollection):
            image_collection = image_collection.collection
        elif isinstance(image_collection, ee.ImageCollection):
            pass
        else:
            raise ValueError(f'The chosen `image_collection`: {image_collection} is not a valid GenericCollection or ee.ImageCollection object.')
        
        if target_band is None:
            raise ValueError('The `target_band` parameter must be specified.')
        if not isinstance(target_band, str):
            raise ValueError(f'The chosen `target_band`: {target_band} is not a valid string.')

        if geometry is not None and not isinstance(geometry, ee.Geometry):
            raise ValueError(f'The chosen `geometry`: {geometry} is not a valid ee.Geometry object.')
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
        
        native_projection = image_collection.first().select(target_band).projection()

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
            if isinstance(image_collection, GenericCollection):
                image_collection = image_collection.collection
            elif isinstance(image_collection, ee.ImageCollection):
                pass
            else:
                raise ValueError(f'The chosen `image_collection`: {image_collection} is not a valid GenericCollection or ee.ImageCollection object.')
            
            if target_band is None:
                raise ValueError('The `target_band` parameter must be specified.')
            if not isinstance(target_band, str):
                raise ValueError(f'The chosen `target_band`: {target_band} is not a valid string.')

            if geometry is not None and not isinstance(geometry, ee.Geometry):
                raise ValueError(f'The chosen `geometry`: {geometry} is not a valid ee.Geometry object.')

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

            return slope_band
    

    def mask_to_polygon(self, polygon):
        """
        Function to mask GenericCollection image collection by a polygon (ee.Geometry), where pixels outside the polygon are masked out.

        Args:
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            GenericCollection: masked GenericCollection image collection

        """
        # Convert the polygon to a mask
        mask = ee.Image.constant(1).clip(polygon)

        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(mask)\
                                .copyProperties(img).set('system:time_start', img.get('system:time_start')))

        # Return the updated object
        return GenericCollection(collection=masked_collection)

    def mask_out_polygon(self, polygon):
        """
        Function to mask GenericCollection image collection by a polygon (ee.Geometry), where pixels inside the polygon are masked out.

        Args:
            polygon (ee.Geometry): ee.Geometry polygon or shape used to mask image collection.

        Returns:
            GenericCollection: masked GenericCollection image collection

        """
        # Convert the polygon to a mask
        full_mask = ee.Image.constant(1)

        # Use paint to set pixels inside polygon as 0
        area = full_mask.paint(polygon, 0)

        # Update the mask of each image in the collection
        masked_collection = self.collection.map(lambda img: img.updateMask(area)\
                            .copyProperties(img).set('system:time_start', img.get('system:time_start')))

        # Return the updated object
        return GenericCollection(collection=masked_collection)

    
    def binary_mask(self, threshold=None, band_name=None, classify_above_threshold=True, mask_zeros=False):
        """
        Function to create a binary mask (value of 1 for pixels above set threshold and value of 0 for all other pixels) of the GenericCollection image collection based on a specified band.
        If a singleband image is provided, the band name is automatically determined.
        If multiple bands are available, the user must specify the band name to use for masking.

        Args:
            threshold (float, optional): The threshold value for creating the binary mask. Defaults to None.
            band_name (str, optional): The name of the band to use for masking. Defaults to None.
            classifiy_above_threshold (bool, optional): If True, pixels above the threshold are classified as 1. If False, pixels below the threshold are classified as 1. Defaults to True.
            mask_zeros (bool, optional): If True, pixels with a value of 0 after the binary mask are masked out in the output binary mask. Useful for classifications. Defaults to False.

        Returns:
            GenericCollection: GenericCollection singleband image collection with binary masks applied.
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

        if classify_above_threshold:
            if mask_zeros:
                col = self.collection.map(
                    lambda image: image.select(band_name).gte(threshold).rename(band_name)
                    .updateMask(image.select(band_name).gt(0)).copyProperties(image)
                    .set('system:time_start', image.get('system:time_start'))
                )
            else:
                col = self.collection.map(
                    lambda image: image.select(band_name).gte(threshold).rename(band_name)
                    .copyProperties(image).set('system:time_start', image.get('system:time_start'))
                )
        else:
            if mask_zeros:
                col = self.collection.map(
                    lambda image: image.select(band_name).lte(threshold).rename(band_name)
                    .updateMask(image.select(band_name).gt(0)).copyProperties(image)
                    .set('system:time_start', image.get('system:time_start'))
                )
            else:
                col = self.collection.map(
                    lambda image: image.select(band_name).lte(threshold).rename(band_name)
                    .copyProperties(image).set('system:time_start', image.get('system:time_start'))
                )
        return GenericCollection(collection=col)
    
    def anomaly(self, geometry, band_name=None, anomaly_band_name=None, replace=True, scale=None):
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
            scale (int, optional): The scale (in meters) to use for the image reduction. If not provided, the nominal scale of the image will be used.

        Returns:
            GenericCollection: A GenericCollection where each image represents the anomaly (deviation from
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

        col = self.collection.map(lambda image: GenericCollection.anomaly_fn(image, geometry=geometry, band_name=band_name, anomaly_band_name=anomaly_band_name, replace=replace, scale=scale))
        return GenericCollection(collection=col)
    
    def mask_via_band(self, band_to_mask, band_for_mask, threshold=-1, mask_above=True, add_band_to_original_image=False):
        """
        Masks select pixels of a selected band from an image based on another specified band and threshold (optional). 
        Example use case is masking vegetation from image when targeting land pixels. Can specify whether to mask pixels above or below the threshold.

        Args:
            band_to_mask (str): name of the band which will be masked (target image)
            band_for_mask (str): name of the band to use for the mask (band you want to remove/mask from target image)
            threshold (float): value between -1 and 1 where pixels less than threshold will be masked; defaults to -1 assuming input band is already classified (masked to pixels of interest).
            mask_above (bool): if True, masks pixels above the threshold; if False, masks pixels below the threshold

        Returns:
            GenericCollection: A new GenericCollection with the specified band masked to pixels excluding from `band_for_mask`.
        """
        if self.collection.size().eq(0).getInfo():
            raise ValueError("The collection is empty.")

        col = self.collection.map(
            lambda image: GenericCollection.mask_via_band_fn(
                image,
                band_to_mask=band_to_mask,
                band_for_mask=band_for_mask,
                threshold=threshold,
                mask_above=mask_above,
                add_band_to_original_image=add_band_to_original_image
            )
        )
        return GenericCollection(collection=col)

    def mask_via_singleband_image(self, image_collection_for_mask, band_name_to_mask, band_name_for_mask, threshold=-1, mask_above=False, add_band_to_original_image=False):
        """
        Masks select pixels of a selected band from an image collection based on another specified singleband image collection and threshold (optional).
        Example use case is masking vegetation from image when targeting land pixels. Can specify whether to mask pixels above or below the threshold.
        This function pairs images from the two collections based on an exact match of the 'Date_Filter' property.
        
        Args:
            image_collection_for_mask (GenericCollection): GenericCollection image collection to use for masking (source of pixels that will be used to mask the parent image collection)
            band_name_to_mask (str): name of the band which will be masked (target image)
            band_name_for_mask (str): name of the band to use for the mask (band which contains pixels the user wants to remove/mask from target image)
            threshold (float): threshold value where pixels less (or more, depending on `mask_above`) than threshold will be masked; defaults to -1.
            mask_above (bool): if True, masks pixels above the threshold; if False, masks pixels below the threshold
            add_band_to_original_image (bool): if True, adds the band used for masking to the original image as an additional band; if False, only the masked band is retained in the output image.

        Returns:
            GenericCollection: A new GenericCollection with the specified band masked to pixels excluding from `band_for_mask`.
        """
        
        if self.collection.size().eq(0).getInfo():
            raise ValueError("The collection is empty.")
        if not isinstance(image_collection_for_mask, GenericCollection):
            raise ValueError("image_collection_for_mask must be a GenericCollection object.")
        size1 = self.collection.size().getInfo()
        size2 = image_collection_for_mask.collection.size().getInfo()
        if size1 != size2:
            raise ValueError(f"Warning: Collections have different sizes ({size1} vs {size2}). Please ensure both collections have the same number of images and matching dates.")
        if size1 == 0 or size2 == 0:
            raise ValueError("Warning: One of the input collections is empty.")

        # Pair by exact Date_Filter property
        primary   = self.collection.select([band_name_to_mask])
        secondary = image_collection_for_mask.collection.select([band_name_for_mask])
        join = ee.Join.inner()
        flt  = ee.Filter.equals(leftField='Date_Filter', rightField='Date_Filter')
        paired = join.apply(primary, secondary, flt)

        def _map_pair(f):
            f = ee.Feature(f)                     # <-- treat as Feature
            prim = ee.Image(f.get('primary'))     # <-- get the primary Image
            sec  = ee.Image(f.get('secondary'))   # <-- get the secondary Image

            merged = prim.addBands(sec.select([band_name_for_mask]))

            out = GenericCollection.mask_via_band_fn(
                merged,
                band_to_mask=band_name_to_mask,
                band_for_mask=band_name_for_mask,
                threshold=threshold,
                mask_above=mask_above,
                add_band_to_original_image=add_band_to_original_image
            )

            # guarantee single band + keep properties
            out = ee.Image(out).select([band_name_to_mask]).copyProperties(prim, prim.propertyNames())\
                                    .set('system:time_start', prim.get('system:time_start'))
            out = out.set('Date_Filter', prim.get('Date_Filter'))
            return ee.Image(out)                  # <-- return as Image

        col = ee.ImageCollection(paired.map(_map_pair))
        return GenericCollection(collection=col)

    def band_rename(self, current_band_name, new_band_name):
        """Renames a band in all images of the GenericCollection in-place.

        Replaces the band named `current_band_name` with `new_band_name` without
        retaining the original band name. If the band does not exist in an image,
        that image is returned unchanged.

        Args:
            current_band_name (str): The existing band name to rename.
            new_band_name (str): The desired new band name.

        Returns:
            GenericCollection: The GenericCollection with the band renamed in all images.
        """
        # check if `current_band_name` exists in the first image
        first_image = self.collection.first()
        has_band = first_image.bandNames().contains(current_band_name).getInfo()
        if not has_band:
            raise ValueError(f"Band '{current_band_name}' does not exist in the collection.")

        renamed_collection = self.collection.map(
            lambda img: self.band_rename_fn(img, current_band_name, new_band_name)
        )
        return GenericCollection(collection=renamed_collection)

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
            img_col: ee.ImageCollection with same dates as another GenericCollection image collection object.
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
        Selects ("grabs") image of a specific date in format of 'YYYY-MM-dd' - will not work correctly if collection is composed of multiple images of the same date.

        Args:
            img_date: date (str) of image to select in format of 'YYYY-MM-dd'

        Returns:
            ee.Image: ee.Image of selected image
        """
        new_col = self.collection.filter(ee.Filter.eq("Date_Filter", img_date))
        return new_col.first()

    def collectionStitch(self, img_col2):
        """
        Function to mosaic two GenericCollection objects which share image dates.
        Mosaics are only formed for dates where both image collections have images.
        Image properties are copied from the primary collection. Server-side friendly.

        Args:
            img_col2: secondary GenericCollection image collection to be mosaiced with the primary image collection

        Returns:
            GenericCollection: GenericCollection image collection
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

        # Return a GenericCollection instance
        return GenericCollection(collection=new_col)
    
    def CollectionStitch(self, img_col2):
        warnings.warn(
            "CollectionStitch is deprecated. Please use collectionStitch instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.collectionStitch(img_col2)

    @property
    def mosaicByDateDepr(self):
        """
        Property attribute function to mosaic collection images that share the same date.

        The property CLOUD_COVER for each image is used to calculate an overall mean,
        which replaces the CLOUD_COVER property for each mosaiced image.
        Server-side friendly.

        NOTE: if images are removed from the collection from cloud filtering, you may have mosaics composed of only one image.

        Returns:
            GenericCollection: GenericCollection image collection with mosaiced imagery and mean CLOUD_COVER as a property
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

                props_of_interest = [
                    "system:time_start"
                ]

                # mosaic = mosaic.copyProperties(self.image_grab(0), props_of_interest).set({
                #     'CLOUD_COVER': cloud_percentage
                # })
                mosaic = mosaic.copyProperties(first_image, props_of_interest)

                return list_accumulator.add(mosaic)

            # Get distinct dates
            distinct_dates = input_collection.aggregate_array("Date_Filter").distinct()

            # Initialize an empty list as the accumulator
            initial = ee.List([])

            # Iterate over each date to create mosaics and accumulate them in a list
            mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

            new_col = ee.ImageCollection.fromImages(mosaic_list)
            col = GenericCollection(collection=new_col)
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
                    "system:time_start",
                    "Date_Filter" 
                ]
                
                # Return mosaic with properties set
                return mosaic.copyProperties(img, props_of_interest)
            # 5. Map the function and wrap the result
            mosaiced_col = joined_col.map(_mosaic_day)
            self._MosaicByDate = GenericCollection(collection=mosaiced_col)

        # Convert the list of mosaics to an ImageCollection
        return self._MosaicByDate
    
    @property
    def MosaicByDate(self):
        warnings.warn(
            "MosaicByDate is deprecated. Please use mosaicByDate instead.",
            DeprecationWarning,
            stacklevel=2)
        return self.mosaicByDate

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
                return GenericCollection.ee_to_df(transect)
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
                transect_data = GenericCollection.extract_transect(
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
                transect_data = GenericCollection.extract_transect(
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
        dist_interval=90,
        n_segments=None,
        scale=30,
        processing_mode='aggregated',
        save_folder_path=None,
        sampling_method='line',
        point_buffer_radius=15,
        batch_size=10
    ):
        """
        Computes and returns pixel values along transects. Provide a list of ee.Geometry.LineString objects and corresponding names, and the function will compute the specified reducer value
        at regular intervals along each line for all images in the collection. Use `dist_interval` or `n_segments` to control sampling resolution. The user can choose between 'aggregated' mode (returns a dictionary of DataFrames) or 'iterative' mode (saves individual CSVs for each transect).
        Alter `sampling_method` to sample directly along the line or via buffered points along the line. Buffered points can help capture more representative pixel values in heterogeneous landscapes, and the buffer radius can be adjusted via `point_buffer_radius`.
        
        Args:
            lines (list): List of ee.Geometry.LineString objects.
            line_names (list): List of string names for each transect.
            reducer (str, optional): Reducer name. Defaults to 'mean'.
            dist_interval (float, optional): Distance interval in meters. Defaults to 90.
            n_segments (int, optional): Number of segments (overrides dist_interval).
            scale (int, optional): Scale in meters. Defaults to 30.
            processing_mode (str, optional): 'aggregated' or 'iterative'.
            save_folder_path (str, optional): Path to save CSVs.
            sampling_method (str, optional): 'line' or 'buffered_point'.
            point_buffer_radius (int, optional): Buffer radius if using 'buffered_point'.
            batch_size (int, optional): Images per request in 'aggregated' mode. Defaults to 10. Lower the value if you encounter a 'Too many aggregations' error.

        Returns:
            dict or None: Dictionary of DataFrames (aggregated) or None (iterative).
        """
        if len(lines) != len(line_names):
            raise ValueError("'lines' and 'line_names' must have the same number of elements.")

        first_img = self.collection.first()
        bands = first_img.bandNames().getInfo()
        is_multiband = len(bands) > 1
        
        # Setup robust dictionary for handling masked/zero values
        default_val = -9999
        dummy_dict = ee.Dictionary.fromLists(bands, ee.List.repeat(default_val, len(bands)))
        
        if is_multiband:
            reducer_cols = [f"{b}_{reducer}" for b in bands]
            clean_names = bands 
            rename_keys = bands
            rename_vals = reducer_cols
        else:
            reducer_cols = [reducer]
            clean_names = [bands[0]]
            rename_keys = bands
            rename_vals = reducer_cols

        print("Pre-computing transect geometries from input LineString(s)...")
        
        master_transect_fc = ee.FeatureCollection([])
        geom_error = 1.0 

        for i, line in enumerate(lines):
            line_name = line_names[i]
            length = line.length(geom_error)
            
            eff_interval = length.divide(n_segments) if n_segments else dist_interval
            
            distances = ee.List.sequence(0, length, eff_interval)
            cut_lines = line.cutLines(distances, geom_error).geometries()
            
            def create_feature(l):
                geom = ee.Geometry(ee.List(l).get(0))
                dist = ee.Number(ee.List(l).get(1))
                
                final_geom = ee.Algorithms.If(
                    ee.String(sampling_method).equals('buffered_point'),
                    geom.centroid(geom_error).buffer(point_buffer_radius),
                    geom
                )
                
                return ee.Feature(ee.Geometry(final_geom), {
                    'transect_name': line_name,
                    'distance': dist
                })

            line_fc = ee.FeatureCollection(cut_lines.zip(distances).map(create_feature))
            master_transect_fc = master_transect_fc.merge(line_fc)

        try:
            ee_reducer = getattr(ee.Reducer, reducer)()
        except AttributeError:
            raise ValueError(f"Unknown reducer: '{reducer}'.")

        def process_image(image):
            date_val = image.get('Date_Filter')
            
            # Map over points (Slower but Robust)
            def reduce_point(f):
                stats = image.reduceRegion(
                    reducer=ee_reducer,
                    geometry=f.geometry(),
                    scale=scale,
                    maxPixels=1e13
                )
                # Combine with defaults (preserves 0, handles masked)
                safe_stats = dummy_dict.combine(stats, overwrite=True)
                # Rename keys to match expected outputs (e.g. 'ndvi' -> 'ndvi_mean')
                final_stats = safe_stats.rename(rename_keys, rename_vals)
                
                return f.set(final_stats).set({'image_date': date_val})

            return master_transect_fc.map(reduce_point)

        export_cols = ['transect_name', 'distance', 'image_date'] + reducer_cols

        if processing_mode == 'aggregated':
            collection_size = self.collection.size().getInfo()
            print(f"Starting batch process of {collection_size} images...")
            
            dfs = []
            for i in range(0, collection_size, batch_size):
                print(f"  Processing image {i} to {min(i + batch_size, collection_size)}...")
                
                batch_col = ee.ImageCollection(self.collection.toList(batch_size, i))
                results_fc = batch_col.map(process_image).flatten()
                
                # Dynamic Class Call for ee_to_df
                df_batch = self.__class__.ee_to_df(results_fc, columns=export_cols, remove_geom=True)
                
                if not df_batch.empty:
                    dfs.append(df_batch)

            if not dfs:
                print("Warning: No transect data generated.")
                return {}

            df = pd.concat(dfs, ignore_index=True)

            # Post-Process & Split
            output_dfs = {}
            for col in reducer_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace(-9999, np.nan)

            for name in sorted(df['transect_name'].unique()):
                line_df = df[df['transect_name'] == name]
                
                for raw_col, band_name in zip(reducer_cols, clean_names):
                    try:
                        # Safety drop for duplicates
                        line_df_clean = line_df.drop_duplicates(subset=['distance', 'image_date'])
                        
                        pivot = line_df_clean.pivot(index='distance', columns='image_date', values=raw_col)
                        pivot.columns.name = 'Date'
                        key = f"{name}_{band_name}"
                        output_dfs[key] = pivot
                        
                        if save_folder_path:
                            safe_key = "".join(x for x in key if x.isalnum() or x in "._-")
                            fname = f"{save_folder_path}{safe_key}_transects.csv"
                            pivot.to_csv(fname)
                            print(f"Saved: {fname}")
                    except Exception as e:
                        print(f"Skipping pivot for {name}/{band_name}: {e}")

            return output_dfs

        elif processing_mode == 'iterative':
            if not save_folder_path:
                raise ValueError("save_folder_path is required for iterative mode.")
            
            image_collection_dates = self.dates
            for i, date in enumerate(image_collection_dates):
                try:
                    print(f"Processing image {i+1}/{len(image_collection_dates)}: {date}")
                    image_list = self.collection.toList(self.collection.size())
                    image = ee.Image(image_list.get(i))
                    
                    fc_result = process_image(image)
                    df = self.__class__.ee_to_df(fc_result, columns=export_cols, remove_geom=True)
                    
                    if not df.empty:
                        for col in reducer_cols:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            df[col] = df[col].replace(-9999, np.nan)
                        
                        fname = f"{save_folder_path}{date}_transects.csv"
                        df.to_csv(fname, index=False)
                        print(f"Saved: {fname}")
                    else:
                        print(f"Skipping {date}: No data.")
                except Exception as e:
                    print(f"Error processing {date}: {e}")
        else:
            raise ValueError("processing_mode must be 'iterative' or 'aggregated'.")

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

        df = GenericCollection.ee_to_df(stats_fc, remove_geom=True)

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
        file_path=None,
        unweighted=False
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
            unweighted (bool, optional): If True, uses unweighted statistics when applicable (e.g., for 'mean'). Defaults to False.

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
            img_collection_obj = GenericCollection(collection=img_collection_obj.collection.select(band))
        else: 
            # If no band is specified, default to using the first band of the first image in the collection
            first_image = img_collection_obj.image_grab(0)
            first_band = first_image.bandNames().get(0)
            img_collection_obj = GenericCollection(collection=img_collection_obj.collection.select([first_band]))
        
        # If a list of dates is provided, filter the collection to include only images matching those dates
        if dates:
            img_collection_obj = GenericCollection(
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
        
        if unweighted:
            reducer = reducer.unweighted()

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
        df = GenericCollection.ee_to_df(results_fc, remove_geom=True)

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
    
    def multiband_zonal_stats(
            self,
            geometry,
            bands,
            reducer_types,
            scale=30,
            geometry_name='geom',
            dates=None,
            include_area=False,
            file_path=None,
            unweighted=False
        ):
            """
            Calculates zonal statistics for multiple bands over a single geometry for each image in the collection.
            Allows for specifying different reducers for different bands. Optionally includes the geometry area.

            Args:
                geometry (ee.Geometry or ee.Feature): The single geometry to calculate statistics for.
                bands (list of str): A list of band names to include in the analysis.
                reducer_types (str or list of str): A single reducer name (e.g., 'mean') to apply to all bands, 
                    or a list of reducer names matching the length of the 'bands' list to apply specific reducers 
                    to specific bands.
                scale (int, optional): The scale in meters for the reduction. Defaults to 30.
                geometry_name (str, optional): A name for the geometry, used in column naming. Defaults to 'geom'.
                dates (list of str, optional): A list of date strings ('YYYY-MM-DD') to filter the collection. 
                    Defaults to None (processes all images).
                include_area (bool, optional): If True, adds a column with the area of the geometry in square meters. 
                    Defaults to False.
                file_path (str, optional): If provided, saves the resulting DataFrame to a CSV file at this path.
                unweighted (bool, optional): If True, uses unweighted statistics when applicable (e.g., for 'mean'). Defaults to False.

            Returns:
                pd.DataFrame: A pandas DataFrame indexed by Date, with columns named as '{band}_{geometry_name}_{reducer}'.
            """
            # 1. Input Validation and Setup
            if not isinstance(geometry, (ee.Geometry, ee.Feature)):
                raise ValueError("The `geometry` argument must be an ee.Geometry or ee.Feature.")
            
            region = geometry.geometry() if isinstance(geometry, ee.Feature) else geometry

            if isinstance(bands, str):
                bands = [bands]
            if not isinstance(bands, list):
                raise ValueError("The `bands` argument must be a string or a list of strings.")

            # Handle reducer_types (str vs list)
            if isinstance(reducer_types, str):
                reducers_list = [reducer_types] * len(bands)
            elif isinstance(reducer_types, list):
                if len(reducer_types) != len(bands):
                    raise ValueError("If `reducer_types` is a list, it must have the same length as `bands`.")
                reducers_list = reducer_types
            else:
                raise ValueError("`reducer_types` must be a string or a list of strings.")

            # 2. Filter Collection
            processing_col = self.collection

            if dates:
                processing_col = processing_col.filter(ee.Filter.inList('Date_Filter', dates))
            
            processing_col = processing_col.select(bands)

            # 3. Pre-calculate Area (if requested)
            area_val = None
            area_col_name = f"{geometry_name}_area_m2"
            if include_area:
                # Calculate geodesic area in square meters with maxError of 1m
                area_val = region.area(1)

            # 4. Define the Reduction Logic
            def calculate_multiband_stats(image):
                # Base feature with date property
                date_val = image.get('Date_Filter')
                feature = ee.Feature(None, {'Date': date_val})

                # If requested, add the static area value to every feature
                if include_area:
                    feature = feature.set(area_col_name, area_val)

                unique_reducers = list(set(reducers_list))
                
                # OPTIMIZED PATH: Single reducer type for all bands
                if len(unique_reducers) == 1:
                    r_type = unique_reducers[0]
                    try:
                        reducer = getattr(ee.Reducer, r_type)()
                    except AttributeError:
                        reducer = ee.Reducer.mean() 

                    if unweighted:
                        reducer = reducer.unweighted()

                    stats = image.reduceRegion(
                        reducer=reducer,
                        geometry=region,
                        scale=scale,
                        maxPixels=1e13
                    )

                    for band in bands:
                        col_name = f"{band}_{geometry_name}_{r_type}"
                        val = stats.get(band)
                        feature = feature.set(col_name, val)

                # ITERATIVE PATH: Different reducers for different bands
                else:
                    for band, r_type in zip(bands, reducers_list):
                        try:
                            reducer = getattr(ee.Reducer, r_type)()
                        except AttributeError:
                            reducer = ee.Reducer.mean() 

                        if unweighted:
                            reducer = reducer.unweighted()

                        stats = image.select(band).reduceRegion(
                            reducer=reducer,
                            geometry=region,
                            scale=scale,
                            maxPixels=1e13
                        )
                        
                        val = stats.get(band)
                        col_name = f"{band}_{geometry_name}_{r_type}"
                        feature = feature.set(col_name, val)

                return feature

            # 5. Execute Server-Side Mapping (with explicit Cast)
            results_fc = ee.FeatureCollection(processing_col.map(calculate_multiband_stats))

            # 6. Client-Side Conversion
            try:
                df = GenericCollection.ee_to_df(results_fc, remove_geom=True)
            except Exception as e:
                raise RuntimeError(f"Failed to convert Earth Engine results to DataFrame. Error: {e}")

            if df.empty:
                print("Warning: No results returned. Check if the geometry intersects the imagery or if dates are valid.")
                return pd.DataFrame()

            # 7. Formatting & Reordering
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').set_index('Date')

            # Construct the expected column names in the exact order of the input lists
            expected_order = [f"{band}_{geometry_name}_{r_type}" for band, r_type in zip(bands, reducers_list)]
            
            # If area was included, append it to the END of the list
            if include_area:
                expected_order.append(area_col_name)
            
            # Reindex the DataFrame to match this order. 
            existing_cols = [c for c in expected_order if c in df.columns]
            df = df[existing_cols]

            # 8. Export (Optional)
            if file_path:
                if not file_path.lower().endswith('.csv'):
                    file_path += '.csv'
                try:
                    df.to_csv(file_path)
                    print(f"Multiband zonal stats saved to {file_path}")
                except Exception as e:
                    print(f"Error saving file to {file_path}: {e}")

            return df
    
    def sample(
        self,
        locations,
        band=None,
        scale=None,
        location_names=None,
        dates=None,
        file_path=None,
        tileScale=1
    ):
        """
        Extracts time-series pixel values for a list of locations.
        

        Args:
            locations (list, tuple, ee.Geometry, or ee.FeatureCollection): Input points.
            band (str, optional): The name of the band to sample. Defaults to the first band.
            scale (int, optional): Scale in meters. Defaults to 30 if None.
            location_names (list of str, optional): Custom names for locations.
            dates (list, optional): Date filter ['YYYY-MM-DD'].
            file_path (str, optional): CSV export path.
            tileScale (int, optional): Aggregation tile scale. Defaults to 1.

        Returns:
            pd.DataFrame (or CSV if file_path is provided): DataFrame indexed by Date, columns by Location.
        """
        col = self.collection
        if dates:
            col = col.filter(ee.Filter.inList('Date_Filter', dates))

        first_img = col.first()
        available_bands = first_img.bandNames().getInfo()
        
        if band:
            if band not in available_bands:
                raise ValueError(f"Band '{band}' not found. Available: {available_bands}")
            target_band = band
        else:
            target_band = available_bands[0]
            
        processing_col = col.select([target_band])

        def set_name(f):
            name = ee.Algorithms.If(
                f.get('geo_name'), f.get('geo_name'),
                ee.Algorithms.If(f.get('name'), f.get('name'),
                ee.Algorithms.If(f.get('system:index'), f.get('system:index'), 'unnamed'))
            )
            return f.set('geo_name', name)

        if isinstance(locations, (ee.FeatureCollection, ee.Feature)):
            features = ee.FeatureCollection(locations)
        elif isinstance(locations, ee.Geometry):
            lbl = location_names[0] if (location_names and location_names[0]) else 'Point_1'
            features = ee.FeatureCollection([ee.Feature(locations).set('geo_name', lbl)])
        elif isinstance(locations, tuple) and len(locations) == 2:
            lbl = location_names[0] if location_names else 'Location_1'
            features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(locations), {'geo_name': lbl})])
        elif isinstance(locations, list):
            if all(isinstance(i, tuple) for i in locations):
                names = location_names if location_names else [f"Loc_{i+1}" for i in range(len(locations))]
                features = ee.FeatureCollection([
                    ee.Feature(ee.Geometry.Point(p), {'geo_name': str(n)}) for p, n in zip(locations, names)
                ])
            elif all(isinstance(i, ee.Geometry) for i in locations):
                names = location_names if location_names else [f"Geom_{i+1}" for i in range(len(locations))]
                features = ee.FeatureCollection([
                    ee.Feature(g, {'geo_name': str(n)}) for g, n in zip(locations, names)
                ])
            else:
                raise ValueError("List must contain (lon, lat) tuples or ee.Geometry objects.")
        else:
            raise TypeError("Invalid locations input.")

        features = features.map(set_name)


        def sample_image(img):
            date = img.get('Date_Filter')
            use_scale = scale if scale is not None else 30
            

            default_dict = ee.Dictionary({target_band: -9999})

            def extract_point(f):
                stats = img.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=f.geometry(),
                    scale=use_scale,
                    tileScale=tileScale
                )
                
                # Combine dictionaries. 
                # If stats has 'target_band' (even if 0), it overwrites -9999.
                # If stats is empty (masked), -9999 remains.
                safe_stats = default_dict.combine(stats, overwrite=True)
                val = safe_stats.get(target_band)
                
                return f.set({
                    target_band: val,
                    'image_date': date
                })

            return features.map(extract_point)

        # Flatten the results
        flat_results = processing_col.map(sample_image).flatten()

        df = GenericCollection.ee_to_df(
            flat_results, 
            columns=['image_date', 'geo_name', target_band], 
            remove_geom=True
        )

        if df.empty:
            print("Warning: No data returned.")
            return pd.DataFrame()

        # 6. Clean and Pivot
        df[target_band] = pd.to_numeric(df[target_band], errors='coerce')
        
        # Filter out ONLY the sentinel value (-9999), preserving 0.
        df = df[df[target_band] != -9999]

        if df.empty:
             print(f"Warning: All data points were masked (NoData) for band '{target_band}'.")
             return pd.DataFrame()

        pivot_df = df.pivot(index='image_date', columns='geo_name', values=target_band)
        pivot_df.index.name = 'Date'
        pivot_df.columns.name = None
        pivot_df = pivot_df.reset_index()

        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            pivot_df.to_csv(file_path, index=False)
            print(f"Sampled data saved to {file_path}")
            return None

        return pivot_df

    def multiband_sample(
        self,
        location,
        scale=30,
        file_path=None
    ):
        """
        Extracts ALL band values for a SINGLE location across the entire collection.

        Args:
            location (tuple or ee.Geometry): A single (lon, lat) tuple OR ee.Geometry.
            scale (int, optional): Scale in meters. Defaults to 30.
            file_path (str, optional): Path to save CSV.

        Returns:
            pd.DataFrame: DataFrame indexed by Date, with columns for each Band.
        """
        if isinstance(location, tuple) and len(location) == 2:
            geom = ee.Geometry.Point(location)
        elif isinstance(location, ee.Geometry):
            geom = location
        else:
            raise ValueError("Location must be a single (lon, lat) tuple or ee.Geometry.")

        first_img = self.collection.first()
        band_names = first_img.bandNames()
        
        # Create a dictionary of {band_name: -9999}
        # fill missing values so the Feature structure is consistent
        dummy_values = ee.List.repeat(-9999, band_names.length())
        default_dict = ee.Dictionary.fromLists(band_names, dummy_values)

        def get_all_bands(img):
            date = img.get('Date_Filter')
            
            # reduceRegion returns a Dictionary. 
            # If a pixel is masked, that band key is missing from 'stats'.
            stats = img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=geom,
                scale=scale,
                maxPixels=1e13
            )
            
            # Combine stats with defaults. 
            # overwrite=True means real data (stats) overwrites the -9999 defaults.
            complete_stats = default_dict.combine(stats, overwrite=True)
            
            return ee.Feature(None, complete_stats).set('Date', date)

        fc = ee.FeatureCollection(self.collection.map(get_all_bands))
        
        df = GenericCollection.ee_to_df(fc, remove_geom=True)

        if df.empty:
            print("Warning: No data found.")
            return pd.DataFrame()

        # 6. Cleanup
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

        # Replace our sentinel -9999 with proper NaNs
        df = df.replace(-9999, np.nan)

        # 7. Export
        if file_path:
            if not file_path.lower().endswith('.csv'):
                file_path += '.csv'
            df.to_csv(file_path)
            print(f"Multiband sample saved to {file_path}")
            return None

        return df
        
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




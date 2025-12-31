import ee

class ExportToDrive:
    """
    A class to handle exporting Earth Engine images and image collections to Google Drive.

    This class supports exporting both native Earth Engine objects (ee.Image, ee.ImageCollection)
    and RadGEEToolbox objects (e.g., LandsatCollection, Sentinel2Collection). 
    
    It is designed to intelligently handle RadGEEToolbox collections by utilizing their cached 
    `.dates` property for naming files, ensuring readable filenames (e.g., 'MyExport_2023-06-01') 
    instead of long system IDs.

    IMPORTANT: After creating an instance of this class, you must call the `export()` method
    to initiate the export process.

    For example, to export a RadGEEToolbox collection:
    ```python
        collection = LandsatCollection(...)  # Assume this is a RadGEEToolbox object
        exporter = ExportToDrive(
            input_data=collection,
            description="Landsat_Export",
            folder="GEE_Exports",
            scale=30,
            name_pattern="{date}",
        )
        exporter.export()
    ```

    Args:
        input_data (ee.Image, ee.ImageCollection, or RadGEEToolbox object): The data to export.
            Can be a single image, a collection, or a RadGEEToolbox wrapper object.
        description (str): A description of the export task. This serves as the task name in the
            GEE code editor and the default prefix for filenames. Defaults to 'export'.
        folder (str, optional): The name of the destination folder in Google Drive. Defaults to None (root).
        fileNamePrefix (str, optional): The filename prefix. For collections, this is prepended
            to the generated unique name. If None, it defaults to the `description`.
        scale (int): The resolution in meters per pixel. Defaults to 30.
        region (ee.Geometry, optional): The region/geometry to export. Defaults to None (uses image footprint).
        crs (str, optional): The coordinate reference system (e.g., 'EPSG:4326'). Defaults to None (uses image CRS).
        maxPixels (int, optional): The maximum number of pixels allowed in the export. Defaults to 1e13.
        fileFormat (str, optional): The output file format (e.g., 'GeoTIFF', 'TFRecord'). Defaults to 'GeoTIFF'.
        name_pattern (str, optional): A string pattern for naming files when exporting a collection.
            Supported placeholders:
                - {date}: The date of the image. Prioritizes the RadGEEToolbox cached '.dates' list, 
                          then 'Date_Filter' property, then formatted 'system:time_start'.
                - {id}: The system:index of the image.
            Defaults to '{date}'. The final filename will generally be "{fileNamePrefix}_{name_pattern}".
        date_pattern (str, optional): The date format string to use if falling back to 'system:time_start' 
            (e.g., 'YYYY-MM-dd'). Defaults to 'YYYY-MM-dd'.
        **kwargs: Additional keyword arguments passed directly to ee.batch.Export.image.toDrive
            (e.g., 'formatOptions', 'shardSize').

    Raises:
        ValueError: If the input data type is not supported or if required arguments (like scale) are missing/invalid.
    """

    def __init__(
        self,
        input_data,
        description="export",
        folder=None,
        fileNamePrefix=None,
        scale=30,
        region=None,
        crs=None,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        name_pattern="{date}",
        date_pattern="YYYY-MM-dd",
        **kwargs
    ):
        # 1. Capture RadGEEToolbox Metadata BEFORE unwrapping
        # We explicitly look for the cached .dates list which is standard in RadGEEToolbox collections
        self.radgee_dates = getattr(input_data, "dates", None)

        # 2. Input Processing & Conversion
        self.ee_object = self._validate_and_convert_input(input_data)
        
        # Determine strict type of the resulting EE object
        self.is_collection = isinstance(self.ee_object, ee.ImageCollection)
        self.is_image = isinstance(self.ee_object, ee.Image)

        if not self.is_collection and not self.is_image:
            raise ValueError(
                "Processed input must be an ee.Image or ee.ImageCollection. "
                f"Got type: {type(self.ee_object)}"
            )

        # 3. Argument Validation
        if scale is None:
            raise ValueError("The 'scale' argument is required for export.")
        
        if description is None or not isinstance(description, str):
            raise ValueError("The 'description' argument must be a valid string.")

        # Check for extraneous collection arguments when exporting a single image
        if self.is_image:
            if name_pattern != "{date}" or date_pattern != "YYYY-MM-dd":
                print(
                    "Info: Extra arguments 'name_pattern' and/or 'date_pattern' were provided "
                    "but will be ignored because the input is a single image."
                )

        # 4. Store Attributes
        self.description = description
        self.folder = folder
        self.fileNamePrefix = fileNamePrefix if fileNamePrefix else description
        self.scale = scale
        self.region = region
        self.crs = crs
        self.maxPixels = maxPixels
        self.fileFormat = fileFormat
        self.name_pattern = name_pattern
        self.date_pattern = date_pattern
        self.extra_kwargs = kwargs

    def _validate_and_convert_input(self, input_data):
        """
        Helper to unwrap RadGEEToolbox objects or verify EE objects.
        """
        # Check if it's a RadGEEToolbox object (duck typing checks for 'collection' attribute)
        if hasattr(input_data, "collection") and isinstance(input_data.collection, ee.ImageCollection):
            return input_data.collection
        
        # Check if it is already a valid EE object
        if isinstance(input_data, (ee.Image, ee.ImageCollection)):
            return input_data
        
        # Attempt to handle computed objects that are implicitly images
        try:
            return ee.Image(input_data)
        except Exception:
            pass

        raise ValueError(
            "Input data is not a recognized RadGEEToolbox object, ee.Image, or ee.ImageCollection."
        )

    def export(self):
        """
        Initiates the export process. Detects whether to run a single task
        or iterate through a collection.
        """
        if self.is_image:
            self._export_single_image()
        elif self.is_collection:
            self._export_collection()

    def _export_single_image(self):
        """Internal method to export a single image."""
        print(f"Starting export task for single image: {self.description}")
        
        # Construct parameters dict, filtering out None values
        params = {
            "image": self.ee_object,
            "description": self.description,
            "folder": self.folder,
            "fileNamePrefix": self.fileNamePrefix,
            "scale": self.scale,
            "region": self.region,
            "crs": self.crs,
            "maxPixels": self.maxPixels,
            "fileFormat": self.fileFormat,
        }
        # Merge basic params with any extra kwargs provided
        params.update(self.extra_kwargs)
        # Remove keys with None values to allow GEE defaults to take over
        params = {k: v for k, v in params.items() if v is not None}

        task = ee.batch.Export.image.toDrive(**params)
        task.start()
        print(f"Task ID: {task.id}")

    def _export_collection(self):
        """Internal method to iterate and export a collection."""
        # Get the size of the collection locally
        try:
            count = self.ee_object.size().getInfo()
        except Exception as e:
            raise ValueError(f"Could not determine collection size. Error: {e}")

        if count == 0:
            print("The image collection is empty. No export tasks were started.")
            return

        print(f"Processing collection export... ({count} images found)")

        # Convert collection to list for iteration
        col_list = self.ee_object.toList(count)

        for i in range(count):
            img = ee.Image(col_list.get(i))
            
            # Fetch metadata efficiently in one call
            meta = {}
            try:
                meta = img.toDictionary(["Date_Filter", "system:time_start", "system:index"]).getInfo()
            except Exception:
                pass

            # --- Resolve Date ---
            # Priority 1: Use the cached client-side list from RadGEEToolbox if available
            if self.radgee_dates and i < len(self.radgee_dates):
                date_val = self.radgee_dates[i]
            # Priority 2: Check for 'Date_Filter' property (RadGEEToolbox standard)
            elif "Date_Filter" in meta:
                date_val = meta["Date_Filter"]
            # Priority 3: Fallback to 'system:time_start'
            elif "system:time_start" in meta:
                date_val = ee.Date(meta["system:time_start"]).format(self.date_pattern).getInfo()
            else:
                date_val = "no_date"

            # --- Resolve ID ---
            # Use metadata dictionary or fallback to generated index
            sys_index = meta.get("system:index", f"img_{i}")

            # --- Generate Filename ---
            # Apply pattern. Default is "{date}".
            name_str = self.name_pattern.format(id=sys_index, date=date_val)
            
            # Combine with global prefix. 
            # If prefix is "MyExport" and date is "2023-01-01", result is "MyExport_2023-01-01"
            final_filename = f"{self.fileNamePrefix}_{name_str}" if self.fileNamePrefix else name_str
            
            # Ensure unique description for the task (must be unique per task)
            # We use the date in the description to make it easier to track in the GEE Task Manager
            final_desc = f"{self.description}_{date_val}_{i}"

            # Construct parameters
            params = {
                "image": img,
                "description": final_desc,
                "folder": self.folder,
                "fileNamePrefix": final_filename,
                "scale": self.scale,
                "region": self.region,
                "crs": self.crs,
                "maxPixels": self.maxPixels,
                "fileFormat": self.fileFormat,
            }
            params.update(self.extra_kwargs)
            params = {k: v for k, v in params.items() if v is not None}

            task = ee.batch.Export.image.toDrive(**params)
            task.start()
        
        print(f"Successfully queued {count} export tasks.")
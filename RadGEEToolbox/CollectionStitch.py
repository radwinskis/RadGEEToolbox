import ee
import geemap
def CollectionStitch(img_col1, img_col2, copy_properties_from=1):
    """
    Function to mosaic two RadGEETools image collection objects which share image dates. 
    Mosaics are only formed for dates where both image collections have images. Server-side friendly.
    Returned image collection is an eeImageCollection object. NOTE this is different from the CollectionStitch function available in the LandsatCollection and SentinelCollection classes.

    Args:
    img_col2: primary LandsatCollection image collection to be mosaiced with the secondary image collection
    img_col2: secondary LandsatCollection image collection to be mosaiced with the primary image collection
    copy_properties_from: image collection used for copying image properties

    Returns:
    image collection: eeImageCollection image collection with mosaiced imagery and image properties from chosen collection
    """
    image_list = []
    dates_list = img_col1.dates_list + img_col2.dates_list
    dates_list = sorted(list(set(dates_list)))  # Get unique sorted list of dates

    for date in dates_list:
        if date in img_col1.dates_list and date in img_col2.dates_list:
            filtered_col1 = img_col1.image_grab(img_col1.dates_list.index(date))
            filtered_col2 = img_col2.image_grab(img_col2.dates_list.index(date))
            merged_col = ee.ImageCollection.fromImages([filtered_col1, filtered_col2])
            if copy_properties_from == 1:
                mosaic = merged_col.mosaic().copyProperties(filtered_col1)  # new collection images contain all image properties of the northern landsat image
            elif copy_properties_from == 2:
                mosaic = merged_col.mosaic().copyProperties(filtered_col2)  # new collection images contain all image properties of the southern landsat image
            else:
                raise ValueError("Invalid value for 'copy_properties_from'. Must be 1 or 2.")  # new collection images contain all image properties of the northern landsat image
            image_list.append(mosaic)
        else:
            None  # If the condition isn't met, do nothing and keep going through the list
    new_col = ee.ImageCollection.fromImages(image_list)
    return new_col

def MosaicByDate(img_col):
    """
    Function to mosaic collection images that share the same date. Server-side friendly. Requires images to have date property of "Date_Filter"

    Args:
    img_col: eeImageCollection object

    Returns:
    image collection: eeImageCollection with mosaiced imagery
    """
    input_collection = img_col
# Function to mosaic images of the same date and accumulate them
    def mosaic_and_accumulate(date, list_accumulator):
        # date = ee.Date(date)
        list_accumulator = ee.List(list_accumulator)
        date_filter = ee.Filter.eq('Date_Filter', date)
        date_collection = input_collection.filter(date_filter)
        
        # Create mosaic
        mosaic = date_collection.mosaic().set('Date_Filter', date)

        return list_accumulator.add(mosaic)

    # Get distinct dates
    distinct_dates = input_collection.aggregate_array('Date_Filter').distinct()

    # Initialize an empty list as the accumulator
    initial = ee.List([])

    # Iterate over each date to create mosaics and accumulate them in a list
    mosaic_list = distinct_dates.iterate(mosaic_and_accumulate, initial)

    new_col = ee.ImageCollection.fromImages(mosaic_list)

    # Convert the list of mosaics to an ImageCollection
    return new_col
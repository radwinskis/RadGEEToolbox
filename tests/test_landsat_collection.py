from RadGEEToolbox import LandsatCollection
import ee

def test_landsat_collection():

    col = LandsatCollection(start_date='2023-06-01', end_date='2023-06-30', tile_row=32, tile_path=38)
    dates = col.dates
    assert isinstance(dates, list), "Expected dates to be a list - issue in defining base collection"

    col_scaled = LandsatCollection(start_date='2023-06-01', end_date='2023-06-30', tile_row=32, tile_path=38, scale_bands=True)
    scaled_dates = col_scaled.dates
    assert isinstance(scaled_dates, list), "Expected scaled_dates to be a list - issue in defining reflectance scaled collection"

    water_area = LandsatCollection.pixelAreaSum(image=col.ndwi.image_grab(-1), band_name='ndwi', geometry=col.ndwi.image_grab(-1).geometry(), threshold=0, scale=90)
    assert water_area.getInfo().get('properties')['ndwi'] is not None, "Expected water_area to have a non-None value - issue in calculating pixel area sum for NDWI"

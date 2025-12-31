from RadGEEToolbox import Sentinel2Collection
import ee

def test_sentinel2_collection():

    col = Sentinel2Collection(start_date='2023-06-01', end_date='2023-06-30', tile='12TUL')
    dates = col.dates
    assert isinstance(dates, list), "Expected dates to be a list - issue in defining base collection"

    col_scaled = Sentinel2Collection(start_date='2023-06-01', end_date='2023-06-30', tile='12TUL', scale_bands=True)
    scaled_dates = col_scaled.dates
    assert isinstance(scaled_dates, list), "Expected scaled_dates to be a list - issue in defining reflectance scaled collection"

    water_area = Sentinel2Collection.pixelAreaSum(image=col.ndwi.image_grab(-1), band_name='ndwi', geometry=col.ndwi.image_grab(-1).geometry(), threshold=0, scale=50)
    assert water_area.getInfo().get('properties')['ndwi'] is not None, "Expected water_area to have a non-None value - issue in calculating pixel area sum for NDWI"
import ee
from RadGEEToolbox import Sentinel1Collection

def test_sentinel1_collection():
    counties = ee.FeatureCollection('TIGER/2018/Counties')
    salt_lake_county = counties.filter(ee.Filter.And(
        ee.Filter.eq('NAME', 'Salt Lake'),
        ee.Filter.eq('STATEFP', '49')))
    salt_lake_geometry = salt_lake_county.geometry()

    SAR_collection = Sentinel1Collection(
        start_date='2024-05-01',
        end_date='2024-05-31',
        instrument_mode='IW',
        polarization=['VV', 'VH'],
        orbit_direction='DESCENDING',
        boundary=salt_lake_geometry,
        resolution_meters=10
    )

    SAR_collection = SAR_collection.mask_to_polygon(salt_lake_geometry)

    dates = SAR_collection.dates
    assert isinstance(dates, list), "Expected dates to be a list - issue in defining base collection or masking"

    SAR_collection_sigma0 = SAR_collection.sigma0FromDb
    SAR_collection_multilooked = SAR_collection_sigma0.multilook(looks=4)
    SAR_collection_multilooked_and_filtered = SAR_collection_multilooked.speckle_filter(5, geometry=salt_lake_geometry, looks=4)
    multilooked_dates = SAR_collection_multilooked_and_filtered.dates
    assert isinstance(multilooked_dates, list), "Expected multilooked_dates to be a list - issue in defining multilooked and filtered collection"

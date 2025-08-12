import ee
from .GetPalette import get_palette


def get_visualization_params(
    satellite, index, min_val=None, max_val=None, palette=None
):
    """
    Function to define visualization paramaters for image visualization. Outputs an vis_params dictionary.

    Args:
        satellite (string): Denote satellite for visualization. Options: 'Landsat', 'landsat', 'Sentinel2', or 'sentinel2'
        index (string): Multispectral index to visualize. Options: 'TrueColor', 'NDVI', 'NDWI', 'halite', 'gypsum', 'LST', 'NDTI', 'KIVU', or '2BDA'. NOTE: LST and KIVU is not available for Sentinel2, and 2BDA is not available for Landsat.
        min_val (int or float): Optional override for minimum value to stretch raster
        max_val (int or float): Optional override for maximum value to stretch raster
        palette (string): Optional override for color palette used for image visualization. Must be from GetPalette.get_palette() function. Options: 'algae', 'dense', 'greens', 'haline', 'inferno', 'jet', 'matter', 'pubu', 'soft_blue_green_red', 'thermal', 'turbid', 'ylord'

    Returns:
        dictionary: vis_params dictionary to be used when visualizing an image on a map. Supplies min, max, band(s), and color palette (when appropriate).
    """
    # Define visualization parameters for various image types
    if satellite == "Landsat" or satellite == "landsat":
        params_dict = {
            "TrueColor": {"min": 0, "max": 30000, "bands": ["SR_B4", "SR_B3", "SR_B2"]},
            "NDVI": {
                "min": 0,
                "max": 0.5,
                "bands": ["ndvi"],
                "palette": get_palette("greens"),
            },
            "NDWI": {
                "min": -0.2,
                "max": 0.2,
                "bands": ["ndwi"],
                "palette": get_palette("inferno"),
            },
            "halite": {
                "min": 0.1,
                "max": 0.5,
                "bands": ["halite"],
                "palette": get_palette("haline"),
            },
            "gypsum": {
                "min": 0.0,
                "max": 0.5,
                "bands": ["gypsum"],
                "palette": get_palette("ylord"),
            },
            "LST": {
                "min": 0,
                "max": 40,
                "bands": ["LST"],
                "palette": get_palette("thermal"),
            },
            "NDTI": {
                "min": -0.2,
                "max": 0.2,
                "bands": ["ndti"],
                "palette": get_palette("turbid"),
            },
            "KIVU": {
                "min": -0.5,
                "max": 0.2,
                "bands": ["kivu"],
                "palette": get_palette("algae"),
            },
        }
    elif satellite == "Sentinel2" or satellite == "sentinel2":
        params_dict = {
            "TrueColor": {"min": 0, "max": 3500, "bands": ["B4", "B3", "B2"]},
            "NDVI": {
                "min": 0.5,
                "max": 0.9,
                "bands": ["ndvi"],
                "palette": get_palette("greens"),
            },
            "NDWI": {
                "min": -0.2,
                "max": 0.2,
                "bands": ["ndwi"],
                "palette": get_palette("inferno"),
            },
            "halite": {
                "min": 0.1,
                "max": 0.7,
                "bands": ["halite"],
                "palette": get_palette("haline"),
            },
            "gypsum": {
                "min": 0.0,
                "max": 0.7,
                "bands": ["gypsum"],
                "palette": get_palette("ylord"),
            },
            "NDTI": {
                "min": -0.2,
                "max": 0.5,
                "bands": ["ndti"],
                "palette": get_palette("turbid"),
            },
            "2BDA": {
                "min": 0.5,
                "max": 1.75,
                "bands": ["2BDA"],
                "palette": get_palette("algae"),
            },
        }
    else:
        raise ValueError(
            "Incorrect definition of satellite. Options: 'Landsat', 'landsat', 'Sentinel2', or 'sentinel2'"
        )

    # Get default visualization parameters for the given image name
    params = params_dict.get(index, None)
    if params is None:
        raise ValueError(
            "Incorrect definition of index. Options: 'TrueColor', 'NDVI', 'NDWI', 'halite', 'gypsum', 'LST', 'NDTI', or 'KIVU'. Note LST is not available for Sentinel2"
        )

    # Override default values if provided by the user
    if min_val is not None:
        params["min"] = min_val
    if max_val is not None:
        params["max"] = max_val
    if palette is not None:
        # Check if palette is a string and get the palette list if it is
        if isinstance(palette, str):
            palette = get_palette(palette)
            if palette is None:
                raise ValueError(f"Palette {palette} not found.")
        params["palette"] = palette

    return params

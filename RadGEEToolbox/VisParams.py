import ee
from .GetPalette import get_palette

def get_visualization_params(
    satellite, index, min_val=None, max_val=None, palette=None, scaled_bands=False
):
    """
    Function to define visualization parameters for image visualization. Outputs a vis_params dictionary.

    Args:
        satellite (str): Denote satellite/source for visualization.
                         Options: 'Landsat', 'Sentinel2', 'Sentinel1', 'Generic' (for DEMs, Precip, etc.).
                         Case-insensitive.
        index (str): Multispectral index or band to visualize.
                     Options include: 'TrueColor', 'NDVI', 'NDWI', 'MNDWI', 'EVI', 'SAVI', 'MSAVI', 'NDMI',
                     'NBR', 'NDSI', 'halite', 'gypsum', 'LST', 'NDTI', 'KIVU', '2BDA', 'Elevation',
                     'Precipitation', 'Evapotranspiration'.
        min_val (int or float): Optional override for minimum value to stretch raster.
        max_val (int or float): Optional override for maximum value to stretch raster.
        palette (str or list): Optional override for color palette. Can be a list of hex codes or
                               a string name from GetPalette (e.g., 'ndvi', 'viridis', 'magma').
        scaled_bands (bool): If True, indicates that surface reflectance bands are scaled (0-1) rather than raw DNs. Defaults to False.

    Returns:
        dict: vis_params dictionary to be used when visualizing an image on a map. Supplies min, max, bands, and palette.
    """
    # Normalize input
    satellite_lower = satellite.lower() if satellite else "generic"

    # --- Landsat ---
    if "landsat" in satellite_lower:
        tc_min = 0
        tc_max = 1 if scaled_bands else 30000
        
        params_dict = {
            "TrueColor": {"min": tc_min, "max": tc_max, "bands": ["SR_B4", "SR_B3", "SR_B2"]},
            "NDVI": {"min": 0, "max": 1, "bands": ["ndvi"], "palette": get_palette("ndvi")},
            "NDWI": {"min": -0.5, "max": 0.5, "bands": ["ndwi"], "palette": get_palette("ndwi")},
            "MNDWI": {"min": -0.5, "max": 0.5, "bands": ["mndwi"], "palette": get_palette("ndwi")},
            "EVI": {"min": 0, "max": 1, "bands": ["evi"], "palette": get_palette("ndvi")},
            "SAVI": {"min": 0, "max": 1, "bands": ["savi"], "palette": get_palette("ndvi")},
            "MSAVI": {"min": 0, "max": 1, "bands": ["msavi"], "palette": get_palette("ndvi")},
            "NDMI": {"min": -0.8, "max": 0.8, "bands": ["ndmi"], "palette": get_palette("ocean")},
            "NBR": {"min": -0.5, "max": 1, "bands": ["nbr"], "palette": get_palette("magma")},
            "NDSI": {"min": -0.5, "max": 1, "bands": ["ndsi"], "palette": get_palette("blues")},
            "halite": {"min": 0.1, "max": 0.5, "bands": ["halite"], "palette": get_palette("haline")},
            "gypsum": {"min": 0.0, "max": 0.5, "bands": ["gypsum"], "palette": get_palette("ylord")},
            "LST": {"min": 0, "max": 40, "bands": ["LST"], "palette": get_palette("thermal")},
            "NDTI": {"min": -0.2, "max": 0.2, "bands": ["ndti"], "palette": get_palette("turbid")},
            "KIVU": {"min": -0.5, "max": 0.2, "bands": ["kivu"], "palette": get_palette("algae")},
        }

    # --- Sentinel-2 ---
    elif "sentinel2" in satellite_lower:
        tc_min = 0
        tc_max = 1 if scaled_bands else 3500

        params_dict = {
            "TrueColor": {"min": tc_min, "max": tc_max, "bands": ["B4", "B3", "B2"]},
            "NDVI": {"min": 0, "max": 1, "bands": ["ndvi"], "palette": get_palette("ndvi")},
            "NDWI": {"min": -0.5, "max": 0.5, "bands": ["ndwi"], "palette": get_palette("ndwi")},
            "MNDWI": {"min": -0.5, "max": 0.5, "bands": ["mndwi"], "palette": get_palette("ndwi")},
            "EVI": {"min": 0, "max": 1, "bands": ["evi"], "palette": get_palette("ndvi")},
            "SAVI": {"min": 0, "max": 1, "bands": ["savi"], "palette": get_palette("ndvi")},
            "MSAVI": {"min": 0, "max": 1, "bands": ["msavi"], "palette": get_palette("ndvi")},
            "NDMI": {"min": -0.8, "max": 0.8, "bands": ["ndmi"], "palette": get_palette("ocean")},
            "NBR": {"min": -0.5, "max": 1, "bands": ["nbr"], "palette": get_palette("magma")},
            "NDSI": {"min": -0.5, "max": 1, "bands": ["ndsi"], "palette": get_palette("blues")},
            "halite": {"min": 0.1, "max": 0.7, "bands": ["halite"], "palette": get_palette("haline")},
            "gypsum": {"min": 0.0, "max": 0.7, "bands": ["gypsum"], "palette": get_palette("ylord")},
            "NDTI": {"min": -0.2, "max": 0.5, "bands": ["ndti"], "palette": get_palette("turbid")},
            "2BDA": {"min": 0, "max": 2, "bands": ["2BDA"], "palette": get_palette("algae")},
        }

    # --- Sentinel-1 ---
    elif "sentinel1" in satellite_lower:
        params_dict = {
            "VV": {"min": -25, "max": 0, "bands": ["VV"]},
            "VH": {"min": -30, "max": -5, "bands": ["VH"]},
            "HH": {"min": -25, "max": 0, "bands": ["HH"]},
            "HV": {"min": -30, "max": -5, "bands": ["HV"]},
            "RGB": {"min": [-20, -25, 1], "max": [0, -5, 15], "bands": ["VV", "VH", "VV/VH"]},
        }

    # --- Generic (DEM, Precipitation, ET, etc.) ---
    elif "generic" in satellite_lower or satellite is None:
        params_dict = {
            "Elevation": {"min": 0, "max": 3000, "bands": ["elevation"], "palette": get_palette("dem")},
            "DEM": {"min": 0, "max": 3000, "bands": ["elevation"], "palette": get_palette("dem")},
            "Terrain": {"min": 0, "max": 3000, "bands": ["elevation"], "palette": get_palette("terrain")},
            "Precipitation": {"min": 0, "max": 2000, "bands": ["precipitation"], "palette": get_palette("precipitation")},
            "Evapotranspiration": {"min": 0, "max": 1000, "bands": ["et"], "palette": get_palette("evapotranspiration")},
            "ET": {"min": 0, "max": 1000, "bands": ["et"], "palette": get_palette("evapotranspiration")},
            "NDVI": {"min": 0, "max": 1, "bands": ["ndvi"], "palette": get_palette("ndvi")},
            "NDWI": {"min": -0.5, "max": 0.5, "bands": ["ndwi"], "palette": get_palette("ndwi")},
            "Temperature": {"min": 270, "max": 310, "bands": ["temperature"], "palette": get_palette("thermal")},
        }

    else:
        raise ValueError(
            f"Incorrect definition of satellite '{satellite}'. Options: 'Landsat', 'Sentinel2', 'Sentinel1', or 'Generic'."
        )

    # --- Retrieval & Override Logic ---
    
    # Get default params for the requested index
    params = params_dict.get(index, None)
    if params is None:
        # Fallback: If index isn't a known key, create a basic config assuming 'index' is the band name
        # This makes the function flexible for unknown bands
        params = {"bands": [index]}
        # Default min/max/palette if creating a generic fallback (user overrides encouraged)
        if min_val is None: params["min"] = 0
        if max_val is None: params["max"] = 1
        
        print(f"Warning: Index '{index}' not explicitly defined for '{satellite}'. Using '{index}' as band name with default 0-1 stretch.")

    # Override values if provided
    if min_val is not None:
        params["min"] = min_val
    if max_val is not None:
        params["max"] = max_val
    
    if palette is not None:
        # If palette is a string name, look it up; otherwise assume it's a list of colors
        if isinstance(palette, str):
            fetched_palette = get_palette(palette)
            if fetched_palette is not None:
                params["palette"] = fetched_palette
            else:
                # Assume user provided a custom string that isn't in our dictionary? 
                # Actually, GEE usually expects a list. If it's a string not in our dict, 
                # it might be a single hex code or error. 
                # We'll assume it's a custom palette name we don't know, or just pass it through if valid.
                pass
        else:
            params["palette"] = palette

    return params

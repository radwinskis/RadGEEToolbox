def get_palette(name):
    """
    Returns the color palette associated with the given name.

    Args:
        name (str): Options include:
            - Scientific: 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'coolwarm', 'spectral'
            - Sequential: 'blues', 'greens', 'reds', 'greys', 'oranges', 'purples'
            - Diverging: 'rdylgn' (Red-Yellow-Green), 'rdylbu' (Red-Yellow-Blue), 'rdbu' (Red-White-Blue), 'brbg' (Brown-Blue-Green), 'piyg' (Pink-Yellow-Green)
            - Domain Specific: 'dem' (Elevation), 'terrain' (Topography), 'ndvi' (Vegetation), 'ndwi' (Water), 'precipitation' (Rain/Snow), 'thermal' (Temperature), 'evapotranspiration' (ET)
            - Custom/Legacy: 'algae', 'dense', 'haline', 'jet', 'matter', 'pubu', 'soft_blue_green_red', 'turbid', 'ylord', 'ocean'

    Returns:
        list: list of colors to be used for image visualization in GEE vis params

    """
    palettes = {
        # --- Scientific / Perceptually Uniform ---
        "viridis": [
            "#440154", "#482475", "#414487", "#355f8d", "#2a788e", 
            "#21918c", "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725"
        ],
        "plasma": [
            "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", 
            "#d8576b", "#ed7953", "#fb9f3a", "#fdc924", "#f0f921"
        ],
        "inferno": [
            "#000004", "#160b39", "#420a68", "#6a176e", "#932667", 
            "#bc3754", "#dd513a", "#f37819", "#fca50a", "#f6d746", "#fcffa4"
        ],
        "magma": [
            "#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", 
            "#b73779", "#de4968", "#f7705c", "#fe9f6d", "#fecf92", "#fcfdbf"
        ],
        "cividis": [
            "#00204d", "#002c69", "#003989", "#184a8c", "#3f5b8a", 
            "#5d6d85", "#78807f", "#969576", "#b4ab6a", "#d4c359", "#fdea45"
        ],
        "turbo": [
            "#30123b", "#466be3", "#28bbec", "#32f298", "#a2fc3c", 
            "#f2ea33", "#fe9b2d", "#e4460a", "#7a0403"
        ],
        "coolwarm": [
            "#3d4c8a", "#6282ea", "#99baff", "#cdd9ec", "#eaf0e4", 
            "#f4dcb8", "#e8ac80", "#d4654d", "#b2182b"
        ],

        # --- Sequential (ColorBrewer & Standard) ---
        "blues": [
            "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", 
            "#4292c6", "#2171b5", "#08519c", "#08306b"
        ],
        "greens": [
            "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476", 
            "#41ab5d", "#238b45", "#006d2c", "#00441b"
        ],
        "reds": [
            "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", 
            "#ef3b2c", "#cb181d", "#a50f15", "#67000d"
        ],
        "greys": [
            "#ffffff", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", 
            "#737373", "#525252", "#252525", "#000000"
        ],
        "oranges": [
            "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c", 
            "#f16913", "#d94801", "#a63603", "#7f2704"
        ],
        "purples": [
            "#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8", 
            "#807dba", "#6a51a3", "#54278f", "#3f007d"
        ],

        # --- Diverging ---
        "spectral": [
            "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b", 
            "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2"
        ],
        "rdylgn": [
            "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", 
            "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"
        ],
        "rdylbu": [
            "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", 
            "#ffffbf", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"
        ],
        "rdbu": [
            "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7", 
            "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061"
        ],
        "brbg": [
            "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", 
            "#f5f5f5", "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30"
        ],
        "piyg": [
            "#8e0152", "#c51b7d", "#de77ae", "#f1b6da", "#fde0ef", 
            "#f7f7f7", "#e6f5d0", "#b8e186", "#7fbc41", "#4d9221", "#276419"
        ],

        # --- Domain Specific ---
        "dem": [
            "#006600", "#002200", "#fff700", "#ab7634", "#c4d0ff", "#ffffff"
        ],  # Classic Green-Brown-White Elevation
        "terrain": [
            "#00A600", "#63C600", "#E6E600", "#E9BD3A", "#ECB176", 
            "#EFC2B3", "#F2F2F2"
        ],  # Alternative Terrain
        "ndvi": [
            "#FFFFFF", "#CE7E45", "#DF923D", "#F1B555", "#FCD163", "#99B718",
            "#74A901", "#66A000", "#529400", "#3E8601", "#207401", "#056201",
            "#004C00", "#023B01", "#012E01", "#011D01", "#011301"
        ],  # Standard MODIS/Landsat NDVI ramp
        "ndwi": [
            "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", 
            "#0570b0", "#045a8d", "#023858"
        ],  # Blue ramp for water
        "precipitation": [
            "#ffffff", "#00ffff", "#0000ff", "#00ff00", "#ffff00", 
            "#ff0000", "#ff00ff"
        ],  # Classic Precip: White-Blue-Green-Yellow-Red-Purple
        "evapotranspiration": [
            "#ffffff", "#fcd163", "#99b718", "#74a901", "#66a000", 
            "#529400", "#3e8601", "#207401", "#056201", "#004c00"
        ],  # Modeled on NDVI/Vegetation water use
        "thermal": [
            "#042333", "#2c3395", "#744992", "#b15f82", "#eb7958", 
            "#fbb43d", "#e8fa5b"
        ],

        # --- Custom / Legacy from Original ---
        "jet": [
            "#00007F", "#002AFF", "#00D4FF", "#7FFF7F", "#FFD400", 
            "#FF2A00", "#7F0000"
        ],
        "soft_blue_green_red": ["#deeaee", "#b1cbbb", "#eea29a", "#c94c4c"],
        "algae": [
            "#d7f9d0", "#a2d595", "#64b463", "#129450", "#126e45", 
            "#1a482f", "#122414"
        ],
        "turbid": [
            "#e9f6ab", "#d3c671", "#bf9747", "#a1703b", "#795338", 
            "#4d392d", "#221f1b"
        ],
        "dense": [
            "#e6f1f1", "#a2cee2", "#76a4e5", "#7871d5", "#7642a5", 
            "#621d62", "#360e24"
        ],
        "matter": [
            "#feedb0", "#f7b37c", "#eb7858", "#ce4356", "#9f2462", 
            "#66185c", "#2f0f3e"
        ],
        "haline": [
            "#2a186c", "14439c", "#206e8b", "#3c9387", "#5ab978", 
            "#aad85c", "#fdef9a"
        ],
        "ylord": [
            "#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", 
            "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
        ],
        "pubu": [
            "#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", 
            "#3690c0", "#0570b0", "#045a8d", "#023858"
        ][::-1],
        "ocean": [
            "#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4",
            "#1d91c0", "#225ea8", "#253494", "#081d58"
        ],
    }
    return palettes.get(name, None)

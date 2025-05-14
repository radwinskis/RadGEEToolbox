def get_palette(name):
    """
    Returns the color palette associated with the given name.

    Args:
        name (str): options are 'algae', 'dense', 'greens', 'haline', 'inferno', 'jet', 'matter', 'pubu', 'soft_blue_green_red', 'thermal', 'turbid', 'ylord'
    
    Returns:
        list: list of colors to be used for image visualization in GEE vis params
    
    """
    palettes = {
        'jet': ['#00007F', '#002AFF', '#00D4FF', '#7FFF7F', '#FFD400', '#FF2A00', '#7F0000'],
        'soft_blue_green_red': ['#deeaee', '#b1cbbb', '#eea29a', '#c94c4c'],
        'inferno': ['#000004', '#320A5A', '#781B6C', '#BB3654', '#EC6824', '#FBB41A', '#FCFFA4'],
        'thermal': ['#042333', '#2c3395', '#744992', '#b15f82', '#eb7958', '#fbb43d', '#e8fa5b'],
        'algae': ['#d7f9d0', '#a2d595', '#64b463', '#129450', '#126e45', '#1a482f', '#122414'],
        'turbid': ['#e9f6ab', '#d3c671', '#bf9747', '#a1703b', '#795338', '#4d392d', '#221f1b'],
        'dense': ['#e6f1f1', '#a2cee2', '#76a4e5', '#7871d5', '#7642a5', '#621d62', '#360e24'],
        'matter': ['#feedb0', '#f7b37c', '#eb7858', '#ce4356', '#9f2462', '#66185c', '#2f0f3e'],
        'haline': ['#2a186c', '14439c', '#206e8b', '#3c9387', '#5ab978', '#aad85c', '#fdef9a'],
        'ylord': ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'],
        'pubu': ['#fff7fb', '#ece7f2', '#d0d1e6', '#a6bddb', '#74a9cf', '#3690c0', '#0570b0', '#045a8d', '#023858'][::-1],
        'greens': ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    }
    return palettes.get(name, None)
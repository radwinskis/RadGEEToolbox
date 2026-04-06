import io
import math
import urllib.request
import numpy as np
import ee
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image

class Visualize:
    """
    Single image or multiple image visualization for the creation of publication-grade map figures. Built as a pure Cartopy/Matplotlib wrapper for Earth Engine mapping. 
    Inspired by the need for a more intuitive and flexible alternative to cartoee, with a focus on perfect aspect ratios, native Cartopy features, 
    and seamless handling of both single images and image collections. Credit to cartoee and cartopy packages for foundational templates.
    
    Args:
        images (ee.Image, ee.ImageCollection, or list of ee.Images): The Earth Engine image(s) to visualize.
        region (ee.Geometry): The geographic region to focus on.
        vis_params (dict): Visualization parameters for the base images (e.g., bands, min/max).
        ncols (int): Number of columns in the grid layout.
        base_width (float): Base width of the figure in inches; height is calculated to maintain aspect ratio.
        subplot_spacing (float): Spacing between subplots as a fraction of the figure size.
        cmap (str): Matplotlib colormap to use for single-band images without a palette.
        dpi (int): Resolution of the output figure in dots per inch.
    """
    def __init__(self, images, region, vis_params, ncols=1, base_width=12, 
                 subplot_spacing=0.05, cmap='inferno', dpi=300, max_pixels=None):
        
        self.region = region
        self.vis_params = vis_params.copy()
        self.cmap = cmap
        self.subplot_spacing = subplot_spacing
        self.dpi = dpi
        # self.dimensions = 1024 # Resolution of the requested EE thumbnails
        self.max_pixels_override = max_pixels
        
        if isinstance(images, ee.Image):
            self.images = [images]
        elif isinstance(images, ee.ImageCollection):
            count = images.size().getInfo()
            self.images = [ee.Image(images.toList(count).get(i)) for i in range(count)]
        elif isinstance(images, list) and all(isinstance(img, ee.Image) for img in images):
            self.images = images
        else:
            raise ValueError("Input must be an ee.Image, ee.ImageCollection, or list of ee.Images.")
            
        self.n_images = len(self.images)
        self.ncols = min(ncols, self.n_images)
        self.nrows = math.ceil(self.n_images / self.ncols)
        self.band_count = len(self.vis_params.get('bands', []))
        self.is_rgb = (self.band_count == 3)
        
        # 2. Geometry & Perfect Figsize Math
        self._calculate_geometry(base_width)
        
        # 3. Initialize Figure & Projection
        self.proj = ccrs.PlateCarree()
        self.fig, self.axs = plt.subplots(
            nrows=self.nrows, ncols=self.ncols, 
            figsize=self.figsize, dpi=self.dpi,
            subplot_kw={'projection': self.proj}
        )
        self.fig.subplots_adjust(wspace=self.subplot_spacing, hspace=self.subplot_spacing)
        
        # Safely flatten axes and store active ones
        all_axs = self.axs.flatten() if hasattr(self.axs, 'flatten') else [self.axs]
        self.active_axs = []
        
        # 4. Plot Base Images
        for i, ax in enumerate(all_axs):
            if i >= self.n_images:
                ax.set_visible(False) 
                continue
                
            self.active_axs.append(ax)

            has_palette = 'palette' in self.vis_params
            apply_cmap = not (self.is_rgb or has_palette)
            
            # Fetch and draw the base layer
            self._fetch_and_draw(ax, self.images[i], self.vis_params, apply_cmap=apply_cmap)
            
            ax.set_aspect(self.aspect_ratio)
            self.xlim = ax.get_xlim()
            self.ylim = ax.get_ylim()

    def _calculate_geometry(self, base_width):
        """Calculates the perfect figure dimensions and Cartesian extents."""
        coords = self.region.bounds().coordinates().getInfo()[0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]

        self.lon_min, self.lon_max = min(lons), max(lons) 
        self.lat_min, self.lat_max = min(lats), max(lats) 
        
        # The true extent Cartopy uses: [West, East, South, North]
        self.cartopy_extent = [self.lon_min, self.lon_max, self.lat_min, self.lat_max]
        
        # Aspect Ratio based on latitude
        center_lat = (self.lat_min + self.lat_max) / 2
        self.aspect_ratio = 1 / math.cos(math.radians(center_lat))
        
        # Aspect-aware figure sizing
        data_width = self.lon_max - self.lon_min
        data_height = self.lat_max - self.lat_min
        single_map_aspect = (data_height / data_width) * self.aspect_ratio
        
        total_height_ratio = self.nrows * single_map_aspect + (self.nrows - 1) * self.subplot_spacing
        total_width_ratio = self.ncols + (self.ncols - 1) * self.subplot_spacing
        
        self.figsize = (base_width, base_width * (total_height_ratio / total_width_ratio))
        if self.max_pixels_override:
            self.dimensions = self.max_pixels_override
        else:
            # Calculate physical width of a single subplot in inches
            single_plot_inches = base_width / self.ncols
            
            # Calculate required pixels, multiply by 1.2 for a sharpness buffer
            ideal_pixels = int(single_plot_inches * self.dpi * 1.2)
            
            # Clamp the value between a safe minimum (1024) and EE's maximum limit (10000)
            self.dimensions = max(1024, min(ideal_pixels, 10000))
            # self.dimensions = max(1024, min(ideal_pixels, 3400))

    def _lock_extents(self):
        """Forces axes to maintain their perfectly wrapped bounding boxes."""
        for ax in self.active_axs:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)

    def _fetch_and_draw(self, ax, ee_object, vis_params, apply_cmap=False, req_dim=None):
        """The native engine that replaces cartoee.add_layer."""

        fetch_dim = req_dim if req_dim else self.dimensions
        # Setup Earth Engine thumbnail parameters
        thumb_params = {
            'region': self.region,
            'dimensions': fetch_dim,
            'crs': 'EPSG:4326',
            'format': 'png'
        }
        if vis_params:
            thumb_params.update(vis_params)
            
        # Request and read the image directly into memory
        url = ee_object.getThumbURL(thumb_params)
        response = urllib.request.urlopen(url)
        img_data = np.array(Image.open(io.BytesIO(response.read())))
        
        if img_data.ndim == 3:
            channels = img_data.shape[2]
            if apply_cmap:
                # Matplotlib colormaps require a flat 2D array. 
                data = img_data[:, :, 0] # Extract the base data
                
                # If EE sent an alpha channel (mask), apply it natively as a NumPy mask
                if channels in [2, 4]: 
                    alpha = img_data[:, :, -1]
                    img_data = np.ma.masked_where(alpha == 0, data)
                else:
                    img_data = data
        
        # Plot to Cartopy
        cmap_to_use = self.cmap if apply_cmap else None
        ax.imshow(
            img_data, 
            extent=self.cartopy_extent, 
            origin='upper', 
            transform=self.proj,
            cmap=cmap_to_use
        )
        ax.set_aspect(self.aspect_ratio)

    def add_layer(self, ee_object, vis_params=None, color=None, width=1, is_vector=False):
            """
            Stacks an ee.Image or ee.FeatureCollection onto the grid.
            """
            v_params = vis_params.copy() if vis_params else {}
            apply_cmap = False
            
            fetch_dim = self.dimensions

            is_ee_vector = isinstance(ee_object, (ee.FeatureCollection, ee.Feature, ee.Geometry))
            
            if is_ee_vector or is_vector:
                vector_fc = ee.FeatureCollection(ee_object)
                # Use Earth Engine's native, highly optimized styling engine for vectors
                style_dict = {'color': color if color else 'black', 'width': width, 'fillColor': '00000000'}
                ee_object = vector_fc.style(**style_dict)
                
                # Since .style() outputs a pre-rendered RGB image, we clear the v_params payload
                v_params = {} 
                fetch_dim = min(self.dimensions, 2500)
                
            elif 'bands' in v_params and len(v_params['bands']) == 1 and 'palette' not in v_params:
                apply_cmap = True 

            for ax in self.active_axs:
                self._fetch_and_draw(ax, ee_object, v_params, apply_cmap=apply_cmap, req_dim=fetch_dim)
                
            self._lock_extents()
            return self

    def add_scalebar(self, length_km=10, xy=(0.05, 0.05), color='black', linewidth=3, fontsize=10, apply_to_all=False):
        """
        Adds a native, mathematically accurate scale bar.
        
        Args:
            length_km (float): The real-world length of the scale bar in kilometers.
            xy (tuple): Fractional coordinates (x, y) for the starting point of the scale bar on the map.
            color (str): Color of the scale bar and text.
            linewidth (int): Thickness of the scale bar line.
            fontsize (int): Font size for the scale bar label.
            apply_to_all (bool): Whether to apply the scale bar to all active subplots.
        
        """
        target_axs = self.active_axs if apply_to_all else [self.active_axs[0]]
        
        lon_span = self.lon_max - self.lon_min
        lat_span = self.lat_max - self.lat_min
        start_lon = self.lon_min + (lon_span * xy[0])
        start_lat = self.lat_min + (lat_span * xy[1])
        
        km_per_deg_lon = 111.32 * math.cos(math.radians(start_lat))
        lon_width = length_km / km_per_deg_lon
        end_lon = start_lon + lon_width
        
        for ax in target_axs:
            ax.plot([start_lon, end_lon], [start_lat, start_lat], 
                    color=color, linewidth=linewidth, transform=self.proj)
            ax.text((start_lon + end_lon) / 2, start_lat + (lat_span * 0.015), 
                    f'{length_km} km', color=color, ha='center', va='bottom', 
                    fontsize=fontsize, transform=self.proj)
            
        self._lock_extents()
        return self

    def add_gridlines(self, n_ticks=4, color='gray', fontsize=9):
        """
        Native Cartopy gridlines.
        
        Args:
            n_ticks (int): Number of grid ticks along each axis.
            color (str): Color of the gridlines.
            fontsize (int): Font size for the grid labels.
        
        """
        for ax in self.active_axs:
            # Disable Matplotlib ticks to prevent messy overlaps
            ax.tick_params(left=False, right=False, bottom=False, top=False, 
                           labelleft=False, labelbottom=False)
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color=color, alpha=0.5, crs=self.proj)
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlocator = mticker.MaxNLocator(n_ticks)
            gl.ylocator = mticker.MaxNLocator(n_ticks)
            gl.xlabel_style = {'size': fontsize, 'color': 'black'}
            gl.ylabel_style = {'size': fontsize, 'color': 'black'}
            
        self._lock_extents()
        return self

    def annotate_dates(self, dates, xy=(0.05, 0.95), fontsize=10, color='white', bg_color='black', add_letter_labels=False):
        """
        Overlays text annotations (e.g., dates) onto the maps.
        
        Args:
            dates (list of str): List of date strings to annotate on each subplot. Should match the number of active subplots.
            xy (tuple): Fractional coordinates (x, y) for the position of the annotations on the map.
            fontsize (int): Font size for the annotations.
            color (str): Text color for the annotations.
            bg_color (str): Background color for the annotation text to improve readability.
            add_letter_labels (bool): Whether to add letter labels (a, b, c, ...) to each subplot.
        """
        for i, ax in enumerate(self.active_axs):
            if i < len(dates):
                label = f"{chr(97 + i)}) {dates[i]}" if add_letter_labels else dates[i]
                ax.annotate(label, xy=xy, xycoords='axes fraction', ha='left', va='top', 
                            fontsize=fontsize, color=color, backgroundcolor=bg_color, alpha=0.7)
        return self
    
    def add_letter_labels(self, xy=(0.05, 0.95), fontsize=10, color='white', bg_color='black'):
        """
        Adds letter labels (a, b, c, ...) to each subplot for easy reference.
        
        Args:
            xy (tuple): Fractional coordinates (x, y) for the position of the labels on the map.
            fontsize (int): Font size for the labels.
            color (str): Text color for the labels.
            bg_color (str): Background color for the labels to improve readability.
        """
        for i, ax in enumerate(self.active_axs):
            label = chr(97 + i)  # Convert index to ASCII lowercase letter (a, b, c, ...)
            ax.annotate(label + ")", xy=xy, xycoords='axes fraction', ha='left', va='top', 
                        fontsize=fontsize, color=color, backgroundcolor=bg_color, alpha=0.7)
        return self
    
    def add_north_arrow(self, xy=(0.95, 0.05), size=20, color='black', apply_to_all=False):
        """
        Adds a native north arrow to the map.
        
        Args:
            xy (tuple): Fractional coordinates (x, y) for the position of the north arrow on the map.
            size (int): Size of the north arrow.
            color (str): Color of the north arrow.
            apply_to_all (bool): Whether to apply the north arrow to all active subplots.
        """
        target_axs = self.active_axs if apply_to_all else [self.active_axs[0]]
        
        for ax in target_axs:
            # FIX: Using stacked text avoids the 'squished arrow' bug caused by transAxes on forced aspect ratios
            ax.annotate('N\n↑', xy=xy, xycoords='axes fraction', ha='center', va='center', 
                        fontsize=size, color=color, fontweight='bold')
        return self
    
    def add_legend(self, labels, colors, loc='upper right', fontsize=10, apply_to_all=False):
        """
        Adds a native Matplotlib legend to the map.
        
        Args:
            labels (list of str): List of labels for the legend entries.
            colors (list of str): List of colors corresponding to each label.
            loc (str): Location of the legend on the map.
            fontsize (int): Font size for the legend text.
            apply_to_all (bool): Whether to apply the legend to all active subplots.

        """
        from matplotlib.patches import Patch
        target_axs = self.active_axs if apply_to_all else [self.active_axs[0]] # Default to first map
        
        legend_elements = [Patch(facecolor=color, edgecolor='black', label=label) 
                           for label, color in zip(labels, colors)]
        
        for ax in target_axs:
            ax.legend(handles=legend_elements, loc=loc, fontsize=fontsize, frameon=True)
        return self

    def add_colorbar(self, label='Value', loc='right', labelfontsize=10, color='black'):
        """
        Native Matplotlib colorbar.
        
        Args:
            label (str): Label for the colorbar.
            loc (str): Location of the colorbar ('right', 'left', 'top', 'bottom').
            labelfontsize (int): Font size for the colorbar label.
            color (str): Color of the colorbar label.
        """
        if not self.is_rgb:
            vmin = self.vis_params.get('min', 0)
            vmax = self.vis_params.get('max', 1)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            mapper = cm.ScalarMappable(norm=norm, cmap=self.cmap)
            
            orientation = 'vertical' if loc in ['right', 'left'] else 'horizontal'
            cbar = self.fig.colorbar(mapper, ax=self.active_axs, orientation=orientation, 
                                     fraction=0.03, pad=0.04)
            cbar.set_label(label, fontsize=labelfontsize, color=color)
        return self

    def save(self, filepath, filetype=None):
        """
        Saves the figure to disk locally.
        
        Args:
            filepath (str): Path to save the figure.
        """
        if filetype:
            # Strip any accidental dots passed by user (e.g., '.png' becomes 'png')
            filetype = filetype.lower().strip('.')
            if filetype not in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                raise ValueError("Unsupported filetype. Supported types are: png, jpg, jpeg, pdf, svg.")
            
            # Safely append if missing
            if not filepath.lower().endswith(f".{filetype}"):
                filepath = f"{filepath}.{filetype}"
        else:
            # Fallback if neither filetype nor extension was provided
            if '.' not in filepath:
                filepath = f"{filepath}.png"

        self.fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', transparent=False)
        print(f"Figure saved to {filepath}")
        return self

    def show(self):
        """Displays the figure."""
        plt.show()
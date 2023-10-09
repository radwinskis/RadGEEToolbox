# RadGEEToolbox

### Python package simplifying large-scale operations using Google Earth Engine (GEE) for users who utilize Landsat and Sentinel 

Initially created by Mark Radwin to help simplify processing imagery for PhD studies and general image exploration, this package offers helpful functionality with an outlook to add furthur functionality to aid assorted Earth observation specialists. 

The package is divided into four modules:
- LandsatCollection
- SentinelCollection
- CollectionStitch
- GetPalette

where LandsatCollection.py and SentinelCollection.py arethe main two modules for the majority of image processing. 

Almost all functionality is server-side friendly.

You can easily go back-and-forth from RadGEEToolbox and GEE objects to maximize efficiency in workflow.


## 🚀 Installation Instructions

### 🔍 Prerequisites

- **Python**: Ensure you have version 3.6 or higher installed.
- **pip**: This is Python's package installer. 

### 📦 Installing via pip (COMING SOON!!!)

To install `RadGEETools` version 1.3 using pip:

```bash
pip install RadGEETools==1.3
```

### 🔧 Manual Installation from Source

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/yourusername/RadGEETools.git
   ```

2. **Navigate to Directory**: 
   ```bash
   cd RadGEETools
   ```

3. **Install the Package**:
   ```bash
   pip install .
   ```

### ✅ Verifying the Installation

To verify that `RadGEETools` was installed correctly:

```python
python -c "import RadGEETools; print(RadGEETools.__version__)"
```

You should see `1.3` printed as the version number.

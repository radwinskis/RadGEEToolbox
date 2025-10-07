Installation Instructions
=========================

Prerequisites
-------------

- **Python**: Ensure you have version 3.8 or higher installed.
- **pip**: This is Python's package installer.
- **conda-forge**: Community-led Conda package installer channel.

Installing via pip
------------------

To install ``RadGEEToolbox`` version 1.6.8 using pip (recommended: use a virtual environment)::

    pip install RadGEEToolbox==1.6.8

Installing via Conda
--------------------

To install ``RadGEEToolbox`` version 1.6.8 using conda-forge (recommended: use a virtual environment)::

    conda install conda-forge::radgeetoolbox

Manual Installation from Source
-------------------------------

1. **Clone the repository**::

       git clone https://github.com/radwinskis/RadGEEToolbox.git

2. **Navigate to the directory**::

       cd RadGEEToolbox

3. **Install the package**::

       pip install .

Verifying the Installation
-----------------------------

To verify that ``RadGEEToolbox`` was installed correctly::

    python -c "import RadGEEToolbox; print(RadGEEToolbox.__version__)"

You should see ``1.6.8`` printed as the version number.

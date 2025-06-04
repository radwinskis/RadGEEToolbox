Getting Started with Google Earth Engine
========================================

RadGEEToolbox requires `access <https://developers.google.com/earth-engine/guides/access>`_ to `Google Earth Engine (GEE) <https://earthengine.google.com/>`_ and proper `authentication <https://developers.google.com/earth-engine/guides/auth>`_ with the Python API.

For more details, see the official `Google Earth Engine Python API Getting Started Guide <https://developers.google.com/earth-engine/guides/quickstart_python>`_ or `Python Installation Guide <https://developers.google.com/earth-engine/guides/python_install>`_.

1. Sign Up for Earth Engine
---------------------------

Apply for access here: https://earthengine.google.com/signup/ using a Google account. Approval typically takes a few days.

2. Install the Earth Engine Python API
--------------------------------------

**The Earth Engine API is installed automatically when you install ``RadGEEToolbox``.** However, if you would like to install manually:

.. code-block:: bash

   pip install earthengine-api

3. Authenticate & Initialize
----------------------------

Prior to using the Earth Engine Python client library, you need to authenticate and use the resultant credentials to initialize the Python client. The authentication flows use Cloud Projects to authenticate, and they're used for unpaid (free, noncommercial) use as well as paid use.

**Run the following during your first use:**

.. code-block:: python

   import ee
   ee.Authenticate()

.. note::

   Your GEE credentials will not permanently be stored on your PC and you will periodically need to re-run ``ee.Authenticate()`` when ``ee.Initialize()`` returns an authentication error.

``ee.Authenticate()`` will select the best authentication mode for your environment, and prompt you to confirm access for your scripts. To initialize, you will need to provide a project that you own, or have permissions to use. This project will be used for running all Earth Engine operations:

.. code-block:: python

   ee.Initialize(project='my-project')

Replace ``'my-project'`` with the name of the Google Cloud Project you created on sign-up or any Google Cloud Project that has the GEE API enabled.

4. Authentication Best Practices
--------------------------------

It is reccomended to use the following authentication procedure once you have completed your initial authentication:

.. code-block:: python
    import ee

    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

5. Troubleshooting
------------------

**AttributeError: module 'ee' has no attribute 'Initialize'**

➤ Ensure that ``earthengine-api`` is installed properly.

**403 Access Denied**

➤ Your GEE account might not be approved, or the API is not enabled for your project.

**Initialization fails**

➤ Try using ``ee.Initialize(project='your-project-id')`` explicitly in case you are just calling ``ee.Initialize()``.

See the official `GEE documentation for authentication » <https://developers.google.com/earth-engine/guides/auth>`_

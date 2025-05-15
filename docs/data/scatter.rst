.. _scattered_data:

Using Scattered Data
====================

Scattered data is commonly encountered in glaciology and geoscience applications where observations are irregularly distributed (e.g., satellite tracks, field campaigns). PINNICLE is designed to handle such data efficiently and robustly during both forward and inverse modeling tasks.

Overview
--------

Scattered data refers to observations without a structured grid or mesh, such as point measurements, flight tracks, etc.

.. note::

   Currently, PINNICLE only support loading this type of data from :code:`.mat` files, but is planning to support other data format, such as :code:`.csv`, :code:`.nc`, etc.

Preprocessing Recommendations
-----------------------------

It is highly recommanded to use one data file per variable, so that you can easily have different sizes and coordinates for different variables.

Configuration
-------------

To use scattered data in your PINNICLE configuration, specify a dataset like this:

.. code-block:: python

   hp["data"] = {
       "flight track": {
           "data_path": "glacier_obs.mat",
           "data_size": {"H": 2000},
           "name_map": {"H":"H_mat"},
           "X_map": {"x":"x_mat", "y":"y_mat"},
           "source": "mat"
       }
   }

To add multiple data sources in PINNICLE, you can put them in the same dict in :code:`hp["data"]`, just with different `key`, and the `key` can be any string.

.. code-block:: python

   hp["data"] = {
       "flight track 1": {
           "data_path": "glacier_obs.mat",
           "data_size": {"H": 2000},
           "name_map": {"H":"H_mat"},
           "X_map": {"x":"x_mat", "y":"y_mat"},
           "source": "mat"
       },
       "flight track 2": {
       "data_path": "otherdata.mat",
       "data_size": {"H": 500},
       "name_map": {"H":"H_other"},
       "X_map": {"x":"x_other", "y":"y_other"},
       "source": "mat"
       }
   }

See the :ref:`examples` section for full demonstrations using Matlab data.


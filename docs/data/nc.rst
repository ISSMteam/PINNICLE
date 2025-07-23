.. _nc_data:

Using netCDF (.nc) Data
=====================

PINNICLE supports reading observational or model data stored in the netCDF (`.nc`) format, a widely used standard in Earth science and machine learning for managing large, hierarchical datasets. 

Overview
--------

Similar in :ref:`h5_data` , the netCDF files store data are also in a hierarchical format using datasets and groups. PINNICLE can navigate these structures and extract relevant fields for use in physics-informed neural networks. It supports both structured gridded data and scattered point data stored in `.nc` containers.

Configuration
-------------

To use :code:`.nc` data in PINNICLE, add a new data source with :code:`"source": "nc"` in your configuration, and you **need** to set the mapping for both the coordinates and variables:

.. code-block:: python

   hp["data"] = {
      "MynetCDF": {
         "data_path": ncpath,
         "X_map": {"x":"surf_x", "y":"surf_y" },
         "name_map": {"s":"surf_elv", "u":"surf_vx", "v":"surf_vy", "a":"surf_SMB", "b":"bed_BedMachine"},
         "data_size": {"u": 5000, "v": 5000, "H": 5000, "s": 5000},
         "scaling": {"u":1.0/31536000, "v":1.0/31536000},
         "source": "nc"
       }
   }


- :code:`"data_path"`: Path to the :code:`.nc` file
- :code:`"X_map"`: set the name mapping of the coordinates in PINNICLE to the :code:`.nc` file 
- :code:`"name_map"`: set the name mapping of the variables in PINNICLE to the :code:`.nc` file 
- :code:`"data_size"`: Number of data points to randomly sample for each variable
- :code:`"scaling"`: scaling factors multiplied to the data, the key is the same as the :code:`"name_map"`
- Set a variable to :code:`"None"` to infer it is only used as a Dirichlet boundary condition
- If the key is not mentioned in :code:`"data_size"`, then the corresponding field will not use data from this file


See the :ref:`examples` section for workflows that incorporate `.nc` datasets.

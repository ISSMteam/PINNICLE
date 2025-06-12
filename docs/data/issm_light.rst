.. _issm_light_data:

Using ISSM Model Data (light version)
=====================================

PINNICLE is fully compatible with output from the `Ice-sheet and Sea-level System Model (ISSM) <https://github.com/ISSMteam/ISSM>`_. 
This ISSM Light data loader is a streamlined alternative to the standard :ref:`issm_data` module.
The key difference is that this lighter version does not load any mesh or connectivity information. It only extracts data as pointwise coordinate and associated fields. As a result, all fields are treated as scattered data, allowing users to easily subset the domain without modifying the original ISSM model file.


Overview
--------

ISSM model output typically includes spatial fields such as:

.. list-table:: 
   :widths: 25 50 30
   :header-rows: 1

   * - **Field**
     - **Name in ISSM**
     - **Key in PINNICLE**
   * - Mesh information
     - :code:`md.mesh.x`, :code:`md.mesh.y`
     - :code:`"x"`, :code:`"y"`
   * - Ice velocity components
     - :code:`md.inversion.vx_obs`, :code:`md.inversion.vy_obs`
     - :code:`"u"`, :code:`"v"`
   * - Ice thickness
     - :code:`md.geometry.thickness`
     - :code:`"H"`
   * - Surface elevation
     - :code:`md.geometry.surface`
     - :code:`"s"`
   * - Basal friction coefficient
     - :code:`md.friction.C`
     - :code:`"C"`
   * - Ice rheology factor
     - :code:`md.materials.rheology_B``
     - :code:`"B"`
   * - Surface mass balance
     - :code:`md.smb.mass_balance-md.balancethickness.thickening_rate`
     - :code:`"a"`

PINNICLE automatically reads, processes, and extracts relevant fields for training and model initialization, and assigns them to the corresponding variables.

Preprocessing Recommendations
-----------------------------

- Export variables from ISSM with consistent units (SI system: m, s, Pa).
- Save structured data using :code:`saveasstruct(md, filename)` in MATLAB to export the ISSM model to a nested struct.
- Use mesh files (:code:`.exp`) from ISSM as shapefile input to define simulation domain.

Configuration
-------------

To use ISSM data, specify a dataset block in the configuration dictionary:

.. code-block:: python

   hp["data"] = {
       "ISSM": {
           "source":"ISSM Light",
           "data_path": "Helheim.mat",
           "data_size": {"u": 4000, "v": 4000, "H": 4000, "s": 4000, "C": None}
       }
   }

- :code:`"source"`: must be set to :code:`"ISSM Light"` to activate this loader
- :code:`"data_path"`: Path to the :code:`.mat` file containing ISSM model
- :code:`"data_size"`: Number of data points to randomly sample for each variable
- Set a variable to :code:`"None"` to infer it is only used as a Dirichlet boundary condition
- If the key is not mentioned in :code:`"data_size"`, then the corresponding field will not use data from this file


See the :ref:`examples` section for full demonstrations using ISSM input.

.. _issm_data:

Using ISSM Model Data
=====================

PINNICLE is fully compatible with output from the Ice-sheet and Sea-level System Model (ISSM). ISSM is a widely used finite-element ice sheet model that generates structured, mesh-based data in `.mat` format. PINNICLE includes dedicated functionality to parse, align, and sample ISSM-generated datasets for both forward and inverse simulations.

Overview
--------

ISSM model output typically includes spatial fields such as:

- Ice velocity components (:math:`u`, :math:`v`)
- Ice thickness (:math:`H`)
- Surface elevation (:math:`s`)
- Bed elevation (:math:`b`)
- Basal friction coefficient (:math:`C`)
- Ice rheology (e.g., Glen’s flow-law pre-factor :math:`B`)

These variables are stored in MATLAB `.mat` files as structured arrays (e.g., `md.mesh`, `md.results`, `md.inversion`).

PINNICLE automatically reads and processes these fields, extracting relevant variables for training and model initialization.

Configuration
-------------

To use ISSM data, specify a dataset block in the configuration dictionary:

.. code-block:: python

   hp["data"] = {
       "ISSM": {
           "data_path": "Helheim.mat",
           "data_size": {"u": 4000, "v": 4000, "H": 4000, "s": 4000, "C": None}
       }
   }

- `data_path`: Path to the ISSM `.mat` file
- `data_size`: Number of data points to randomly sample for each variable
- Set a variable to `None` to infer it (e.g., `"C": None` for basal friction inversion)

ISSM Compatibility Notes
-------------------------

- PINNICLE can read both node-based and element-based fields from ISSM.
- It automatically identifies variables within the ISSM `model struct`.
- Mesh geometry (e.g., coordinates, connectivity) is used to generate collocation points and define domain boundaries.
- Works seamlessly with time-dependent ISSM simulations if each time step is exported separately.

Time-Dependent ISSM Data
------------------------

For transient modeling, provide a dictionary of time-stamped ISSM datasets:

.. code-block:: python

   for t in np.linspace(2008, 2009, 11):
       issm = {}
       if t == 2008:
           issm["data_size"] = {"u": 3000, "v": 3000, "a": 3000, "H": 3000}
       else:
           issm["data_size"] = {"u": 3000, "v": 3000, "a": 3000, "H": None}

       issm["data_path"] = f"Helheim_Transient_{t}.mat"
       issm["default_time"] = t
       issm["source"] = "ISSM"
       hp["data"][f"ISSM{t}"] = issm

PINNICLE will align and sample the time series accordingly, allowing for smooth transient modeling from ISSM outputs.

Preprocessing Recommendations
-----------------------------

- Export variables from ISSM with consistent units (SI system: m, s, Pa).
- Save structured data using `save -v7.3` in MATLAB to ensure compatibility with Python’s `h5py`.
- Confirm field names (e.g., `md.results.Vx`, `md.mesh.x`) align with PINNICLE expectations.
- Use mesh files (`.exp`) from ISSM as shapefile input to define simulation domain.

Inverse Modeling from ISSM
--------------------------

You can infer unobserved parameters (e.g., basal friction or rheology) by using ISSM-generated synthetic observations and omitting the target from the input:

.. code-block:: python

   "C": None  # PINNICLE will solve for basal friction to match other fields

This is demonstrated in:
- **Example 1:** Helheim inverse problem
- **Example 2:** Joint inference on Pine Island Glacier

Summary
-------

ISSM integration in PINNICLE allows for:
- Direct reuse of model output for PINN training
- High-fidelity benchmarking and inversion
- Seamless transition from traditional models to PINNs

See the `Examples <examples.html>`_ section for full demonstrations using ISSM input.


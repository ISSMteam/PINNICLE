.. _scattered_data:

Using Scattered Data
====================

Scattered data is commonly encountered in glaciology and geoscience applications where observations are irregularly distributed (e.g., satellite tracks, field campaigns). PINNICLE is designed to handle such data efficiently and robustly during both forward and inverse modeling tasks.

Overview
--------

Scattered data refers to observations without a structured grid, such as point measurements of:

- Ice surface velocity (:math:`u`, :math:`v`)
- Ice thickness (:math:`H`)
- Surface elevation (:math:`s`)
- Basal friction coefficient (:math:`C`)
- Surface mass balance (:math:`a`)

PINNICLE can load this type of data from `.mat`, `.csv`, or `.nc` files and integrate it into the physics-informed loss function.

Configuration
-------------

To use scattered data in your PINNICLE configuration, specify a dataset like this:

.. code-block:: python

   hp["data"] = {
       "Scattered": {
           "data_path": "observations.csv",
           "data_size": {"u": 3000, "v": 3000, "s": 3000},
           "source": "csv"
       }
   }

Or for MATLAB files:

.. code-block:: python

   hp["data"] = {
       "Survey": {
           "data_path": "glacier_obs.mat",
           "data_size": {"H": 2000},
           "source": "mat"
       }
   }

Each entry must specify:
- `data_path`: path to the file
- `data_size`: number of points to randomly sample for each variable
- `source`: (optional) file format, default is `"mat"`

Sampling Strategy
-----------------

To ensure uniform spatial coverage, PINNICLE applies the following process:

1. **Grid-Based Downsampling**: Data points are assigned to a temporary Cartesian grid.
2. **Spatial Filtering**: Points within the same grid cell are randomly filtered to avoid clustering.
3. **Random Sampling**: From the filtered dataset, a user-specified number of points are selected for training.

This process improves generalization and prevents local overfitting around data clusters.

Inverse Problems with Scattered Data
------------------------------------

In inverse modeling, scattered data can be used to infer unknown fields. For example, to infer basal friction using only surface velocity and elevation data:

.. code-block:: python

   hp["data"] = {
       "Surface": {
           "data_path": "survey_data.mat",
           "data_size": {"u": 5000, "v": 5000, "s": 5000, "C": None}
       }
   }

Here, setting `"C": None` indicates that the basal friction coefficient should be learned through PDE constraints.

Benefits
--------

- No interpolation or gridding required
- Compatible with sparse field measurements
- Can be combined with structured datasets
- Downsampling ensures even geographic representation

Best Practices
--------------

- Choose sample sizes that balance accuracy and computational cost.
- Use unit-consistent data (e.g., m/s, m, Pa) to ensure stability during training.
- For highly clustered datasets, increase the grid resolution or reduce sample size to avoid overrepresentation.

Examples
--------

Scattered data is used in:
- **Example 1:** Inverse friction inference on Helheim Glacier
- **Example 2:** Joint inversion of friction and rheology using pointwise surface observations

Refer to the `Examples <examples.html>`_ section for complete training scripts and visualizations.


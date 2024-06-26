Prepare Training Data
=====================

Data Format
-----------

PINNICLE currently supports the following data formats:

- `ISSM <https://issm.jpl.nasa.gov>`_ ``model()`` type
- Scattered data

You can also add customized data loaders in `pinnicle.modeldata <https://github.com/ISSMteam/PINNICLE/tree/main/pinnicle/modeldata>`_.


ISSM Data
---------

This is the `default` data format that PINNICLE can load.

Since ``.mat`` files lack class information about the objects, the easiest way to load an ISSM model is to first save it from ISSM using ``saveasstruct(md, filename)``, and then load it with the following settings:

.. code-block:: python

   issm = {}
   issm["data_path"] = filename
   issm["data_size"] = {"u": 1000, "v": 1000, "s": 1000, "H": 1000, "C": None, "vel": 1000}
   hp["data"] = {"ISSM": issm}

The ``filename`` is the data file name (including the path) saved from ISSM. The value of ``"data_size"`` specifies the number of data points used for training. The key in ``hp["data"]`` is simply an identifier, which in the example above is set to ``"ISSM"``.

For the PINNICLE data loader to work properly, the ISSM model must contain the following variables:

- ``md.mesh.x`` and ``md.mesh.y``: Used to get ``x`` and ``y`` coordinates. Currently, PINNICLE only supports **2D** meshes.
- ``md.mask.ice_levelset``: Used to determine the ice-covered region for ``ice_mask``.

Additionally, PINNICLE will try to load the following variables from the ``.mat`` file. Depending on the physics being solved, some of these can be empty if not needed:

- Ice velocity ``u`` and ``v`` from ``md.inversion.vx_obs`` and ``md.inversion.vy_obs``. In ISSM, these velocities are in `m/yr`, but they will be converted to `m/s` in PINNICLE.
- Surface elevation ``s`` from ``md.geometry.surface``.
- Ice thickness ``H`` from ``md.geometry.thickness``.
- Surface accumulation rate ``a`` from ``md.smb.mass_balance - md.balancethickness.thickening_rate``, converted from `m/yr` (default in ISSM model) to `m/s`.
- Friction coefficient ``C`` from ``md.friction.C`` for Weertman's law.
- Rheology ``B`` from ``md.materials.rheology_B``.
- Velocity magnitudes ``vel`` computed from :math:`\sqrt{u^2+v^2}`.

More details are in `pinnicle.modeldata.ISSMmdData.load_data() <https://pinnicle.readthedocs.io/en/latest/_modules/pinnicle/modeldata/issm_data.html#ISSMmdData.load_data>`_.


Scattered Data
--------------

A more general data format is scattered data saved in ``.mat`` format.

.. code-block:: python

   flightTrack = {}
   flightTrack["data_path"] = filename
   flightTrack["data_size"] = {"H": 1000}
   flightTrack["name_map"] = {"H": "thickness"}
   flightTrack["source"] = "mat"
   hp["data"] = {"ft": flightTrack}

The ``filename`` is the path and name of the data file. The value of ``"data_size"`` specifies the number of data points used for training. The value of ``"name_map"`` defines the mapping between the variable name in PINNICLE (``"H"``) and the corresponding name in the data file (``"thickness"`` in this case). The value of ``"source"`` is the data file type. Currently, PINNICLE only supports ``"ISSM"`` and ``"mat"``. The key in ``hp["data"]`` is simply an identifier.


Multiple Data Sources
---------------------

PINNICLE can load training data from multiple data files. Simply include all the data file settings in the dictionary ``hp["data"]`` with different keys.

.. code-block:: python

   hp["data"] = {"ISSM": issm, "ft": flightTrack}

Add Customized Data Loader
--------------------------

To define a customized data loader, you can create a class as follows. The property ``_DATA_TYPE`` needs to be unique for each new data loader class:

.. code-block:: python

   from . import DataBase
   from ..parameter import SingleDataParameter
   from ..physics import Constants

   class YourData(DataBase, Constants):
       """Data loaded from a file."""
       _DATA_TYPE = "your data format"
       
       def __init__(self, parameters=SingleDataParameter()):
           Constants.__init__(self)
           super().__init__(parameters)

and define the following functions:

- ``get_ice_coordinates(self, mask_name="")``: Returns a 2D array with coordinates of the ice-covered region.
- ``load_data(self)``: Loads the data from the data file.
- ``prepare_training_data(self, data_size=None)``: Puts the coordinates in ``self.X`` and data in ``self.sol`` with the keys corresponding to the physics.


Activation Catcher
==================

Capture module inputs/outputs during a forward pass and select them by
``product_id``. See :doc:`../architecture` ("Capturing activations as products")
for how this plugs into a teacher supplier.

.. autoclass:: softs.catcher.ModelIOCatcher
   :members:
   :undoc-members:

.. autofunction:: softs.catcher.parse_io_spec

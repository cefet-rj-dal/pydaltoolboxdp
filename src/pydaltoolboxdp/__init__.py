"""Deep ++ extensions for the Python DAL ecosystem.

The package owns the Python implementations consumed directly in Python and through
the sibling R package ``daltoolboxdp`` via ``reticulate``.
"""

from .adversarial import AutoencAdvE, AutoencAdvEd
from .autoencoder import AutoencDenoiseE, AutoencDenoiseEd, AutoencE, AutoencEd
from .convolutional import AutoencConvE, AutoencConvEd
from .recurrent import AutoencLstmE, AutoencLstmEd
from .stacked import AutoencStackedE, AutoencStackedEd
from .variational import AutoencVariationalE, AutoencVariationalEd

__all__ = [
    "AutoencAdvE",
    "AutoencAdvEd",
    "AutoencConvE",
    "AutoencConvEd",
    "AutoencDenoiseE",
    "AutoencDenoiseEd",
    "AutoencE",
    "AutoencEd",
    "AutoencLstmE",
    "AutoencLstmEd",
    "AutoencStackedE",
    "AutoencStackedEd",
    "AutoencVariationalE",
    "AutoencVariationalEd",
]
__version__ = "0.3.1"

"""Deep ++ extensions for the Python DAL ecosystem.

The package owns the Python implementations consumed directly in Python and through
the sibling R package ``daltoolboxdp`` via ``reticulate``.
"""

from .adversarial import AutoencAdvE, AutoencAdvEd
from .autoencoder import AutoencDenoiseE, AutoencDenoiseEd, AutoencE, AutoencEd
from .convolutional import AutoencConvE, AutoencConvEd
from .events import HanAutoencoder, HcpWaypoint
from .recurrent import AutoencLstmE, AutoencLstmEd
from .stacked import AutoencStackedE, AutoencStackedEd
from .supervised import SkclaGb, TorchClaMlp, TorchRegMlp
from .timeseries import TorchTsMlp, TsConv1D, TsLstm
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
    "HanAutoencoder",
    "HcpWaypoint",
    "SkclaGb",
    "TorchClaMlp",
    "TorchRegMlp",
    "TorchTsMlp",
    "TsConv1D",
    "TsLstm",
]
__version__ = "0.6.0"

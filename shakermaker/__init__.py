# Core simulation engine
from shakermaker.shakermaker import ShakerMaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource
from shakermaker.faultsource import FaultSource
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from shakermaker.stationlistwriter import StationListWriter
from shakermaker.sourcetimefunction import SourceTimeFunction
from shakermaker.ffspsource import FFSPSource

# Source time function extensions (lightweight, no heavy external dependencies)
from shakermaker.stf_extensions import Dirac, Discrete, Brune, Gaussian, SRF2

__all__ = [
    "__version__",
    # Core
    "ShakerMaker",
    "CrustModel",
    "PointSource",
    "FaultSource",
    "Station",
    "StationList",
    "StationListWriter",
    "StationObserver",
    "SourceTimeFunction",
    "FFSPSource",
    # Source time functions
    "Dirac",
    "Discrete",
    "Brune",
    "Gaussian",
    "SRF2",
]

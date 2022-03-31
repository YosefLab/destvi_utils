"""destvi_utils."""

import logging

from rich.console import Console
from rich.logging import RichHandler
from ._mymodule import automatic_proportion_threshold, explore_gamma_space, de_genes

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "destvi_utils"
__version__ = importlib_metadata.version(package_name)

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("destvi_utils: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["automatic_proportion_threshold", "explore_gamma_space", "de_genes"]
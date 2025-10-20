"""Shared logging setup for the mesh identifier pipeline."""
import logging

LOG_FORMAT = "[%(asctime)s][%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("mesh_identifier")

__all__ = ["LOGGER"]

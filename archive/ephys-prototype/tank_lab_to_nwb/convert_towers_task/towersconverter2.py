"""Authors: Cody Baker and Ben Dichter."""
from pathlib import Path
from typing import Optional, Union
import numpy as np

#from neuroconv import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface
from neuroconv import NWBConverter
from neuroconv.datainterfaces import SpikeGLXRecordingInterface
from neuroconv.datainterfaces import *
import spikeextractors as se

from .virmenbehaviordatainterface import VirmenDataInterface
from ..utils import convert_mat_file_to_dict

OptionalArrayType = Optional[Union[list, np.ndarray]]
PathType = Union[Path, str]

import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def get_memmap_shape(
    filename: Union[str, Path],
    dtype: Union[str, np.dtype],
    num_channels: Optional[int] = None,
    offset: int = 0,
) -> Tuple[int, ...]:
    """Calculate the shape of a memory-mapped binary file.

    Parameters
    ----------
    filename : str or Path
        Path to the binary file.
    dtype : str or np.dtype
        Data type of the binary file contents.
    num_channels : int, optional
        Number of channels in the data (if multi-channel).
    offset : int, default=0
        Number of bytes to skip at start of file.

    Returns
    -------
    tuple
        Shape of the memory-mapped array.

    Raises
    ------
    ValueError
        If the file size is not compatible with the dtype.
    """
    dtype = np.dtype(dtype)
    with open(filename, mode="rb") as f:
        f.seek(0, 2)  # Seek to end of file
        flen = f.tell()
        bytes = flen - offset

        if bytes % dtype.itemsize != 0:
            raise ValueError(
                "Size of available data is not a multiple of the data-type size."
            )

        size = bytes // dtype.itemsize
        if num_channels is None:
            shape = (size,)
        else:
            shape = (size // num_channels, num_channels)

        return shape


def create_data_stub(
    source_folder: Union[str, Path],
    samples: int,
    output_folder: Optional[Union[str, Path]] = None,
    file_pattern: str = "*.ap.bin",
    dtype: Union[str, np.dtype] = np.int16,
    num_channels: int = 385,
    copy_meta: bool = True,
    meta_pattern: str = "*.meta",
) -> Path:
    """Create a stubbed version of a dataset by copying a limited number of samples.

    Parameters
    ----------
    source_folder : str or Path
        Path to the folder containing the original data.
    samples : int
        Number of samples to include in the stub.
    output_folder : str or Path, optional
        Path where to save the stubbed data. If None, creates a folder
        with '_stubbed' suffix in the same location as source_folder.
    file_pattern : str, default="*.ap.bin"
        Pattern to match the data files.
    dtype : str or np.dtype, default=np.int16
        Data type of the binary files.
    num_channels : int, default=385
        Number of channels in the data.
    copy_meta : bool, default=True
        Whether to copy associated metadata files.
    meta_pattern : str, default="*.meta"
        Pattern to match metadata files.

    Returns
    -------
    Path
        Path to the output folder.

    Raises
    ------
    FileNotFoundError
        If source folder doesn't exist.
    ValueError
        If samples is not positive.

    Examples
    --------
    >>> # Basic usage with defaults
    >>> source_path = Path("/path/to/data")
    >>> stubbed_path = create_data_stub(source_path, samples=100)

    >>> # Custom configuration
    >>> stubbed_path = create_data_stub(
    ...     source_folder=source_path,
    ...     samples=200,
    ...     output_folder="/custom/output",
    ...     file_pattern="*.bin",
    ...     dtype=np.float32,
    ...     num_channels=385,
    ...     copy_meta=False
    ... )
    """
    # Input validation
    source_folder = Path(source_folder)
    if not source_folder.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if samples <= 0:
        raise ValueError("Number of samples must be positive")

    # Setup output folder
    if output_folder is None:
        output_folder = Path.home() / f"{source_folder.name}_stubbed"
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Copy meta files if requested
    if copy_meta:
        meta_files = list(source_folder.rglob(meta_pattern))
        for meta_file in meta_files:
            output_path = output_folder / meta_file.relative_to(source_folder)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(meta_file, output_path)

    # Process binary files
    bin_files = list(source_folder.rglob(file_pattern))
    for bin_file in bin_files:
        # Get shape and create memory map
        shape = get_memmap_shape(bin_file, dtype=dtype, num_channels=num_channels)
        memmap = np.memmap(bin_file, dtype=dtype, order="C", mode="r", shape=shape)

        # Select samples to stub
        memmap_to_stub = memmap[:samples, :]

        # Create output file
        output_path = output_folder / bin_file.relative_to(source_folder)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Create and fill stubbed memmap
        stubbed_memmap = np.memmap(
            output_path, dtype=dtype, mode="w+", shape=memmap_to_stub.shape
        )
        stubbed_memmap[:] = memmap_to_stub[:]
        stubbed_memmap.flush()

    return output_folder


if __name__ == "__main__":
    # Example with default settings
    source_path = Path("/Volumes/scratch/ct5868/fake_jesse_spikeglx/")
    stubbed_path = create_data_stub(source_path, samples=100)
    print(f"Created stubbed dataset at: {stubbed_path}")

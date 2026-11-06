"""Reading ScanImage TIFF headers: every frame field, or a clear error."""

import numpy as np
import pytest
import tifffile

from tank_lab_to_nwb.imaging.scanimage import ScanImageError, read_acquisition, read_frames
from tests.synthetic_tiff import artist, frame_pixels, software, write_acquisition


def test_frames_are_numbered_across_files_in_order(tmp_path):
    paths = write_acquisition(tmp_path, 10, frames_per_file=4)
    acquisition = read_acquisition(list(reversed(paths)))  # the order given doesn't matter
    assert [file.name for file in acquisition.files] == [path.name for path in paths]
    assert acquisition.frames["frame_number"].tolist() == list(range(1, 11))
    assert acquisition.frames["file_index"].tolist() == [0] * 4 + [1] * 4 + [2] * 2
    assert acquisition.frames["page_index"].tolist() == [0, 1, 2, 3] * 2 + [0, 1]
    assert acquisition.frames["timestamp"][3] == pytest.approx(0.06)
    assert acquisition.epoch == "[2026  7 15 11 59 50.000]"


def test_single_frame_acquisition(tmp_path):
    acquisition = read_acquisition(write_acquisition(tmp_path, 1))
    assert acquisition.frames["frame_number"].tolist() == [1]
    assert acquisition.plane_frames(0).tolist() == [0]
    assert acquisition.plane_frames(1).tolist() == []


def test_i2c_packets_decode_to_block_trial_iteration(tmp_path):
    packets = {2: [(0.03, (1, 1, 595))], 3: [(0.05, (1, 1, 597)), (0.055, (2, 300, 65535))]}
    packets = write_acquisition(tmp_path, 4, packets=packets)
    found = read_acquisition(packets).packets
    assert found["frame_index"].tolist() == [1, 2, 2]
    assert found["block"].tolist() == [1, 1, 2]
    assert found["trial"].tolist() == [1, 1, 300]
    assert found["iteration"].tolist() == [595, 597, 65535]
    assert found["timestamp"].tolist() == pytest.approx([0.03, 0.05, 0.055])


def test_no_packets_gives_empty_columns(tmp_path):
    found = read_acquisition(write_acquisition(tmp_path, 2)).packets
    assert found["timestamp"].size == 0 and found["iteration"].size == 0


def test_a_negative_packet_time_is_kept(tmp_path):
    # ScanImage stores packets from just before the trigger on the first frame.
    found = read_acquisition(write_acquisition(tmp_path, 1, packets={1: [(-0.5, (2, 93, 209))]})).packets
    assert found["timestamp"].tolist() == [-0.5]


def test_a_packet_with_the_wrong_length_is_an_error(tmp_path):
    paths = write_acquisition(tmp_path, 1, raw_i2c={1: "{{0.1, [1,0,1]} }"})
    with pytest.raises(ScanImageError, match="3 bytes"):
        read_acquisition(paths)


def test_space_separated_packet_bytes_from_older_scanimage(tmp_path):
    paths = write_acquisition(tmp_path, 1, raw_i2c={1: "{{0.1, [1 0 2 0 3 0]} }"})
    packets = read_acquisition(paths).packets
    assert (packets["block"].item(), packets["trial"].item(), packets["iteration"].item()) == (1, 2, 3)


def test_unreadable_i2c_text_is_an_error(tmp_path):
    paths = write_acquisition(tmp_path, 1, raw_i2c={1: "{{oops, [1,0,1,0,1,0]} }"})
    with pytest.raises(ScanImageError, match="I2CData"):
        read_acquisition(paths)


def test_an_unknown_header_field_is_an_error_not_a_silent_drop(tmp_path):
    paths = write_acquisition(tmp_path, 1, extra="newField = 3\n")
    with pytest.raises(ScanImageError, match="newField"):
        read_acquisition(paths)


def test_a_used_aux_trigger_is_an_error_not_a_silent_drop(tmp_path):
    paths = write_acquisition(tmp_path, 1, aux="[0.5]")
    with pytest.raises(ScanImageError, match="auxTrigger0"):
        read_acquisition(paths)


def test_missing_frames_are_an_error(tmp_path):
    paths = write_acquisition(tmp_path, 6, frames_per_file=2, skip_frames=(3, 4))
    with pytest.raises(ScanImageError, match="frame 5 follows frame 2"):
        read_acquisition(paths)


def test_a_missing_file_is_an_error(tmp_path):
    paths = write_acquisition(tmp_path, 6, frames_per_file=2)
    with pytest.raises(ScanImageError, match="file 00001, 00003"):
        read_acquisition([paths[0], paths[2]])


def test_a_file_cut_off_inside_a_frame_is_an_error(tmp_path):
    paths = write_acquisition(tmp_path, 4)
    with tifffile.TiffFile(paths[0]) as tif:
        offset = tif.pages[3].dataoffsets[0]
    with open(paths[0], "r+b") as file:
        file.truncate(offset + 1)
    with pytest.raises(ScanImageError):
        read_acquisition(paths)


def test_a_file_without_scanimage_metadata_is_an_error(tmp_path):
    # The pipeline's per-ROI copies look like this.
    path = tmp_path / "copy_00001.tif"
    tifffile.imwrite(path, frame_pixels(1), description="frameNumbers = 1\n", metadata=None)
    with pytest.raises(ScanImageError, match="original acquisition"):
        read_acquisition([path])


def test_two_epochs_are_an_error(tmp_path):
    first = write_acquisition(tmp_path, 2, frames_per_file=2, stem="a_00001")
    (tmp_path / "b").mkdir()
    write_acquisition(tmp_path / "b", 4, frames_per_file=2, stem="a_00001", epoch="[2026  7 15 11 59 51.000]")
    with pytest.raises(ScanImageError, match="epoch"):
        read_acquisition([first[0], tmp_path / "b" / "a_00001_00002.tif"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"SI.hChannels.channelSave": "[1;2]"}, "channel"),
        ({"SI.hScan2D.logAverageFactor": "2"}, "logAverageFactor"),
        ({"SI.hStackManager.framesPerSlice": "2"}, "framesPerSlice"),
        ({"SI.hStackManager.numFramesPerVolumeWithFlyback": "3"}, "flyback"),
        ({"SI.hFastZ.discardFlybackFrames": "true"}, "flyback"),
        ({"SI.hStackManager.zs": "[10 50 90]"}, "zs"),
        ({"SI.hRoiManager.linesPerFrame": "5"}, "linesPerFrame"),
        ({"SI.hStackManager.zs": None}, "zs"),
    ],
)
def test_layouts_the_converter_cannot_map_are_errors(tmp_path, override, message):
    with pytest.raises(ScanImageError, match=message):
        read_acquisition(write_acquisition(tmp_path, 2, software_text=software(**override)))


def test_several_rois_are_an_error_until_they_are_supported(tmp_path):
    paths = write_acquisition(tmp_path, 2, artist_text=artist(n_rois=3))
    with pytest.raises(ScanImageError, match="3 ROIs"):
        read_acquisition(paths)


def test_one_roi_with_mroi_enabled_is_fine(tmp_path):
    text = software(**{"SI.hRoiManager.mroiEnable": "1"})
    assert read_acquisition(write_acquisition(tmp_path, 2, software_text=text)).layout.n_planes == 2


def test_layout_values(tmp_path):
    layout = read_acquisition(write_acquisition(tmp_path, 2)).layout
    assert layout.n_planes == 2
    assert layout.zs_um == (10.0, 50.0)
    assert layout.frame_shape == (3, 4)
    assert layout.frame_period == 0.02
    assert layout.volume_rate == 25.0
    assert layout.line_period == 6.3e-05
    assert layout.field_of_view_um == (120.0, 120.0)
    assert layout.pixel_size_um == (30.0, 40.0)
    assert layout.version == "2022"


def test_single_plane_has_a_scalar_z(tmp_path):
    text = software(**{"SI.hStackManager.zs": "30", "SI.hStackManager.actualNumSlices": "1", "SI.hStackManager.numFramesPerVolumeWithFlyback": "1"})
    acquisition = read_acquisition(write_acquisition(tmp_path, 3, software_text=text))
    assert acquisition.layout.zs_um == (30.0,)
    assert acquisition.plane_frames(0).tolist() == [0, 1, 2]


def test_planes_take_every_nth_frame_and_a_partial_last_volume_is_kept(tmp_path):
    acquisition = read_acquisition(write_acquisition(tmp_path, 5, frames_per_file=2))
    assert acquisition.plane_frames(0).tolist() == [0, 2, 4]
    assert acquisition.plane_frames(1).tolist() == [1, 3]


def test_a_plane_outside_the_stack_is_an_error(tmp_path):
    acquisition = read_acquisition(write_acquisition(tmp_path, 2))
    for plane in (-1, 2):
        with pytest.raises(IndexError):
            acquisition.plane_frames(plane)


def test_read_frames_returns_exact_pixels_across_files(tmp_path):
    acquisition = read_acquisition(write_acquisition(tmp_path, 5, frames_per_file=2))
    frames = read_frames(acquisition, [0, 2, 4])
    assert frames.dtype == np.int16
    for got, number in zip(frames, (1, 3, 5)):
        np.testing.assert_array_equal(got, frame_pixels(number))


def test_read_frames_of_nothing_keeps_the_frame_shape(tmp_path):
    acquisition = read_acquisition(write_acquisition(tmp_path, 2))
    assert read_frames(acquisition, []).shape == (0, 3, 4)


def test_no_files_is_an_error():
    with pytest.raises(ScanImageError, match="no TIFF"):
        read_acquisition([])

# Suite2p results on the behavior clock

The converter takes a trusted Suite2p output folder, the TIFFs of one acquisition, and its ViRMEn log. It reads Suite2p
with NumPy; Suite2p and torch are not converter dependencies. Object NPY files contain pickle and must come from a
trusted run. Nothing uploads.

```
python -m tank_lab_to_nwb.imaging.convert_suite2p LOG.mat SUITE2P_DIR OUTPUT_DIR TIFF [TIFF ...]
```

The output is `OUTPUT_DIR/staging/suite2p.nwb`, with a report beside it. Organize it using
`virmen.layout.organize_for_dandi(staging, separate_dandiset)` before local DANDI validation. Keep this separate from
the raw imaging dandiset, since the subject and session identifiers are shared.

## Alignment and what it means

`prepare_session` supplies the normalized behavior, frame times, session-start shift, in-progress trial, frame labels,
I2C packets and fit provenance. The raw converter and its tests are unchanged. The processed and raw NWB files describe
the same acquisition and behavior on the same clock; the processed file does not duplicate raw pixels.

Every response series uses `prepared.frame_times[acquisition.plane_frames(plane)]`. Neither `fs` nor a volume-average
time supplies timestamps. A partial last volume therefore produces unequal series lengths. Bad frames are retained,
with their original flags and samples. All ROIs are retained, including ones classified as noncells.

The clock fit is described in [imaging-timing.md](imaging-timing.md). Residuals measure packet scatter around a line,
not frame-time error. The constant I2C send latency is unmeasured, and the ViRMEn source sets no upper bound on it.
Timestamps mark frame starts; scanning within a frame and display latency remain separate limitations.

## Establishing the frame order

Matching trace lengths is necessary but does not establish frame identity. Before writing, the converter reconstructs
registered frames from raw TIFF pages using the saved rigid and nonrigid offsets. It samples the first, middle and last
frame of **every file for every plane**, including the first and last session frames and both sides of file boundaries.
The evidence records the raw global frame, file, page, Suite2p frame and reconstruction RMSE.

The reconstruction follows the Suite2p 1.1 registration implementation: signed `int16` pixels are unchanged before
registration; unsigned `uint16` pixels are integer-divided by two. A rigid shift rolls by `(-yoff, -xoff)`. Nonrigid
block shifts are bilinearly expanded with aligned corners, border-padded, and used for bilinear image sampling with
border replication. The final integer conversion truncates toward zero. Normalization for estimating offsets is not
an intensity scaling of saved frames. NumPy/SciPy reconstruction can differ from CUDA interpolation at integer rounding
boundaries; its RMSE is measured separately from the byte-exact NWB round trip.

For each sampled registered frame, neighboring raw pages **within that plane** (`i-1` and `i+1`) are transformed with
the same target-frame offsets. Each available control must have more than ten times the reconstruction RMSE (with a
0.01-unit floor), and the reconstruction RMSE must be at most two intensity units. Failure stops conversion. This is
sampled evidence throughout the session, not a claim that every registered frame was pixel-checked. Neither a
behavior/activity correlation nor a successful NWB validator proves frame identity.

Current reconstruction deliberately refuses bidiphase correction, two-pass registration, unsupported pixel types, and
nonrigid grids with only one block on an axis. Empty planes, missing files, frame-count mismatches and invalid masks
also stop conversion. Zero-ROI planes retain an empty segmentation and their source arrays without inventing traces.

## NWB representation and inventory

`processing/ophys` contains one `PlaneSegmentation` linked to one `ImagingPlane` per plane, all ROIs in source order,
`iscell` and `cell_probability` columns, and `Fluorescence`, `Neuropil` and `Deconvolved` interfaces. Deconvolution values
are activity estimates, not measured spike counts. Masks use `(x=column, y=row, weight)`. Summary images transpose
`(row,column)` to `(x,y)`, matching the raw `TwoPhotonSeries` convention. `summary_image_coordinates` records each
image's full-frame origin and dimensions. Subtract its origin before indexing a cropped image with full-frame mask
coordinates; images are not silently padded or masks shifted.

Registration series preserve `xoff`, `yoff`, `corrXY`, `xoff1`, `yoff1`, `corrXY1`, `badframes` and `badframes0`, each with
its plane's timestamps. `meanImg`, `meanImgE`, `refImg`, `meanImg_crop`, `max_proj` and `Vcorr` are also standard images.
The source inventory records the original relative path, SHA256 and scratch dataset of every nonmovie artifact.
`suite2p_settings` also exposes every original setting and version as text, including the unchanged bidiphase decision.

Every source artifact except per-plane `data.bin` is stored **byte-for-byte** as compressed `uint8` scratch data. This
includes all NPY files, every `Fall.mat`, root metadata and logs, and the whole `combined/` folder. Combined is retained
only as a source artifact; its padded traces never supply response-series samples. This costs space but preserves
fields without an NWB semantic equivalent, native dtypes and even duplicate fields that disagree between files.
To recover an artifact, write the named scratch dataset's bytes to the inventory's relative path. Object NPY files
still need a trusted pickle reader. The adjacent report inventories every NPY dictionary key and every `stat` key.

The measured run's `stat` keys will be reported below. Each also has a `stat_*` text column containing its original
scalar or array value as JSON with NaN literals; the artifact retains the exact native dtype and NaN representation.
All `F`, `Fneu`, `spks`, `iscell`, registration/detection dictionaries, `ops`, `db`, settings/version, `zcorr`,
`regPC`, `tPC`, `regDX`, and processing times remain in their native source artifacts even when also represented
semantically. `data.bin` is the only excluded data: its registered pixels remain external, and its sampled frame
identity evidence is stored as `scratch/suite2p_frame_mapping`.

Before PyNWB sees masks, flags or traces, the converter checks their numeric conversion with the existing exact
coercion rules. No new missing sentinel is introduced. Float32 conversion has tolerance zero. Tests run with
`error::UserWarning`. A separate verification path rereads NPY files, TIFF descriptions and HDF5 datasets; it checks
behavior, provenance, times, traces, masks, stat values, images, registration values and every archived artifact byte.
A mismatch raises rather than being rounded away.

## Session measurements

Measurements and validation results will be recorded here after the clean run and final conversion complete.

### Why the pre-existing run was not reused

The initial plane-0/frame-0 reconstruction had RMSE **607.127904** intensity units; mapping its next same-plane raw
page with the same offsets gave **662.818332**. Suite2p's own transform on CPU independently gave **607.127844** for
the correct raw page, while the independent reconstruction differed from that CPU result by only **0.015729** RMSE.

The saved log starts with `FOUND BINARIES AND DBS` and then runs registration on every plane. This proves that it
registered existing binaries without re-reading the TIFFs in that invocation. It does **not** by itself establish the
pixels those binaries contained before that invocation or count previous registrations. Re-registration of already
registered pixels is a plausible explanation, not a demonstrated two-pass history. The saved offsets do not reconstruct
the raw page tested. Its fluorescence traces may still describe its registered movie, but their biological validity
and full timing identity were not established by this failed check. A clean TIFF-based rerun was selected rather than
assigning raw-page timestamps to those results without evidence.

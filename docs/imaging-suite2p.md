# Suite2p results on the behavior clock

The converter takes a trusted Suite2p output folder, the TIFFs of one acquisition, and its ViRMEn log. It reads Suite2p
with NumPy; Suite2p and torch are not converter dependencies. Object NPY files contain pickle and must come from a
trusted run. Nothing uploads.

```
python -m tank_lab_to_nwb.imaging.convert_suite2p LOG.mat SUITE2P_DIR OUTPUT_DIR TIFF [TIFF ...] [--raw-nwb RAW.nwb]
```

The output is `OUTPUT_DIR/staging/suite2p.nwb`, with a report beside it. Organize it using
`virmen.layout.organize_for_dandi(staging, dandiset, description="suite2p")` before local DANDI validation. It can share a
dandiset with the raw file, named with `description="rawtiff"`: the two get the same subject and session and differ by `desc-`.

The two files are tied together in three ways. Both carry the same `subject_id` and `session_id`, and the same `acquisition.id`
in `processing/ophys/scanimage_alignment`. With `--raw-nwb`, this converter also checks that the raw file has that `acquisition.id` and
exactly the same frame times, refusing it otherwise, and records its NWB identifier and SHA256 in
its `companion.rawtiff.*` rows. The raw file can't name this one, because it's written first.

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

The same samples tie the traces to those frames. `F.npy` at frame `i` must equal what Suite2p's extraction gives on registered frame
`i`: each ROI's normalized `lam` over its pixels, leaving out `overlap` pixels unless `allow_overlap` is set. The RMSE must be at
most 0.01, and registered frames `i-1` and `i+1` must be more than ten times worse. A sampled frame where no ROI has a value isn't
measured. `Fneu` and `spks` aren't recomputed; they're tied to `F` only by order and length.

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

The measured run's `stat` keys are listed below. Each also has a `stat_*` text column containing its original
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

The clean run used Suite2p **1.1.0**, Python **3.12.15**, torch **2.15.0+cu132**, the GTX 1660 Ti, and the supplied
configuration. Every per-plane settings dictionary matches that configuration without a changed field. Only output
paths were redirected. The run started from all **45 TIFFs**, not existing binaries, and completed successfully in
**1,382.52 s**, including TIFF conversion. GPU compute was idle before launch; over 800 GB of disk was free.

The TIFFs contain **89,508 signed int16 frames**, each 300 rows by 512 columns. No uint16 halving applies: plane-0
frame-0 reconstruction with unscaled pixels has RMSE **0.024206**, while halving these signed pixels gives
**292.612127**. Both the interpolation and the final truncation to int16 are included in the reconstruction.

The **675 samples** (135 per plane) include every file's first, middle and last frame. All passed the mapping gate;
all **1,340 available ±1 controls** were worse. Matching RMSE spans **0.010520–0.026639** intensity units; controls span
**754.697–939.568**. The smallest same-sample control/match ratio is **31,177.9**. These results support Suite2p frame
`i` of plane `p` mapping to `acquisition.plane_frames(p)[i]` at the sampled positions throughout this acquisition.
The intervening unsampled registered pixels were not individually checked.

| Plane | Frames | All ROIs | Maximum matching RMSE | Minimum −1 control | Minimum +1 control |
|---|---:|---:|---:|---:|---:|
| 0 | 17,902 | 327 | 0.025643 | 771.649 | 754.697 |
| 1 | 17,902 | 337 | 0.026639 | 798.140 | 799.994 |
| 2 | 17,902 | 326 | 0.025259 | 801.822 | 794.642 |
| 3 | 17,901 | 326 | 0.026393 | 813.851 | 808.412 |
| 4 | 17,901 | 348 | 0.026270 | 827.527 | 825.642 |

Every plane has 96 nonrigid blocks. `nframes`, all trace and offset lengths, both badframe vectors and TIFF plane
counts agree; both badframe vectors are all false. For the first 44 files, every plane has 400 samples per file.
The last file contributes 302 samples to planes 0–2 and 301 to planes 3–4. No sample is invented for the partial volume.
`combined/` NPY traces, classifications, all stat keys, metadata and tiled images match the per-plane-derived view,
including the final zero padding. Its MATLAB variables match those NPY values after MATLAB singleton/empty-value
representation is accounted for. The extra combined `redcell` is a 1,664×2 all-zero placeholder; each per-plane
`redcell` is empty (0×0). Combined holds no additional measured signal or ROI field.

All 19 per-plane `stat` keys are preserved: `aspect_ratio`, `compact`, `footprint`, `lam`, `med`, `mrs`, `mrs0`, `npix`,
`npix_norm`, `npix_norm_no_crop`, `npix_soma`, `overlap`, `radius`, `skew`, `snr`, `soma_crop`, `std`, `xpix`, `ypix`.
Combined adds the derived `iplane`. The report inventories all native dictionary and MATLAB variable names.
**71 nonmovie artifacts** are byte-exact in NWB scratch. The only omitted artifacts are the five registered `data.bin`
movies (27,496,857,600 bytes total), which remain external. Every nonmovie Suite2p field is retained in the archive.

All 675 samples also passed the trace link: `F` at frame `i` matches what's recomputed from registered frame `i` with RMSE
**0.0000256–0.0000674**, and all **1,340** neighboring controls are **108.19–199.18**. These are intensity RMSEs, not time errors.

The clock fit used **73,413 packets**, excluded **180 first-iteration packets**, and rejected **382** additional packets
(0.5216% of eligible packets). Slope is **0.9999948645516386** (−5.13545 ppm), intercept **11.027272188700522 s**,
and shifted clock offset **11.02732920000016 s**. Kept residual standard deviation is **1.051158 ms**;
all-packet residual p1/p99 are **−1.785986 / +3.499372 ms**. These are packet scatter, not frame-time error.
`session_start_time` moved back **10.244864 s**. Frame times span **0.000000381022–1,782.896587029 s**.
The first plane times are **0.000000381, 0.019918908, 0.039837436, 0.059755968, 0.079674495 s**; each response series
round-trips exactly against its own fitted TIFF times. No behavior-level activity sanity check was used as timing proof.

The file has **1,949,976,431 bytes**, SHA256 `0ee1ab3bea6db23c3fbd3621f2a1cb446658beda1d0b354e11033bdf5de7215f`. The raw file it goes
with has **18,266,980,032 bytes**, SHA256 `d923c30b6a7e99f82226bfa47557bb7f9e052a9892bdbc485bf9550336ecf369`, the value recorded in
`companion.rawtiff.sha256`. Both are organized into one local stub dandiset, as `desc-rawtiff` and `desc-suite2p`, and these serial checks ran
there:

- `pynwb.validate`: **[]** on both files.
- NWB Inspector **0.7.2**, via uvx on Python 3.13 with `--config dandi`: exit **0** on both, with **6** best-practice suggestions on the
  raw file and **7** on this one, and no errors, critical findings or best-practice violations. The suggestions are missing keywords,
  the `scanimage` and `in_progress_trial` module names, the all-NaN `easyBlock` and `trialID` columns, and single-row tables: epochs
  on both, and `companion_files` here.
- DANDI **0.82.0** `validate` on the dandiset: exit **0**, **No errors found.**
- Full suite at this conversion's code: **478 passed, 9 skipped**. UserWarnings are errors in tests.

The Inspector uvx environment pinned `h5py==3.13.0`, because the newest h5py has no wheel for this platform. No upstream package was changed.

`registration.do_bidiphase=False` stays as configured on this bidirectionally scanned acquisition, by decision. Indicator, location `root`, and NaN wavelengths retain the raw converter's placeholders. The pending behavior
field decisions were not changed. Biological validity of the ROI detection and deconvolution was not assessed.
No data was uploaded.


### Why the pre-existing run was not reused

The initial plane-0/frame-0 reconstruction had RMSE **607.127904** intensity units; mapping its next same-plane raw
page with the same offsets gave **662.818332**. Suite2p's own transform on CPU independently gave **607.127844** for
the correct raw page, while the independent reconstruction differed from that CPU result by only **0.015729** RMSE.

The saved log starts with `FOUND BINARIES AND DBS` and then runs registration on every plane. Further measurement
found a mixed history in plane 0. At 98 of 135 samples (first, middle and last frame of each of the 45 files), applying
the old saved transform to the fresh registered frame reproduces the old movie with RMSE **0.00920–0.01976**. At the
remaining 37 samples, the old transform applied directly to raw reproduces it with RMSE **0.01693–0.02615**. None of
these samples is unexplained by those alternatives. The transition is between adjacent zero-based frames **12,999 and 13,000**:

| Plane-0 frame | Old transform applied to raw: RMSE | Fresh registration, then old transform: RMSE |
|---|---:|---:|
| 12,999 | 713.97584 | 0.01768 |
| 13,000 | 0.01993 | 709.42511 |

These are intensity RMSEs, not time errors. The sampled early pixels exhibit two registration transforms; the later
ones exhibit one. An interrupted in-place first registration through the 13,000-frame boundary (26 batches of 500),
followed by registration of the partially modified binary, explains the pattern. That execution history is an
inference: the earlier process log is unavailable, and every old frame was not checked. At frame 0 of old planes 1–4,
the direct raw reconstruction has RMSE **0.01052–0.02447**; those first frames do not show the plane-0 problem.

The old fluorescence files remain untouched and were not declared corrupt. Sampled frame identity is supported by
the one- or two-stage reconstructions, but mixed resampling changes plane 0's processing history. Its ROI and trace
biological validity were not assessed. The saved offsets alone do not describe its early raw-to-registered mapping.
A clean TIFF-based rerun was selected to establish one consistent registration history for this conversion.

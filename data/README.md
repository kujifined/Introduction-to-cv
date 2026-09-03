# Data: NEU Surface Defect Database (NEU-CLS)

## Redistribution boundary

Raw images and the downloaded archive are local-only. The official page does
not provide a standard machine-readable licence in the dataset description, so
public download must not be interpreted as permission to redistribute. Review
the source terms, retain attribution, and commit only the small provenance,
manifest, hash, review, and split artifacts needed for reproducibility.

## Official source and task scope

Use the **NEU-CLS** image-only download linked from the
[official NEU surface defect database page on Northeastern University's faculty
portal](https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm).
The official description states that NEU-CLS has:

- 1,800 grayscale images;
- resolution 200 × 200 pixels;
- six hot-rolled steel surface-defect classes;
- 300 source images per class; and
- substantial within-class variation and visual similarity between classes.

NEU-CLS contains **no normal/defect-free class**. It supports classification of
already cropped, known-defect images. It cannot by itself evaluate detection,
localisation, anomaly screening, or a production false-alarm rate on normal
steel. NEU-DET is an annotated detection variant and is not interchangeable
with NEU-CLS for this primary experiment.

## Recorded acquisition

The version-controlled acquisition record is
`data/manifests/neu_cls_source_v1.json`. For the audited local copy it records:

| Field | Recorded value |
|---|---|
| landing page | official URL above |
| download | official page's Google Drive file |
| archive | `NEU-CLS.rar` |
| archive size | 50,733,617 bytes |
| measured archive SHA-256 | `a85bcc6bfa1a40e76f28079e135694aaf5f6a4ee01203f7eaacff0ebafaa96f8` |
| download time | `2026-09-03T22:23:50.362021+00:00` |
| redistribution | false |

The SHA-256 is a fingerprint measured for the downloaded artifact, not a claim
that the publisher supplied an official checksum. Local downloader metadata is
also written beside the ignored archive as
`data/archives/NEU-CLS.rar.metadata.json`.

One-time acquisition command:

```bash
uv sync --extra data
uv run python scripts/download_neu_cls.py \
  --acknowledge-source-terms \
  --extract
```

The downloader refuses to overwrite an existing archive or extract into a
non-empty destination.

## Local and version-controlled layout

```text
data/
├── README.md
├── archives/                           # ignored; local only
│   ├── NEU-CLS.rar
│   └── NEU-CLS.rar.metadata.json
├── raw/                                # ignored; local only
│   └── NEU-CLS/
│       └── NEU-CLS/
│           ├── Cr_1.bmp ...
│           ├── In_1.bmp ...
│           ├── Pa_1.bmp ...
│           ├── PS_1.bmp ...
│           ├── RS_1.bmp ...
│           └── Sc_1.bmp ...
└── manifests/
    ├── neu_cls_source_v1.json
    ├── neu_cls_v2.csv
    ├── neu_cls_v2.audit.json
    ├── neu_cls_v2_near_duplicates.csv
    └── neu_cls_reviewed_groups_v2.csv
```

Raw files remain byte-for-byte unchanged. Manifest paths such as
`NEU-CLS/Cr_1.bmp` are relative to `data/raw/NEU-CLS`.

## Completed archive audit

The strict audit of the official archive found:

| Property | Audited value |
|---|---:|
| source rows | 1,800 |
| decodable grayscale (`L`) images | 1,800 |
| 200 × 200 images | 1,800 |
| nominal source rows per class | 300 |
| extra byte-identical copies | 1 |
| active images after quarantine | 1,799 |
| active patches | 299 |
| active images in each other class | 300 |

`Pa_101.bmp` and `Pa_105.bmp` have the same SHA-256. The lexicographically
first file, `Pa_101.bmp`, remains active; `Pa_105.bmp` is preserved in the
manifest with `status=duplicate_quarantined` and `split=excluded`. Nothing was
silently deleted or re-encoded.

## Near-duplicate review and group decisions

The final review used a 64-bit DCT pHash Hamming-distance threshold of **12**.
Perceptual-hash matches were treated as candidates, never as automatic proof.
Thirteen pairs were reviewed, leaving no unresolved candidate and no unresolved
cross-split candidate:

- nine pairs were marked `not_duplicate` after side-by-side and translation
  review;
- four pairs had pixel-identical overlap under translation and were assigned a
  shared group before the final split.

Confirmed shifted groups:

| Paths | Identical overlap after translation | Final split |
|---|---|---|
| `PS_171.bmp`, `PS_292.bmp` | 198 × 198, `dy=-2`, `dx=-2` | validation |
| `Pa_199.bmp`, `Pa_201.bmp` | 194 × 189, `dy=-6`, `dx=-11` | validation |
| `Pa_2.bmp`, `Pa_5.bmp` | 191 × 198, `dy=-9`, `dx=2` | validation |
| `In_60.bmp`, `In_62.bmp` | 193 × 190, `dy=-7`, `dx=-10` | validation |

The two-pixel diagonal relation between `PS_171` and `PS_292` is exact over the
overlapping 198 × 198 region, so treating them as independent across splits
would be leakage. In contrast, `In_188` and `In_210` (pHash distance 8) show
different defect structures after review. They remain separate groups and are
both assigned to training.

The evidence and decisions live in:

- `neu_cls_v2_near_duplicates.csv` — all candidates and review notes;
- `neu_cls_reviewed_groups_v2.csv` — confirmed group membership; and
- `neu_cls_v2.audit.json` — counts, threshold, resolutions, and review summary.

Unknown original strip/acquisition identities may still exist beyond these
recoverable relationships. The project states this as a limitation rather than
claiming that every active crop is independently sourced.

## Final fixed manifest

`data/manifests/neu_cls_v2.csv` is the authority for all registered training,
validation, clean-test, and robustness evaluation. It was generated with:

- reviewed group map: `neu_cls_reviewed_groups_v2.csv`;
- split seed: `42`;
- ratios: 70% train, 10% validation, 20% test;
- stratification by class; and
- group-level assignment so no active SHA-256 or reviewed group crosses a
  split.

Final active counts:

| Class | Train | Validation | Test | Active total |
|---|---:|---:|---:|---:|
| crazing | 210 | 30 | 60 | 300 |
| inclusion | 210 | 30 | 60 | 300 |
| patches | 209 | 30 | 60 | 299 |
| pitted surface | 210 | 30 | 60 | 300 |
| rolled-in scale | 210 | 30 | 60 | 300 |
| scratches | 210 | 30 | 60 | 300 |
| **Total** | **1,259** | **180** | **360** | **1,799** |

The file contains one additional excluded provenance row for `Pa_105`, so the
CSV itself has 1,800 data rows.

The implemented loader requires these columns:

```text
path,class_name,class_index,split,sha256,group_id,status
```

The full manifest also carries source/inventory fields including width, height,
mode, file size, original relative path/label, and perceptual hash. Permitted
statuses are `ok` and `duplicate_quarantined`; only the latter may use
`split=excluded`.

## Reproducing and verifying preparation

The final artifact is frozen and `neu-prepare` refuses to overwrite it. To
independently reproduce the manifest, use a new temporary destination and the
reviewed group map:

```bash
audit_dir="$(mktemp -d)"
uv run neu-prepare \
  --data-root data/raw/NEU-CLS \
  --manifest "${audit_dir}/neu_cls_v2.csv" \
  --seed 42 \
  --near-duplicate-threshold 12 \
  --group-map data/manifests/neu_cls_reviewed_groups_v2.csv
cmp "${audit_dir}/neu_cls_v2.csv" data/manifests/neu_cls_v2.csv
```

Verify the local files against all manifest hashes before training:

```bash
uv run python -c "from industrial_surface_defects.data import verify_manifest_files; verify_manifest_files('data/manifests/neu_cls_v2.csv', 'data/raw/NEU-CLS'); print('manifest and raw files: OK')"
```

Never create a new split because its validation or test behaviour is more
favourable. If a future audit changes a grouping decision, create a new
versioned manifest and report the change.

## Model preprocessing boundary

Raw images are immutable. At load time the pipeline:

1. decodes grayscale;
2. applies only the augmentation registered for the training cell;
3. replicates the grayscale channel to three channels;
4. resizes to 224 × 224; and
5. applies ImageNet normalisation.

Validation and clean test receive no stochastic augmentation. The robustness
suite applies nine deterministic corruption types at severity levels 1, 2,
and 3 to the same locked test images. Those 27 views per image are paired
derivatives, not new independent observations.

## False-positive interpretation

For any defect class `c`, the evaluator reports an FP when the true class is
not `c` but the predicted class is `c`. Every wrong multiclass prediction is an
FP for its predicted defect type and an FN for its true defect type. Because
NEU-CLS has no normal class, these counts must not be presented as industrial
normal-surface false alarms.

## Citation

Use the citation requested by the dataset page:

> K. Song and Y. Yan, “A noise robust method based on completed local binary
> patterns for hot-rolled steel strip surface defects,” *Applied Surface
> Science*, vol. 285, pp. 858–864, 2013.
> [https://doi.org/10.1016/j.apsusc.2013.09.002](https://doi.org/10.1016/j.apsusc.2013.09.002)

```bibtex
@article{song2013noise,
  title     = {A noise robust method based on completed local binary patterns
               for hot-rolled steel strip surface defects},
  author    = {Song, Kechen and Yan, Yunhui},
  journal   = {Applied Surface Science},
  volume    = {285},
  pages     = {858--864},
  year      = {2013},
  publisher = {Elsevier},
  doi       = {10.1016/j.apsusc.2013.09.002}
}
```

Also cite the official landing page and record its access date in the report.

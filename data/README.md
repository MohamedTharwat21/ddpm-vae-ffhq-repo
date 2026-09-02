# Data

This directory is where local FFHQ image data is expected to live. **No image data
is included in this repository.**

## Expected layout

Any directory containing FFHQ images (any nesting depth, common image extensions -
`.png`, `.jpg`, `.jpeg`, `.webp`) works, as long as it has at least 50,000 images
(the `min_images` default in `ffhq_repo.data.dataset.find_data_root`). For example:

```text
data/
└── ffhq/
    ├── 00000.png
    ├── 00001.png
    ├── ...
```

or the original Kaggle-style nested folders (`images1024x1024/00000/00000.png`, ...)
- the dataset code recursively discovers images (`discover_all_images`), so the exact
nesting does not matter.

## Pointing a config at your data

Set one of, in the config's `data:` section:

- `data_root: ./data/ffhq` - use this exact directory.
- `data_root_search: ./data` (with `data_root: null`) - auto-detect the first
  subdirectory under this root containing at least 50,000 images
  (`ffhq_repo.data.dataset.find_data_root`).

Every supplied original config used `data_root_search: /kaggle/input` (a Kaggle
convention) - this is preserved as a comment/default in the YAML configs for
historical accuracy, but is fully overridable and not hardcoded into any Python
source file.

## Deterministic split

On first use, a 60,000 / 10,000 train/validation split is generated
(`generate_or_load_split`) using a SHA-256 hash of each image's relative path plus
`data.split_seed` (`1234` in every supplied config), and written to
`data.split_file`. Re-running against the same `data_root` and `split_seed`
reproduces the same split even if the split file itself is not shipped alongside the
code (which it is not, in this repository).

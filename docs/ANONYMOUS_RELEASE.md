# Preparing the Anonymous Upload

The upload root is the directory containing `README.md`, `pyproject.toml`,
`mora/`, and `configs/`. The prepared ZIP contains this tree under `MORA/`.

To check and rebuild after an edit:

```bash
python scripts/build_release.py --check-only
python scripts/build_release.py --output dist/MORA_anonymous.zip
```

Existing ZIP files are not overwritten. Choose a new filename for a new
snapshot. Add `--deny-token` for any author, institution, account, or project
identifier that should be checked locally. Do not commit private token lists.

Do not upload the parent research directory or its Git history. The release
builder excludes local runs, weights, datasets, temporary files, bytecode, and
build metadata. It also refuses symlinks that could point back into local
storage. Keep third-party attribution when adding upstream code later.

The source package and a built Python wheel serve different purposes: upload
the source ZIP for code review. Data and backbone checkpoints, if separately
released, need their own license and anonymity review.

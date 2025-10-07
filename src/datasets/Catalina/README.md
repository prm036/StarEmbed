## Catalina Surveys (CSDR1 Periodic Variable Stars)

The Catalina Northern Periodic Variable Star Catalog (CSDR1 PVS) with DR2 photometry. Primary reference: Drake et al., 2014.

- **Homepage**: `http://nesssi.cacr.caltech.edu/DataRelease/`
- **Original catalog**: `http://nesssi.cacr.caltech.edu/DataRelease/CatalinaVars.tbl`
- **Photometry (DR2)**: `http://nesssi.cacr.caltech.edu/DataRelease/AllVar.phot.gz`

See the standardized schema, common preprocessing, and evaluation guidance in `datasets/README.md`.


### Download raw data

```bash
# Catalog (CSDR1 periodic variables)
curl -L -o CSDR1_varstars_v2.txt \
  http://nesssi.cacr.caltech.edu/DataRelease/CatalinaVars.tbl

# DR2 photometry (clear band "C"), gzip-compressed
curl -L -o AllVar.phot.gz \
  http://nesssi.cacr.caltech.edu/DataRelease/AllVar.phot.gz

# Decompress and optionally rename for consistency
gunzip -f AllVar.phot.gz
mv -f AllVar.phot Catalina_PVars_DR2.phot

# (Optional) move into repo data directory
mkdir -p ../../../data
mv -f CSDR1_varstars_v2.txt ../../../data/
mv -f Catalina_PVars_DR2.phot ../../../data/
```


## Convert to Hugging Face Dataset

```bash
python Catalina_to_hf.py \
  --lcs_path ../../../data/Catalina_PVars_DR2.phot \
  --cat_path ../../../data/CSDR1_varstars_v2.txt \
  --output_dir ../../../data/catalina \
  --num_proc 4 \
  --max_shard_size 100MB
```

- **Output**: a `datasets` on-disk directory at `../../../data/catalina` containing Arrow shards and `dataset_info.json`.
- The conversion removes invalid rows, harmonizes coordinates, and maps classes to a consistent taxonomy (Catalina-specific details below).


## Class taxonomy (Catalina)

Mapping applied during conversion from Drake et al. 2014:

- 1: EW
- 2: EA
- 3: Beta_Lyrae
- 4: RRab
- 5: RRc
- 6: RRd
- 7: Blazhko
- 8: RS CVn
- 9: ACEP
- 10: Cep-II
- 11: HADS
- 12: LADS
- 13: LPV
- 14: ELL
- 15: Hump
- 16: PCEB
- 17: EA_UP


## Catalina-specific preprocessing

- Drop photometry rows with negative `mjd`.
- Filter catalog entries with missing periods (`P[d] == \nodata`).
- Map integer `class` to `class_str`.


## License and terms of use

Respect the Catalina Surveys data release policies. See the release portal: `http://nesssi.cacr.caltech.edu/DataRelease/`.


## Citation

If you use this dataset, please cite the original survey and this repository:

- Drake, A. J., et al. 2014. “The Catalina Surveys Periodic Variable Star Catalog.” ApJS, 213, 9. arXiv:1405.4290.
- This repository: please cite per the project’s main `README.md`.

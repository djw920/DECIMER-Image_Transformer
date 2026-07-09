---
title: "DECIMER Structure Extraction — Danková et al. 2025"
paper: "Discovery of De Novo Macrocycle Inhibitors of Histone Deacetylase 11"
journal: "JACS Au 2025, 5, 1299–1307"
doi: "10.1021/jacsau.4c01148"
extracted: "2026-07-09"
extractor: "DECIMER 2.8.0 (EfficientNet-V2 + Transformer) + decimer-segmentation (Mask R-CNN)"
---

# DECIMER Extraction Summary

## Source

**Danková D., Nielsen A.L., Zarda A., et al.**
*Discovery of De Novo Macrocycle Inhibitors of Histone Deacetylase 11.*
JACS Au 2025, 5, 1299–1307.

9-page open-access paper reporting phage-display-derived macrocyclic HDAC11 inhibitors,
profiled against the full HDAC family.

---

## Extraction pipeline

1. **PDF → images**: PyMuPDF rendered all 9 pages at 300 DPI (2531 × 3338 px each).
2. **Segmentation**: `decimer-segmentation` (Mask R-CNN) detected individual structure
   bounding boxes on each page image.
3. **Recognition**: DECIMER 2.8.0 (`predict_SMILES`) predicted SMILES for each crop.
4. **Validation**: datamol / RDKit parsed, sanitized, and standardized all SMILES;
   `compute_many_descriptors` computed properties.
5. **QC filter**: entries removed if SMILES was unparseable; entries flagged REVIEW
   if MW < 150 Da or > 2000 Da.

---

## Results

| Stage | Count |
|---|---|
| Structures segmented | 44 |
| Valid SMILES | 40 (91%) |
| Removed (unparseable) | 4 |
| PASS (MW 150–2000 Da) | 30 |
| REVIEW (outlier MW) | 10 |

### Pages with highest structure density
- **page_02** — 15 structures (main compound series, Figures 2–3)
- **page_03** — 16 structures (SAR table / analogue panel)

### Property distribution (PASS set, n = 30)

| Property | Range | Mean |
|---|---|---|
| MW | 157–1698 | 440 |
| clogP | −7.3 to 9.6 | 3.1 |
| HBA | 0–18 | 4.8 |
| HBD | 0–9 | 1.6 |
| TPSA | 0–267 | 63 |
| Stereocenters | 0–7 | 1.2 |

---

## Known limitations

| Issue | Affected entries | Notes |
|---|---|---|
| Unparseable SMILES | 4 | Truncated PEG chain (token limit); `[R19a]` / `[Y12]` Markush atoms; cyclopropane ring-closure error |
| High-MW entries flagged REVIEW | 5 | Phage-display macrocycle libraries (pages 6–7); DECIMER truncates at sequence limit |
| Low-MW entries flagged REVIEW | 5 | Solvent molecules, protecting groups, or reagents segmented in error |
| Stereochemistry accuracy | All macrocycles | DECIMER frequently misses or inverts stereocenters in macrocycles — **cross-check manually before use in modelling** |

---

## Files in this folder

| File | Description |
|---|---|
| `dankova2025_smiles.csv` | Raw DECIMER output — all 44 entries, one per segmented structure |
| `dankova2025_verified.csv` | Cleaned dataset — 40 valid entries with canonical SMILES, InChIKey, properties, and QC flags |
| `dankova2025_spotcheck.html` | Side-by-side visual report: source crop vs. DECIMER 2D depiction for all 44 entries |

### `dankova2025_verified.csv` columns

`page` · `structure` · `smiles_raw` · `smiles_canonical` · `inchikey` ·
`mw` · `clogp` · `hba` · `hbd` · `tpsa` · `n_rot_bonds` · `n_rings` ·
`n_stereo_centers` · `qc_flag` · `source_crop` · `decimer_elapsed_s` · `verified_at`

---

## Recommended next steps

1. Open `dankova2025_spotcheck.html` and manually verify the PASS entries from
   pages 2–3 (the key macrocycle inhibitor structures).
2. Cross-reference compound names from the paper's figures against the canonical SMILES
   using the InChIKey column for PubChem/ChEMBL lookups.
3. Correct stereochemistry for entries that will be used in docking or pharmacophore work.
4. Merge the verified SMILES into the HDAC11 inhibitor landscape datasheet
   (`HDAC11_IC50_extracted_table_with_reference_urls`).

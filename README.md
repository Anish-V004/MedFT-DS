# MedFT Adverse Event Cardiology Dataset Pipeline

This project provides an automated, end-to-end data pipeline to extract, balance, and clean cardiology-related adverse drug event (ADE) reports from two major clinical sources: the **BioDEX-ICSR** literature dataset and the **openFDA FAERS** direct reporting API. 

The resulting outputs are formatted specifically for downstream LLM fine-tuning (e.g., training models for structured extraction, clinical case note generation, or adverse event prediction).

---

## 🎯 Project Focus & Design

To prepare a high-quality dataset for LLM fine-tuning, the pipeline enforces three core design principles:
1. **Clinical Focus (Cardiology)**: Filters reports using standard Medical Subject Headings (MeSH category `C14` - Cardiovascular Diseases) and a highly specific regex vocabulary (with negative lookbehinds to ignore terms like "non-cardiac").
2. **Clinical Representation (Balanced openFDA FAERS)**: To avoid dataset bias from extremely common conditions (e.g. general hypertension), the openFDA pipeline divides cardiology adverse events into 6 broad clinical categories and fetches exactly 500 unique reports for each.
3. **Clinical Column Selection**: Dropping academic/publishing metadata (DOIs, PMIDs, authors, journal names) and administrative database tags (safety report IDs, submitter dates), saving over **55%** in token storage/processing costs.

---

## 📁 Directory Structure

The project workspace is organized into a modular structure:

```
MedFT/
├── pyproject.toml              # Project dependencies and configs managed by uv
├── main.py                     # Root pipeline entry point
├── README.md                   # Project documentation
├── scripts/
│   ├── __init__.py             # Package initializer
│   ├── prepare_cardio_subset.py  # Stage 1: BioDEX-ICSR downloader and filter
│   ├── fetch_fda_cardio.py       # Stage 2: Balanced openFDA downloader
│   └── preprocess_datasets.py    # Stage 3: Clinical feature preprocessor
└── data/                       # Directory containing all output files
    ├── biodex_cardio_clinical.csv         # Cleaned BioDEX CSV for fine-tuning
    ├── biodex_cardio_clinical.jsonl        # Cleaned BioDEX JSONL for fine-tuning
    ├── fda_cardio_clinical.csv             # Cleaned openFDA CSV for fine-tuning
    ├── fda_cardio_clinical.jsonl            # Cleaned openFDA JSONL for fine-tuning
    ├── biodex/
    │   ├── raw/
    │   │   ├── cardio_dataset.csv
    │   │   ├── cardio_dataset.jsonl
    │   │   └── cardio_subset/              # HF local serialized dataset split
    │   └── processed/
    │       ├── biodex_cardio_clinical.csv
    │       └── biodex_cardio_clinical.jsonl
    └── openfda/
        ├── raw/
        │   ├── fda_cardio_dataset.csv
        │   └── fda_cardio_dataset.jsonl
        └── processed/
            ├── fda_cardio_clinical.csv
            └── fda_cardio_clinical.jsonl
```

---

## ⚙️ Processing & Filtering Logic

### Stage 1: BioDEX-ICSR literature extraction
- Concatenates the Hugging Face `BioDEX/BioDEX-ICSR` dataset splits (`train`, `validation`, `test`).
- Maps `mesh_terms` to the MeSH Tree categories and retains records matching tree `C14` (Cardiovascular Diseases).
- Uses a fallback regex match targeting specific cardiac/cardiology terms (e.g. `arrhythmia`, `myocardial infarction`, `heart failure`) for records without MeSH terms.
- Implements negative lookbehinds `(?<!non)(?<!non-)` to ignore phrases like "non-cardiac chest pain" or "noncardiac symptoms".

### Stage 2: openFDA FAERS Balanced Extraction
Queries the official FDA adverse drug events API (`https://api.fda.gov/drug/event.json`) to fetch exactly 500 unique reports across each of the following 6 clinical groups:
1. **Infarction / Ischemia**: Myocardial infarction, coronary artery disease, angina pectoris, etc.
2. **Heart Failure**: Heart failure, cardiac failure, cardiogenic shock.
3. **Arrhythmias**: Atrial fibrillation, tachycardia, bradycardia, arrhythmia.
4. **Inflammatory / Infectious**: Myocarditis, pericarditis, endocarditis, pericardial effusion.
5. **Vascular / Blood Pressure**: Hypertension, hypotension, hypertensive crisis.
6. **Structural / Valvular / Arrest**: Cardiomyopathy, cardiac arrest, valve disorders.

### Stage 3: Clinical Pre-processing
Retains only clinical features necessary for LLM reasoning and ChatML construction:
* **BioDEX-ICSR Retained Fields**: `title`, `abstract`, `fulltext_processed`, `target`, `mesh_terms`, `keywords`.
* **openFDA FAERS Retained Fields**: `patient_age`, `patient_age_unit`, `patient_sex`, `seriousness_*` (all flags), `reactions`, `drugs`, `cardio_category`.

---

## 🚀 How to Run the Pipeline

### 1. Prerequisites & Installation
Ensure you have the `uv` package manager installed. The virtual environment will be created automatically.

Add dependencies (if not already set up):
```bash
uv add datasets pandas pyarrow requests
```

### 2. Execute the Pipeline
Run the root pipeline script from the root folder:
```bash
uv run python main.py
```
*Note: The script will automatically clear out the `data/` folder and rebuild the directories fresh.*

#### Optional: API Key configuration
To run larger queries or avoid IP rate-limiting on the openFDA API, obtain a free API key from [open.fda.gov](https://open.fda.gov) and set it as an environment variable before running:
```bash
$env:OPENFDA_API_KEY="YOUR_API_KEY"
uv run python main.py
```

---

## 📊 Loading Datasets in Python

You can easily load the output datasets into pandas DataFrames:

```python
import pandas as pd

# Load BioDEX-ICSR Clinical Dataset
biodex_df = pd.read_json("data/biodex_cardio_clinical.jsonl", lines=True)
print(f"BioDEX-ICSR: Loaded {len(biodex_df)} samples.")

# Load openFDA FAERS Clinical Dataset
fda_df = pd.read_json("data/fda_cardio_clinical.jsonl", lines=True)
print(f"openFDA: Loaded {len(fda_df)} samples.")
print(fda_df.groupby('cardio_category').size()) # Confirms 500 records per category
```

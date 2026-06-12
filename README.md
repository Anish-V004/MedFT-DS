# MedFT Adverse Event Cardiology Dataset Pipeline

This project provides an automated, end-to-end data pipeline to extract, evaluate, and compile cardiology-related adverse drug event (ADE) reports from two major clinical sources: the **BioDEX-ICSR** literature dataset and the **openFDA FAERS** direct reporting API. 

The resulting outputs are formatted as instruction-response pairs (ChatML) specifically for downstream LLM fine-tuning, teaching the model how to perform structured pharmacovigilance (PV) reviews.

---

## 🎯 Project Focus & Design

To prepare a high-quality dataset for LLM fine-tuning, the pipeline enforces several core design principles:
1. **Clinical Focus (Cardiology)**: Filters reports using standard Medical Subject Headings (MeSH category `C14` - Cardiovascular Diseases) and a highly specific regex vocabulary.
2. **Clinical Representation (Balanced openFDA FAERS)**: To avoid dataset bias from extremely common conditions, the openFDA pipeline divides cardiology adverse events into 6 broad clinical categories and fetches exactly 500 unique reports for each.
3. **Reference Safety Information (RSI) Grounding**: Fetches safety labels for suspected drugs from the OpenFDA labeling API so that the LLM is grounded in real medical documentation rather than hallucinating expectedness.
4. **LLM Evaluation (PV Review)**: Automates clinical evaluations using Gemini models to determine Seriousness, Expectedness, and Causality (Naranjo score).
5. **Data Balancing & Synthetic Negatives**: Compiles a final dataset (`pv_safety_review_dataset_3000.jsonl`) that perfectly balances positive cases across Naranjo causality tiers, and explicitly injects 300 synthetic "negative" cases to teach the model how to gracefully handle missing data or non-clinical text.

---

## 📁 Directory Structure

The project workspace is organized into a modular structure:

```
MedFT/
├── pyproject.toml              # Project dependencies and configs managed by uv
├── main.py                     # Stage 1-3: Raw pipeline entry point (Extract & clean)
├── generate_reviews.py         # Stage 4: Orchestrates Gemini API for PV evaluation
├── README.md                   # Project documentation
├── scripts/
│   ├── __init__.py             # Package initializer
│   ├── prepare_cardio_subset.py  # Stage 1: BioDEX-ICSR downloader and filter
│   ├── fetch_fda_cardio.py       # Stage 2: Balanced openFDA downloader
│   ├── preprocess_datasets.py    # Stage 3: Clinical feature preprocessor
│   ├── extract_rsi.py            # Utility: Fetches Reference Safety Information (RSI)
│   ├── compile_dataset.py        # Stage 5: Balances & injects negative cases
│   └── validate_outputs.py       # Utility: Validates dataset schema
└── data/                       # Directory containing all input, intermediate, and output files
```

---

## ⚙️ Processing & Filtering Logic

### Stage 1: BioDEX-ICSR Literature Extraction
- Maps `mesh_terms` to the MeSH Tree categories and retains records matching tree `C14` (Cardiovascular Diseases). Uses fallback regex match for specific cardiac terms.
- Implements negative lookbehinds `(?<!non)(?<!non-)` to ignore phrases like "non-cardiac chest pain".

### Stage 2: openFDA FAERS Balanced Extraction
Queries the FDA API (`https://api.fda.gov/drug/event.json`) to fetch exactly 500 unique reports across 6 distinct clinical groups (Infarction/Ischemia, Heart Failure, Arrhythmias, Inflammatory, Vascular, Structural).

### Stage 3: Clinical Pre-processing
Retains only clinical features necessary for LLM reasoning and ChatML construction, dropping heavy metadata (DOIs, PMIDs) to save token processing costs.

### Stage 4: Reference Safety Information (RSI) & LLM Review
- **RSI Extraction**: Queries the FDA Labeling API (`scripts/extract_rsi.py`) to retrieve the "Adverse Reactions" text for every drug in the dataset.
- **LLM Review**: Uses `generate_reviews.py` to prompt Gemini with the Patient Narrative + RSI. It enforces a strict Pydantic JSON schema to evaluate:
  - **Seriousness**: Based on standard regulatory criteria.
  - **Expectedness**: Compared against the RSI.
  - **Causality**: A step-by-step Naranjo score evaluation.
- **Concurrent Batch Processing**: Features multi-threaded execution utilizing a thread-safe `google-genai` client pool. By providing multiple API keys in the `.env` file, the script distributes rate-limits across all keys simultaneously, processing multiple rows in parallel with intelligent exponential backoff on quota limits.

### Stage 5: Final Dataset Compilation
`scripts/compile_dataset.py` parses the Gemini evaluations, filters out bad responses, and balances the final dataset into 3,000 cases:
- 900 Definite/Probable Cases
- 900 Possible Cases
- 900 Doubtful Cases
- 300 Synthetic Negative Cases (Missing drug, missing event, or administrative noise).

---

## 🚀 How to Run the Pipeline

### 1. Prerequisites & Installation
Ensure you have the `uv` package manager installed. The virtual environment will be created automatically.

Add dependencies (if not already set up):
```bash
uv sync
```

### 2. Environment Variables
Create a `.env` file in the root directory by copying `.env.example`:
```bash
cp .env.example .env
```
Fill in your API keys in the `.env` file:
- `GEMINI_API_KEYS`: A comma or semicolon-separated list of Gemini API keys for rate limit rotation.
- `OPENFDA_API_KEY`: (Optional) Your openFDA API key for higher rate limits during data fetching.

### 3. Execute the Pipeline

#### Step A: Fetch & Preprocess Raw Datasets
Run the root pipeline script to download and clean the BioDEX and openFDA datasets:
```bash
uv run python main.py
```
*Note: This script will automatically clear out the `data/` folder and rebuild the directories fresh.*

#### Step B: Extract Reference Safety Information (RSI)
Before generating reviews, extract the RSI for the drugs in the datasets:
```bash
uv run python scripts/extract_rsi.py
```

#### Step C: Generate PV Reviews using Gemini
Run the orchestrator script to query Gemini and generate clinical evaluations. You can do a test run first, or a full run.
```bash
# Test run (10 samples)
uv run python generate_reviews.py

# Full run on both datasets
uv run python generate_reviews.py --full-run

# Run only BioDEX dataset
uv run python generate_reviews.py --full-run --biodex

# Run only openFDA dataset
uv run python generate_reviews.py --full-run --openfda
```

#### Step D: Compile the Final Balanced Dataset
Once the LLM extraction is complete, compile the final `pv_safety_review_dataset_3000.jsonl` containing positive cases and synthetically injected negative cases.
```bash
uv run python scripts/compile_dataset.py
```

#### Step E: Validate Outputs
Verify that the output datasets strictly conform to the expected ChatML schema:
```bash
uv run python scripts/validate_outputs.py
```

---

## 📊 Loading the Final Dataset in Python

You can easily load the final generated ChatML dataset into a pandas DataFrame:

```python
import pandas as pd

# Load the Golden Fine-Tuning Dataset
df = pd.read_json("data/pv_safety_review_dataset_3000.jsonl", lines=True)
print(f"Loaded {len(df)} samples for fine-tuning.")

# Print the first instruction-response pair
print(df.iloc[0]['messages'][1]['content'])  # User Prompt
print(df.iloc[0]['messages'][2]['content'])  # Assistant Response
```

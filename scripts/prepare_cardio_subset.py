import os
import re
import sys
import pandas as pd
from datasets import load_dataset

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# URL for MeSH tree numbers hierarchy (curated by community/dhimmel)
MESH_TREE_URL = 'https://raw.githubusercontent.com/dhimmel/mesh/gh-pages/data/tree-numbers.tsv'
DATASET_ID = 'BioDEX/BioDEX-ICSR'
OUTPUT_DIR = os.path.join('data', 'biodex', 'raw')

def download_mesh_c14_ids():
    """Downloads the MeSH hierarchy and extracts all IDs under the C14 (Cardiovascular Diseases) tree."""
    print(f"Fetching MeSH tree numbers from: {MESH_TREE_URL}")
    try:
        df = pd.read_csv(MESH_TREE_URL, sep='\t')
        c14_df = df[df['mesh_tree_number'].str.startswith('C14')]
        c14_ids = set(c14_df['mesh_id'].unique())
        print(f"Successfully loaded MeSH terms. Found {len(c14_ids)} unique MeSH IDs under the C14 tree.")
        return c14_ids
    except Exception as e:
        print(f"Warning: Failed to fetch MeSH tree from URL ({e}).")
        print("Falling back to a curated list of cardiology MeSH IDs.")
        return {
            'D002318', 'D006331', 'D001145', 'D006333', 'D009203', 'D000787',
            'D009205', 'D010493', 'D004696', 'D006973', 'D007022', 'D013927',
            'D004554', 'D001281'
        }

def get_cardio_regex():
    """Compiles a refined regular expression targeting specific cardiology terms while avoiding common false positives."""
    cardio_keywords = [
        # General terms
        r'cardiac', r'cardiovascular', r'cardiology',
        
        # Anatomy / Structure
        r'myocardial', r'myocardium', r'pericardium', r'endocardium', r'ventricle', r'ventricular',
        r'atrium', r'atrial', r'aorta', r'aortic', r'mitral', r'tricuspid', r'coronary',
        
        # Diseases & Conditions
        r'cardiomyopathy', r'myocarditis', r'pericarditis', r'endocarditis',
        r'arrhythmia', r'fibrillation', r'tachycardia', r'bradycardia', r'angina',
        r'heart failure', r'heart arrest', r'heart disease', r'heart attack', r'heart defect',
        r'infarction', r'thromboembolism', r'atherosclerosis', r'arteriosclerosis',
        r'aneurysm', r'qt prolong', r'torsade', r'extrasystole', r'valvulopathy', r'cardiogenic',
    ]
    
    # Use two fixed-width negative lookbehinds (?<!non)(?<!non-) to prevent matching "non-cardiac" or "noncardiac"
    pattern_parts = [rf'(?<!non)(?<!non-){kw}' for kw in cardio_keywords]
    pattern_parts.append(r'(?<!non)(?<!non-)\becg\b')
    pattern_parts.append(r'(?<!non)(?<!non-)\bekg\b')
    
    pattern_str = r'\b(?:' + '|'.join(pattern_parts) + r')\b'
    return re.compile(pattern_str, re.IGNORECASE)

def main():
    # 1. Download MeSH C14 IDs
    c14_ids = download_mesh_c14_ids()
    
    # 2. Compile keyword regex pattern
    cardio_pattern = get_cardio_regex()
    
    # 3. Load BioDEX-ICSR dataset
    print(f"Loading HF dataset '{DATASET_ID}'...")
    dataset = load_dataset(DATASET_ID)
    print("Dataset loaded successfully.")
    
    # Helper to check if a sample contains a C14 MeSH term
    def has_c14_mesh(mesh_terms_str):
        if not mesh_terms_str:
            return False
        terms = mesh_terms_str.split('; ')
        for t in terms:
            parts = t.split(':')
            if len(parts) > 0:
                mesh_id = parts[0].strip()
                if mesh_id in c14_ids:
                    return True
        return False

    # Helper to check if a sample matches cardiology keywords
    def matches_cardio_text(ex):
        title = ex.get('title') or ''
        keywords = ex.get('keywords') or ''
        target = ex.get('target') or ''
        
        return bool(
            cardio_pattern.search(title) or 
            cardio_pattern.search(keywords) or 
            cardio_pattern.search(target)
        )

    # 4. Process the dataset combined
    print("\nMerging all dataset splits (train, validation, test) into a single dataset...")
    all_splits = [dataset[s] for s in dataset.keys()]
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(all_splits)
    total_rows = len(combined_dataset)
    print(f"Total merged size: {total_rows} samples.")

    print("\nFiltering cardiology samples from the merged dataset...")
    matched_by_mesh = 0
    matched_by_text = 0
    cardio_indices = []

    for idx, ex in enumerate(combined_dataset):
        mesh_terms = ex.get('mesh_terms', '')
        
        # Check MeSH C14 first
        by_mesh = has_c14_mesh(mesh_terms)
        
        if by_mesh:
            matched_by_mesh += 1
            cardio_indices.append(idx)
        else:
            # Fallback to refined text search
            by_text = matches_cardio_text(ex)
            if by_text:
                matched_by_text += 1
                cardio_indices.append(idx)

    # Select matched rows
    filtered_dataset = combined_dataset.select(cardio_indices)
    percentage = (len(filtered_dataset) / total_rows) * 100 if total_rows > 0 else 0

    print(f"  Matched by MeSH C14: {matched_by_mesh}")
    print(f"  Matched by keywords (no MeSH match): {matched_by_text}")
    print(f"  Total Cardiology samples: {len(filtered_dataset)} ({percentage:.2f}%)")

    # 5. Save the subset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save as Hugging Face dataset format
    hf_save_path = os.path.join(OUTPUT_DIR, 'cardio_subset')
    print(f"\nSaving Hugging Face dataset to disk at '{hf_save_path}'...")
    filtered_dataset.save_to_disk(hf_save_path)
    
    # Convert to pandas DataFrame for exporting to tabular formats (JSONL and CSV)
    print("Converting to pandas DataFrame for exporting...")
    df = filtered_dataset.to_pandas()
    
    # Save as CSV (for tabular spreadsheet consumption)
    csv_path = os.path.join(OUTPUT_DIR, 'cardio_dataset.csv')
    print(f"Saving as CSV at '{csv_path}'...")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save as JSONL (which preserves nested list/dict structures perfectly)
    jsonl_path = os.path.join(OUTPUT_DIR, 'cardio_dataset.jsonl')
    print(f"Saving as JSON Lines at '{jsonl_path}'...")
    df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    
    # Print final summary table
    print("\n" + "="*50)
    print("                 SUMMARY STATISTICS")
    print("="*50)
    print(f"Original Combined Size : {total_rows:,}")
    print(f"Cardiology Subset Size : {len(filtered_dataset):,}")
    print(f"Cardiology Percentage  : {percentage:.2f}%")
    print(f"Matched via MeSH C14   : {matched_by_mesh:,}")
    print(f"Matched via Keywords   : {matched_by_text:,}")
    print("="*50)
    print("BioDEX subset preparation completed successfully!")

if __name__ == '__main__':
    main()

import os
import sys
import pandas as pd
import shutil

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = 'data'

def preprocess_biodex():
    raw_path = os.path.join(DATA_DIR, 'biodex', 'raw', 'cardio_dataset.jsonl')
    processed_dir = os.path.join(DATA_DIR, 'biodex', 'processed')
    
    print(f"\n[BioDEX Preprocessing] Loading raw dataset from '{raw_path}'...")
    df = pd.read_json(raw_path, lines=True)
    
    # Identify clinical vs. metadata columns to keep
    cols_to_keep = ['title', 'abstract', 'target', 'mesh_terms', 'keywords']
    if 'fulltext_processed' in df.columns:
        cols_to_keep.append('fulltext_processed')
    elif 'fulltext' in df.columns:
        cols_to_keep.append('fulltext')
        
    print(f"  Refining to clinical columns: {cols_to_keep}")
    df_clinical = df[cols_to_keep].copy()
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Save to data/biodex/processed/
    proc_csv = os.path.join(processed_dir, 'biodex_cardio_clinical.csv')
    proc_jsonl = os.path.join(processed_dir, 'biodex_cardio_clinical.jsonl')
    
    print(f"  Saving clinical dataset to processed/ CSV: '{proc_csv}'...")
    df_clinical.to_csv(proc_csv, index=False, encoding='utf-8-sig')
    print(f"  Saving clinical dataset to processed/ JSONL: '{proc_jsonl}'...")
    df_clinical.to_json(proc_jsonl, orient='records', lines=True, force_ascii=False)
    
    # 2. Copy to data/ (root of data)
    root_csv = os.path.join(DATA_DIR, 'biodex_cardio_clinical.csv')
    root_jsonl = os.path.join(DATA_DIR, 'biodex_cardio_clinical.jsonl')
    
    print(f"  Copying clean files directly to root data folder: '{root_csv}'...")
    shutil.copy2(proc_csv, root_csv)
    shutil.copy2(proc_jsonl, root_jsonl)
    
    print(f"  BioDEX preprocessing complete. Processed {len(df_clinical)} rows.")
    return df_clinical

def preprocess_openfda():
    raw_path = os.path.join(DATA_DIR, 'openfda', 'raw', 'fda_cardio_dataset.jsonl')
    processed_dir = os.path.join(DATA_DIR, 'openfda', 'processed')
    
    print(f"\n[openFDA Preprocessing] Loading raw dataset from '{raw_path}'...")
    df = pd.read_json(raw_path, lines=True)
    
    # Retain patient demographics, drug names, reaction terms, severity indicators, and subcategory
    cols_to_keep = [
        'patient_age', 'patient_age_unit', 'patient_sex',
        'seriousness_death', 'seriousness_hospitalization', 'seriousness_life_threatening', 
        'seriousness_disabling', 'seriousness_other', 
        'reactions', 'drugs', 'cardio_category'
    ]
    
    print(f"  Refining to clinical columns: {cols_to_keep}")
    df_clinical = df[cols_to_keep].copy()
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Save to data/openfda/processed/
    proc_csv = os.path.join(processed_dir, 'fda_cardio_clinical.csv')
    proc_jsonl = os.path.join(processed_dir, 'fda_cardio_clinical.jsonl')
    
    print(f"  Saving clinical dataset to processed/ CSV: '{proc_csv}'...")
    df_clinical.to_csv(proc_csv, index=False, encoding='utf-8-sig')
    print(f"  Saving clinical dataset to processed/ JSONL: '{proc_jsonl}'...")
    df_clinical.to_json(proc_jsonl, orient='records', lines=True, force_ascii=False)
    
    # 2. Copy to data/ (root of data)
    root_csv = os.path.join(DATA_DIR, 'fda_cardio_clinical.csv')
    root_jsonl = os.path.join(DATA_DIR, 'fda_cardio_clinical.jsonl')
    
    print(f"  Copying clean files directly to root data folder: '{root_csv}'...")
    shutil.copy2(proc_csv, root_csv)
    shutil.copy2(proc_jsonl, root_jsonl)
    
    print(f"  openFDA preprocessing complete. Processed {len(df_clinical)} rows.")
    return df_clinical

def main():
    preprocess_biodex()
    preprocess_openfda()
    print("\nDataset pre-processing and mapping completed successfully!")

if __name__ == '__main__':
    main()

import os
import sys
import shutil
import time

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.prepare_cardio_subset import main as run_biodex
from scripts.fetch_fda_cardio import main as run_openfda
from scripts.preprocess_datasets import main as run_preprocess

def main():
    print("="*60)
    print("            BIOMEDICAL ADVERSE EVENT PIPELINE")
    print("="*60)
    
    start_time = time.time()
    
    # Clear existing data folder to start fresh
    if os.path.exists('data'):
        print("Cleaning up existing 'data/' directory...")
        shutil.rmtree('data', ignore_errors=True)
        time.sleep(1)  # Brief pause to allow Windows filesystem to release file locks
        
    # Stage 1: BioDEX-ICSR
    print("\n" + "-"*50)
    print("STAGE 1: Subsetting BioDEX-ICSR Literature Dataset")
    print("-"*50)
    run_biodex()
    
    # Stage 2: openFDA FAERS
    print("\n" + "-"*50)
    print("STAGE 2: Fetching Balanced openFDA FAERS Reports")
    print("-"*50)
    run_openfda()
    
    # Stage 3: Clinical Preprocessing
    print("\n" + "-"*50)
    print("STAGE 3: Pre-processing and Clinical Feature Selection")
    print("-"*50)
    run_preprocess()
    
    end_time = time.time()
    print("\n" + "="*60)
    print(f"PIPELINE EXECUTED SUCCESSFULLY IN {end_time - start_time:.2f} SECONDS!")
    print("="*60)

if __name__ == '__main__':
    main()

import os
import sys
import json
import time
import argparse
import requests
import pandas as pd
import re

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Paths configuration
DATA_DIR = 'data'
RSI_CACHE_PATH = os.path.join(DATA_DIR, 'rsi_mapping.json')
BIODEX_INPUT_PATH = os.path.join(DATA_DIR, 'biodex_cardio_clinical.csv')
FDA_INPUT_PATH = os.path.join(DATA_DIR, 'fda_cardio_clinical.csv')

def clean_drug_name_for_api(drug_name):
    """Cleans raw drug names to improve match rate against openFDA label generic/brand fields."""
    if not isinstance(drug_name, str) or not drug_name:
        return ""
    name = drug_name.lower().strip()
    
    # Remove common salt/formulation suffixes
    suffixes_to_remove = [
        ' hydrochloride', ' sodium', ' calcium', ' sulfate', ' sulphate', ' phosphate', 
        ' mesylate', ' besylate', ' maleate', ' potassium', ' acetate', ' fumarate', 
        ' tartrate', ' bromide', ' iodide', ' chloride', ' gluconate', ' succinate',
        ' dl-lysine', ' dl lysine', ' medoxomil', ' tosylate', ' disodium'
    ]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            
    # Remove punctuation
    name = name.strip('\'".,()[]{}')
    return name

def fetch_drug_rsi(drug_name, api_key=None):
    """Queries openFDA Drug Label API with generic/brand name fallbacks in a single OR query."""
    cleaned_name = clean_drug_name_for_api(drug_name)
    if not cleaned_name:
        return "RSI not available"
        
    base_url = 'https://api.fda.gov/drug/label.json'
    
    # Combined search query: search generic, brand, or substance fields strictly in a single API call
    q = f'openfda.generic_name:"{cleaned_name}" OR openfda.brand_name:"{cleaned_name}" OR openfda.substance_name:"{cleaned_name}"'
    params = {'search': q, 'limit': 1}
    if api_key:
        params['api_key'] = api_key
        
    retries = 3
    while retries > 0:
        try:
            r = requests.get(base_url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                results = data.get('results', [])
                if results:
                    result = results[0]
                    
                    # Double-check that the cleaned name actually matches the returned drug metadata
                    openfda_info = result.get('openfda', {})
                    matched = False
                    for field in ['generic_name', 'brand_name', 'substance_name']:
                        vals = openfda_info.get(field, [])
                        if isinstance(vals, list):
                            for v in vals:
                                if cleaned_name in str(v).lower():
                                    matched = True
                                    break
                        elif isinstance(vals, str):
                            if cleaned_name in vals.lower():
                                matched = True
                        if matched:
                            break
                            
                    if not matched:
                        print(f"    Warning: API returned a record that does not match active names for '{drug_name}' (generic: {openfda_info.get('generic_name', 'None')}). Skipping.")
                        return "RSI not available"
                        
                    extracted = []
                    
                    if 'boxed_warning' in result:
                        text = result['boxed_warning']
                        extracted.append(f"BOXED WARNING:\n" + ("\n".join(text) if isinstance(text, list) else text))
                    if 'warnings_and_cautions' in result:
                        text = result['warnings_and_cautions']
                        extracted.append(f"WARNINGS AND CAUTIONS:\n" + ("\n".join(text) if isinstance(text, list) else text))
                    if 'adverse_reactions' in result:
                        text = result['adverse_reactions']
                        extracted.append(f"ADVERSE REACTIONS:\n" + ("\n".join(text) if isinstance(text, list) else text))
                        
                    if extracted:
                        return "\n\n".join(extracted)
                break # Break retries if empty results but 200 status
            elif r.status_code == 429:
                print("    [429] Rate limit hit. Waiting 10 seconds...")
                time.sleep(10)
                retries -= 1
            else:
                # 404 or other error means drug was not found
                break
        except Exception as e:
            print(f"    Connection error for drug '{drug_name}': {e}. Retrying...")
            time.sleep(2)
            retries -= 1
            
    return "RSI not available"



def extract_primary_suspected_drug(target_str):
    """Helper to parse first suspect drug from BioDEX target text."""
    if not isinstance(target_str, str) or not target_str:
        return "Unknown Drug"
    match = re.search(r'drugs:\s*([^:\n]+)', target_str)
    if match:
        drugs_list = [d.strip() for d in match.group(1).split(',')]
        if drugs_list:
            return drugs_list[0]
    return "Unknown Drug"

def main():
    parser = argparse.ArgumentParser(description="Standalone openFDA RSI safety profile cache builder.")
    parser.add_argument('--limit-drugs', type=int, default=None,
                        help="Limit the number of new drugs to fetch (useful for quick testing).")
    parser.add_argument('--clean', action='store_true',
                        help="Clean/delete the existing cache before running to prevent poisoned data.")
    args = parser.parse_args()

    print("="*60)
    # Load .env if exists for API key
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    openfda_key = os.environ.get('OPENFDA_API_KEY')
    if openfda_key:
        print("Using openFDA API key found in environment (240 requests/min).")
        delay = 0.25
    else:
        print("No openFDA API key found. Using anonymous access (limit 40 requests/min, using 1.6s delay).")
        delay = 1.6

    # Verify input datasets
    if not os.path.exists(BIODEX_INPUT_PATH) or not os.path.exists(FDA_INPUT_PATH):
        print(f"Error: Datasets must be built. Check paths {BIODEX_INPUT_PATH} and {FDA_INPUT_PATH}.")
        sys.exit(1)
        
    df_bio = pd.read_csv(BIODEX_INPUT_PATH)
    df_fda = pd.read_csv(FDA_INPUT_PATH)
    
    print(f"Scanning datasets. BioDEX size: {len(df_bio)} rows, openFDA size: {len(df_fda)} rows.")
    
    drugs_set = set()
    
    # BioDEX drugs
    for _, row in df_bio.iterrows():
        drug = extract_primary_suspected_drug(row.get('target', ''))
        if drug and drug != "Unknown Drug":
            drugs_set.add(drug)
            
    # openFDA drugs
    for _, row in df_fda.iterrows():
        drugs = row.get('drugs', '')
        if isinstance(drugs, str) and drugs:
            drugs_list = [d.strip() for d in drugs.split(';')]
            if drugs_list and drugs_list[0] and drugs_list[0] != "Unknown Drug":
                drugs_set.add(drugs_list[0])
                
    print(f"Found {len(drugs_set)} unique primary suspected drugs in the combined datasets.")
    
    # Handle clean option
    if args.clean and os.path.exists(RSI_CACHE_PATH):
        print("Cleaning up existing rsi_mapping.json cache to purge any poisoned data...")
        try:
            os.remove(RSI_CACHE_PATH)
            print("Cache deleted successfully.")
        except Exception as e:
            print(f"Warning: Failed to delete cache: {e}")

    # Load existing cache
    mapping = {}
    if os.path.exists(RSI_CACHE_PATH):
        try:
            with open(RSI_CACHE_PATH, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f"Loaded existing RSI cache with {len(mapping)} drugs.")
        except Exception as e:
            print(f"Warning: Failed to parse rsi_mapping.json ({e}). Creating new.")
            
    # Identify missing drugs
    missing_drugs = sorted([d for d in drugs_set if d and d not in mapping])
    print(f"Already cached: {len(drugs_set) - len(missing_drugs)} drugs. Missing: {len(missing_drugs)} drugs.")
    
    if not missing_drugs:
        print("All drugs are already cached! Nothing to fetch.")
        return
        
    if args.limit_drugs:
        print(f"Limiting fetch to the first {args.limit_drugs} missing drugs.")
        missing_drugs = missing_drugs[:args.limit_drugs]
        
    print(f"\nFetching safety profiles from openFDA for {len(missing_drugs)} drugs...")
    success_count = 0
    not_found_count = 0
    
    try:
        for idx, drug in enumerate(missing_drugs):
            print(f" [{idx+1}/{len(missing_drugs)}] Fetching '{drug}'...")
            start_t = time.time()
            rsi_text = fetch_drug_rsi(drug, openfda_key)
            
            if rsi_text == "RSI not available":
                print(f"   --> WARNING: Safety profile not found for '{drug}'")
                not_found_count += 1
            else:
                print(f"   --> SUCCESS: Safety profile cached ({len(rsi_text)} chars)")
                success_count += 1
                
            mapping[drug] = rsi_text
            
            # Save progressively
            with open(RSI_CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=2, ensure_ascii=False)
                
            # Rate limiting delay
            elapsed = time.time() - start_t
            sleep_needed = max(0.0, delay - elapsed)
            if sleep_needed > 0:
                time.sleep(sleep_needed)
                
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current cache and exiting...")
    finally:
        with open(RSI_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"\nExecution finished. Cache updated. Total cached drugs now: {len(mapping)}")
        print(f"In this run: {success_count} fetched successfully, {not_found_count} not found.")

if __name__ == '__main__':
    main()

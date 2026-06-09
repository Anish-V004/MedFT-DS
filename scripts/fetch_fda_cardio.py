import os
import sys
import argparse
import requests
import pandas as pd
import time

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# openFDA API Base URL
FDA_API_URL = 'https://api.fda.gov/drug/event.json'
OUTPUT_DIR = os.path.join('data', 'openfda', 'raw')
RECORDS_PER_CATEGORY = 500

# Define our 6 balanced cardiology categories and their respective search terms
CATEGORIES = {
    'Infarction / Ischemia': (
        'patient.reaction.reactionmeddrapt:("myocardial infarction" OR '
        '"myocardial ischaemia" OR "angina pectoris" OR "acute coronary syndrome" OR '
        '"coronary artery disease")'
    ),
    'Heart Failure': (
        'patient.reaction.reactionmeddrapt:("cardiac failure" OR "heart failure" OR '
        '"cardiogenic shock" OR "cardiac failure congestive" OR "congestive heart failure")'
    ),
    'Arrhythmias': (
        'patient.reaction.reactionmeddrapt:("atrial fibrillation" OR "sinus tachycardia" OR '
        '"sinus bradycardia" OR "cardiac arrhythmia" OR "extrasystoles" OR "arrhythmia" OR '
        '"atrial flutter")'
    ),
    'Inflammatory / Infectious': (
        'patient.reaction.reactionmeddrapt:("myocarditis" OR "pericarditis" OR '
        '"endocarditis" OR "pericardial effusion" OR "endocarditis bacterial")'
    ),
    'Vascular / Blood Pressure': (
        'patient.reaction.reactionmeddrapt:("hypertension" OR "hypotension" OR '
        '"hypertensive crisis" OR "blood pressure decreased" OR "blood pressure increased")'
    ),
    'Structural / Valvular / Arrest': (
        'patient.reaction.reactionmeddrapt:("cardiomyopathy" OR "cardiac arrest" OR '
        '"aortic valve stenosis" OR "mitral valve incompetence" OR "valvulopathy" OR '
        '"tricuspid valve disease")'
    )
}

def parse_args():
    parser = argparse.ArgumentParser(description="Fetch balanced cardiology adverse events from openFDA API.")
    parser.add_argument('--api-key', type=str, default=os.getenv('OPENFDA_API_KEY'),
                        help="openFDA API Key. Can also be set via the OPENFDA_API_KEY environment variable.")
    return parser.parse_args()

def fetch_category_records(category_name, search_query, api_key, seen_ids):
    """Fetches exactly RECORDS_PER_CATEGORY unique records for a given search query."""
    unique_records = []
    skip = 0
    limit = 100  # We fetch in moderate chunks to parse and filter duplicates progressively
    
    print(f"\nFetching records for category: '{category_name}'...")
    
    while len(unique_records) < RECORDS_PER_CATEGORY:
        params = {
            'search': search_query,
            'limit': limit,
            'skip': skip
        }
        if api_key:
            params['api_key'] = api_key
            
        try:
            response = requests.get(FDA_API_URL, params=params)
            
            # Handle rate limits (especially if running unauthenticated)
            if response.status_code == 429:
                print("Rate limit hit. Waiting 5 seconds...")
                time.sleep(5)
                continue
                
            if response.status_code != 200:
                print(f"Error fetching data (Status code: {response.status_code}): {response.text}")
                break
                
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                print("No more results available for this query.")
                break
                
            new_unique_found = 0
            for r in results:
                report_id = r.get('safetyreportid')
                if not report_id:
                    continue
                    
                # De-duplicate globally across all categories
                if report_id not in seen_ids:
                    seen_ids.add(report_id)
                    unique_records.append(r)
                    new_unique_found += 1
                    
                    if len(unique_records) == RECORDS_PER_CATEGORY:
                        break
            
            print(f"  Page skip={skip}: Retrieved {len(results)} records, found {new_unique_found} new unique reports. Current category total: {len(unique_records)}/{RECORDS_PER_CATEGORY}")
            
            # Update pagination offsets
            skip += limit
            
            # Add a small delay between requests to be polite to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(2)
            
    return unique_records

def flatten_report(report, category_name):
    """Flattens nested JSON adverse event report into a clean, flat dictionary structure."""
    patient = report.get('patient', {})
    
    # Extract reactions list
    reactions_list = []
    for reaction in patient.get('reaction', []):
        term = reaction.get('reactionmeddrapt')
        if term:
            reactions_list.append(term)
            
    # Extract drugs list
    drugs_list = []
    for drug in patient.get('drug', []):
        name = drug.get('medicinalproduct')
        if name:
            drugs_list.append(name.upper())
            
    # Seriousness flags
    return {
        'safetyreportid': report.get('safetyreportid'),
        'receivedate': report.get('receivedate'),
        'patient_age': patient.get('patientonsetage'),
        'patient_age_unit': patient.get('patientonsetageunit'),
        'patient_sex': patient.get('patientsex'),
        'seriousness_death': report.get('seriousnessdeath', '0'),
        'seriousness_hospitalization': report.get('seriousnesshospitalization', '0'),
        'seriousness_life_threatening': report.get('seriousnesslifethreatening', '0'),
        'seriousness_disabling': report.get('seriousnessdisabling', '0'),
        'seriousness_other': report.get('seriousnessother', '0'),
        'reactions': '; '.join(reactions_list),
        'drugs': '; '.join(drugs_list),
        'cardio_category': category_name
    }

def main():
    args = parse_args()
    api_key = args.api_key
    
    if api_key:
        print("Using provided openFDA API Key for authenticated requests.")
    else:
        print("No API Key detected. Performing unauthenticated requests (rate limits apply).")
        
    seen_ids = set()
    all_flattened_records = []
    category_counts = {}
    
    # Fetch from each subcategory
    for category_name, query in CATEGORIES.items():
        raw_records = fetch_category_records(category_name, query, api_key, seen_ids)
        category_counts[category_name] = len(raw_records)
        
        # Flatten and store
        for r in raw_records:
            flattened = flatten_report(r, category_name)
            all_flattened_records.append(flattened)
            
    # Combine into a DataFrame
    df = pd.DataFrame(all_flattened_records)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(OUTPUT_DIR, 'fda_cardio_dataset.csv')
    print(f"\nSaving dataset to CSV at: '{csv_path}'...")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save as JSON Lines
    jsonl_path = os.path.join(OUTPUT_DIR, 'fda_cardio_dataset.jsonl')
    print(f"Saving dataset to JSON Lines at: '{jsonl_path}'...")
    df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    
    # Print final summary statistics
    print("\n" + "="*50)
    print("           FDA CARDIOLOGY EXTRACTION SUMMARY")
    print("="*50)
    print(f"{'Clinical Category':<35} | {'Records Retrieved':<10}")
    print("-"*50)
    for cat, count in category_counts.items():
        print(f"{cat:<35} | {count:<10}")
    print("-"*50)
    print(f"{'Total Unique Cardiology Records':<35} | {len(df):<10}")
    print("="*50)
    print("openFDA extraction completed successfully!")

if __name__ == '__main__':
    main()

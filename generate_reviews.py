import os
import sys
import json
import time
import urllib.parse
import hashlib
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import re
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Reconfigure stdout to use UTF-8 to prevent console encoding errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Paths configuration
DATA_DIR = 'data'
RSI_CACHE_PATH = os.path.join(DATA_DIR, 'rsi_mapping.json')
BIODEX_INPUT_PATH = os.path.join(DATA_DIR, 'biodex_cardio_clinical.csv')
FDA_INPUT_PATH = os.path.join(DATA_DIR, 'fda_cardio_clinical.csv')
BIODEX_OUTPUT_PATH = os.path.join(DATA_DIR, 'biodex_chatml.jsonl')
FDA_OUTPUT_PATH = os.path.join(DATA_DIR, 'fda_chatml.jsonl')

# Load .env file
load_dotenv(dotenv_path=".env")

# Parse API keys for rotation
api_keys = []
keys_str = os.environ.get('GEMINI_API_KEYS') or os.environ.get('GEMINI_API_KEY')
if keys_str:
    api_keys = [k.strip() for k in re.split(r'[,;]', keys_str) if k.strip()]
    
# Also check for numbered keys: GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
for idx in range(1, 10):
    k = os.environ.get(f'GEMINI_API_KEY_{idx}')
    if k and k.strip() and k.strip() not in api_keys:
        api_keys.append(k.strip())

if not api_keys:
    print("Error: No GEMINI_API_KEY or GEMINI_API_KEYS found in environment.")
    sys.exit(1)

def mask_key(k):
    return k[:8] + "..." + k[-4:] if len(k) > 12 else "..."

print(f"Loaded {len(api_keys)} Gemini API Key(s) for rotation: {[mask_key(k) for k in api_keys]}")

# Create thread-safe pool of clients
client_queue = queue.Queue()
for k in api_keys:
    client_queue.put(genai.Client(api_key=k))
    
# Thread-safe lock for file writing
file_lock = threading.Lock()

# Define Structured Output Schema using Pydantic
class SeriousnessDetails(BaseModel):
    is_serious: bool = Field(description="True if the event meets any regulatory seriousness criteria, False otherwise.")
    criteria: str = Field(description="The seriousness criteria met: 'death', 'hospitalization', 'life-threatening', 'disabling', 'congenital anomaly', 'other serious medical event', or 'none'.")

class CausalityDetails(BaseModel):
    naranjo_score: int = Field(description="Calculated score from Naranjo algorithm (typically between -4 and +13).")
    interpretation: str = Field(description="Causality interpretation based on Naranjo score: 'Definite' (score >= 9), 'Probable' (5-8), 'Possible' (1-4), or 'Doubtful' (<= 0).")

class PVReviewResponse(BaseModel):
    chain_of_thought: str = Field(description="Step-by-step clinical reasoning for seriousness, MedDRA term matching, expectedness, and Naranjo causality scoring.")
    seriousness: SeriousnessDetails
    meddra_pt: str = Field(description="The exact MedDRA Preferred Term (PT) for the primary adverse event (e.g., 'Myocardial infarction').")
    expectedness: str = Field(description="Expected (Labelled) or Unexpected (Unlabelled) based on whether the reaction is listed in the provided drug's RSI text.")
    causality: CausalityDetails

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


def extract_primary_suspected_drug(target_str, narrative_text=None):
    """Helper to parse suspect drug from BioDEX target text, prioritizing the one mentioned in the narrative."""
    if not isinstance(target_str, str) or not target_str:
        return "Unknown Drug"
    match = re.search(r'drugs:\s*([^:\n]+)', target_str)
    if match:
        drugs_list = [d.strip() for d in match.group(1).split(',')]
        if narrative_text and isinstance(narrative_text, str):
            for drug in drugs_list:
                clean = clean_drug_name_for_api(drug)
                if clean and clean in narrative_text.lower():
                    return drug
        if drugs_list:
            return drugs_list[0]
    return "Unknown Drug"

def map_biodex_row(row):
    """Extracts narrative and suspected drug from BioDEX row."""
    narrative = row.get('abstract') or row.get('fulltext_processed') or row.get('title') or ""
    narrative_str = str(narrative).strip()
    drug = extract_primary_suspected_drug(row.get('target', ''), narrative_str)
    return narrative_str, str(drug).strip()

def map_fda_row(row):
    """Constructs a patient narrative and suspects drug from openFDA variables."""
    age = row.get('patient_age')
    sex_val = row.get('patient_sex')
    
    sex = "unknown sex"
    if sex_val == 1.0 or sex_val == 1:
        sex = "male"
    elif sex_val == 2.0 or sex_val == 2:
        sex = "female"
        
    reactions = row.get('reactions', '')
    drugs = row.get('drugs', '')
    
    drug = "Unknown Drug"
    if isinstance(drugs, str) and drugs:
        drugs_list = [d.strip() for d in drugs.split(';')]
        if drugs_list:
            drug = drugs_list[0]
            
    # Professional narrative phrasing
    has_age = pd.notna(age)
    has_sex = (sex != "unknown sex")
    
    if has_age and has_sex:
        patient_str = f"A {int(age)} year-old {sex} patient"
    elif has_age:
        patient_str = f"A {int(age)} year-old patient of unknown sex"
    elif has_sex:
        patient_str = f"A patient of unknown age {sex}"
    else:
        patient_str = "A patient of unknown age and sex"
        
    narrative = (
        f"{patient_str} experienced the following adverse events: {reactions}. The suspected drug is {drugs}."
    )
    return narrative, drug

def is_valid_narrative(narrative, drug):
    """Checks if a patient narrative is valid (length, junk terms, drug presence)."""
    if not narrative or not isinstance(narrative, str):
        return False
        
    narrative_lower = narrative.lower()
    
    # 1. Length check: clinical narratives should be at least 15 words
    if len(narrative_lower.split()) < 15:
        return False
        
    # 2. Administrative/junk phrase matching
    junk_phrases = [
        "no new information",
        "medical records requested",
        "medical records not provided",
        "consumer called",
        "product quality complaint",
        "refund",
        "no additional information",
        "further information has been requested"
    ]
    for junk in junk_phrases:
        if junk in narrative_lower:
            return False
            
    # Check for empty or generic placeholder narratives
    if narrative_lower.strip() in ["blank", "unknown", "nan", "none", "n/a", "null"]:
        return False
        
    # 3. Suspected drug mention check
    clean_drug = clean_drug_name_for_api(drug)
    if not clean_drug or clean_drug not in narrative_lower:
        return False
        
    return True

def load_processed_keys(output_path):
    """Loads keys of processed examples from output JSONL file to support resuming."""
    processed = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        messages = data.get('messages', [])
                        if len(messages) >= 2:
                            user_content = messages[1].get('content', '')
                            # Hash narrative as key
                            narrative_part = user_content.split("\n\nReference Safety Information (RSI):")[0]
                            clean_narrative = narrative_part.replace("Patient Narrative:\n", "").strip()
                            key = hashlib.md5(clean_narrative.encode('utf-8')).hexdigest()
                            processed.add(key)
        except Exception as e:
            print(f"Warning: Failed to parse existing output file {output_path} ({e}).")
    return processed

def run_dataset_pipeline(df, dataset_type, rsi_mapping, limit=None, model_name='gemini-3.1-flash-lite'):
    """Processes rows through Gemini, formats as ChatML, and appends to output files."""
    output_path = BIODEX_OUTPUT_PATH if dataset_type == 'biodex' else FDA_OUTPUT_PATH
    
    system_prompt = (
        "You are a Pharmacovigilance (PV) Medical Review Assistant. "
        "CRITICAL GROUNDING RULE: You must base your entire evaluation STRICTLY and EXCLUSIVELY on the provided Patient Narrative. "
        "Do NOT invent, hallucinate, or bring in external patient cases. Do NOT reference drugs or adverse events that are not explicitly written in the user's prompt. "
        "If the provided RSI does not match the drug in the narrative, explicitly state 'Drug Mismatch - Cannot Evaluate' in your reasoning."
    )
    
    processed_keys = load_processed_keys(output_path)
    
    print(f"\nProcessing {dataset_type.upper()} dataset. Outputs will be saved to '{output_path}'")
    print(f"Found {len(processed_keys)} already processed records to skip.")
    
    # Pre-calculate active rows to process (not in processed_keys and having narrative)
    active_rows = []
    for idx, row in df.iterrows():
        if dataset_type == 'biodex':
            narrative, drug = map_biodex_row(row)
        else:
            narrative, drug = map_fda_row(row)
            
        if not narrative:
            continue
            
        key = hashlib.md5(narrative.encode('utf-8')).hexdigest()
        if key in processed_keys:
            continue
            
        active_rows.append((idx, row, narrative, drug, key))
        
    total_active = len(active_rows)
    # Apply limit if any
    if limit is not None:
        active_rows = active_rows[:limit]
        total_active = len(active_rows)
        
    print(f"Total remaining rows to process in this run: {total_active}")
    
    # We no longer need model=None initialization here
    
    prompt_template = """Conduct a medical safety review of the following adverse event case:

Patient Narrative:
{patient_narrative}

Reference Safety Information (RSI) for {suspected_drug}:
{rsi_text}

[INSTRUCTIONS]
Perform three tasks:
1. Seriousness Assessment: Determine if the adverse event is serious based on standard regulatory criteria (Death, Hospitalization, Life-threatening, Disabling, Congenital Anomaly, or Other medically important event). Identify the exact MedDRA Preferred Term (PT) for the primary adverse event as a text string (e.g. 'Myocardial infarction').
2. Expectedness Assessment: Compare the Patient Narrative adverse event against the provided drug's RSI text to determine if it is 'Expected' (Labelled) or 'Unexpected' (Unlabelled). If the RSI text is not available (i.e. 'RSI not available'), you must use your own pre-trained clinical medical knowledge of this drug's official label and safety profile to determine whether the event is Expected or Unexpected.
3. Causality Assessment: Evaluate the relationship between the drug and the adverse event by applying the Naranjo scale logic (evaluating temporal relationship, dechallenge improvement, alternative causes, etc.). Deduce the score and assign the interpretation: Definite (score >= 9), Probable (5-8), Possible (1-4), or Doubtful (<= 0).
"""

    count_processed = 0
    samples_shown = []
    
    def process_row(args):
        idx, row, narrative, drug, key = args
        rsi_text = rsi_mapping.get(drug, "RSI not available")
        
        prompt_text = prompt_template.format(
            suspected_drug=drug,
            patient_narrative=narrative,
            rsi_text=rsi_text
        )
        
        success = False
        retry_count = 0
        chatml_record = None
        
        while not success:
            # Get an available client (this blocks if all clients are currently executing a request)
            client = client_queue.get()
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PVReviewResponse,
                        system_instruction=system_prompt,
                    )
                )
                
                if response.text:
                    response_json = json.loads(response.text)
                    success = True
                    
                    chain_of_thought = response_json.get('chain_of_thought', '')
                    decision_data = {k: v for k, v in response_json.items() if k != 'chain_of_thought'}
                    json_block = json.dumps(decision_data, indent=2)
                    
                    assistant_content = f"{chain_of_thought}\n\n```json\n{json_block}\n```"
                    user_content = f"Patient Narrative:\n{narrative}\n\nReference Safety Information (RSI):\n{rsi_text}"
                    
                    chatml_record = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                else:
                    raise Exception("Empty response text returned.")
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "quota" in error_str.lower() or "resourceexhausted" in error_str.lower() or "exhausted" in error_str.lower()
                
                if is_rate_limit:
                    # Specific key hit rate limit, backoff gently (other keys continue unhindered)
                    time.sleep(10)
                else:
                    retry_count += 1
                    if retry_count >= 3:
                        print(f"  Fatal Error processing drug '{drug}': {e}.")
                        break
                    time.sleep(retry_count * 5)
            finally:
                # Always return the client to the queue
                client_queue.put(client)
                
        return key, chatml_record, drug

    # Concurrency control: max workers equal to number of keys * 2 (or just number of keys)
    # Using len(api_keys) * 2 allows pipeline to keep queueing requests, but queue.get() regulates API load
    max_workers = max(1, len(api_keys) * 2)
    print(f"Executing {total_active} rows concurrently with {max_workers} threads across {len(api_keys)} API keys...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_row, args): args for args in active_rows}
        
        for future in as_completed(future_to_row):
            key, chatml_record, drug = future.result()
            
            if chatml_record:
                # Thread-safe file writing
                with file_lock:
                    with open(output_path, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(chatml_record, ensure_ascii=False) + '\n')
                        
                    count_processed += 1
                    processed_keys.add(key)
                    
                    if limit is not None and len(samples_shown) < limit:
                        samples_shown.append(chatml_record)
                        
                    left = total_active - count_processed
                    print(f" [{count_processed}/{total_active} done, {left} left] Completed review for '{drug}'")
                    
    return count_processed, samples_shown



def main():
    parser = argparse.ArgumentParser(description="PV Fine-Tuning Dataset Creator using openFDA and Gemini API.")
    parser.add_argument('--limit', type=int, default=10,
                        help="Total samples to process during validation. Default is 10.")
    parser.add_argument('--full-run', action='store_true',
                        help="If set, runs the full datasets (ignores --limit).")
    parser.add_argument('--model', type=str, default='gemini-3.1-flash-lite',
                        help="Gemini model to use. Default is gemini-3.1-flash-lite.")
    parser.add_argument('--biodex', action='store_true',
                        help="Run only the BioDEX dataset.")
    parser.add_argument('--openfda', action='store_true',
                        help="Run only the openFDA dataset.")
    args = parser.parse_args()
    
    # Determine which datasets to run (if neither is specified, run both)
    run_biodex = args.biodex or not (args.biodex or args.openfda)
    run_openfda = args.openfda or not (args.biodex or args.openfda)
    
    # Verify input datasets
    if not os.path.exists(BIODEX_INPUT_PATH) or not os.path.exists(FDA_INPUT_PATH):
        print(f"Error: Datasets must be built. Check paths {BIODEX_INPUT_PATH} and {FDA_INPUT_PATH}.")
        sys.exit(1)
        
    df_bio = pd.read_csv(BIODEX_INPUT_PATH)
    df_fda = pd.read_csv(FDA_INPUT_PATH)
    
    # Calculate limits
    if args.full_run:
        bio_limit = None
        fda_limit = None
        bio_size = len(df_bio) if run_biodex else 0
        fda_size = len(df_fda) if run_openfda else 0
        print(f"Beginning FULL pipeline execution. Target BioDEX size: {bio_size} rows, Target openFDA size: {fda_size} rows.")
    else:
        if run_biodex and run_openfda:
            bio_limit = args.limit // 2
            fda_limit = args.limit - bio_limit
        elif run_biodex:
            bio_limit = args.limit
            fda_limit = 0
        elif run_openfda:
            bio_limit = 0
            fda_limit = args.limit
            
        print(f"Beginning TEST run: targeting {args.limit} samples ({bio_limit} from BioDEX, {fda_limit} from openFDA).")
        
    # Phase 1: Load RSI Mapping Cache
    print("\n" + "="*50)
    print("PHASE 1: Loading Drug Reference Safety Information (RSI) Cache")
    print("="*50)
    
    if not os.path.exists(RSI_CACHE_PATH):
        print(f"Error: RSI cache mapping file not found at {RSI_CACHE_PATH}.")
        print("Please run the standalone extraction script first:")
        print("  uv run python scripts/extract_rsi.py")
        sys.exit(1)
        
    # Attempt to load cache with retry logic for concurrent access
    rsi_mapping = {}
    load_success = False
    for attempt in range(5):
        try:
            with open(RSI_CACHE_PATH, 'r', encoding='utf-8') as f:
                rsi_mapping = json.load(f)
            load_success = True
            break
        except Exception as e:
            if attempt < 4:
                print(f"Warning: Failed to read {RSI_CACHE_PATH} (may be locked/writing). Retrying in 0.5s... ({e})")
                time.sleep(0.5)
            else:
                print(f"Error: Failed to read {RSI_CACHE_PATH} after 5 attempts: {e}")
                sys.exit(1)
    
    print(f"Successfully loaded RSI cache containing {len(rsi_mapping)} drugs.")
    
    # Phase 2 & 3: Generate reviews
    print("\n" + "="*50)
    print("PHASE 2 & 3: Querying Gemini & Structuring Reviews")
    print("="*50)
    
    # Process BioDEX
    bio_count, bio_samples = 0, []
    if run_biodex:
        bio_count, bio_samples = run_dataset_pipeline(df_bio, 'biodex', rsi_mapping, limit=bio_limit, model_name=args.model)
    
    # Process openFDA
    fda_count, fda_samples = 0, []
    if run_openfda:
        fda_count, fda_samples = run_dataset_pipeline(df_fda, 'openfda', rsi_mapping, limit=fda_limit, model_name=args.model)
    
    print(f"\nExtraction complete. Successfully processed: {bio_count} BioDEX samples, {fda_count} openFDA samples.")
    
    # If test run, print sample records to console for user validation
    if not args.full_run:
        print("\n" + "="*70)
        print("                 TEST VALIDATION CHATML SAMPLES")
        print("="*70)
        
        all_test_samples = [('BioDEX', s) for s in bio_samples] + [('openFDA', s) for s in fda_samples]
        for idx, (source, sample) in enumerate(all_test_samples):
            print(f"\n--- [Sample {idx+1}] Source: {source} ---")
            print("System Prompt:")
            print(f"  {sample['messages'][0]['content']}")
            print("\nUser Prompt (Patient Narrative + RSI):")
            print(f"  {sample['messages'][1]['content'][:400]}...")
            print("\nAssistant Response (CoT + Structured Decisions JSON):")
            print(f"  {sample['messages'][2]['content']}")
            print("-" * 70)
            
        print("\nTest run validation completed! Please verify the formatting above. "
              "If correct, run with '--full-run' to execute on all records.")

if __name__ == '__main__':
    main()

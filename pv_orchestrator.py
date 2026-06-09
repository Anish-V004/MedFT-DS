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
import google.generativeai as genai
import re

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
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))

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
            
    narrative = (
        f"A {f'{int(age)} year-old' if pd.notna(age) else 'patient of unknown age'} {sex} patient "
        f"experienced the following adverse events: {reactions}. The suspected drug is {drugs}."
    )
    return narrative, drug

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

def run_dataset_pipeline(df, dataset_type, rsi_mapping, limit=None, model_name='gemini-2.5-flash'):
    """Processes rows through Gemini, formats as ChatML, and appends to output files."""
    output_path = BIODEX_OUTPUT_PATH if dataset_type == 'biodex' else FDA_OUTPUT_PATH
    processed_keys = load_processed_keys(output_path)
    
    print(f"\nProcessing {dataset_type.upper()} dataset. Outputs will be saved to '{output_path}'")
    print(f"Found {len(processed_keys)} already processed records to skip.")
    
    system_prompt = (
        "You are a Pharmacovigilance (PV) Medical Review Assistant. "
        "CRITICAL GROUNDING RULE: You must base your entire evaluation STRICTLY and EXCLUSIVELY on the provided Patient Narrative. "
        "Do NOT invent, hallucinate, or bring in external patient cases. Do NOT reference drugs or adverse events that are not explicitly written in the user's prompt. "
        "If the provided RSI does not match the drug in the narrative, explicitly state 'Drug Mismatch - Cannot Evaluate' in your reasoning."
    )
    
    # Initialize Gemini model
    model = genai.GenerativeModel(model_name, system_instruction=system_prompt)
    
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
    
    for idx, row in df.iterrows():
        # Get narrative and drug
        if dataset_type == 'biodex':
            narrative, drug = map_biodex_row(row)
        else:
            narrative, drug = map_fda_row(row)
            
        if not narrative:
            continue
            
        # Programmatic Grounding Check: skip row if suspected drug is not mentioned in the narrative text
        clean_drug = clean_drug_name_for_api(drug)
        if clean_drug not in narrative.lower():
            continue
            
        # Deduplication key
        key = hashlib.md5(narrative.encode('utf-8')).hexdigest()
        if key in processed_keys:
            continue
            
        rsi_text = rsi_mapping.get(drug, "RSI not available")
        
        # Stop if we hit the limit
        if limit is not None and count_processed >= limit:
            break
            
        # Prompt construction
        prompt_text = prompt_template.format(
            suspected_drug=drug,
            patient_narrative=narrative,
            rsi_text=rsi_text
        )
        
        print(f" [{count_processed+1}] Querying Gemini for suspect drug '{drug}'...")
        
        # Call Gemini API with Pydantic JSON enforcement and robust retry block
        success = False
        retry_count = 0
        while not success and retry_count < 3:
            try:
                response = model.generate_content(
                    prompt_text,
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": PVReviewResponse
                    }
                )
                
                # Check response
                if response.text:
                    response_json = json.loads(response.text)
                    success = True
                else:
                    raise Exception("Empty response text returned.")
            except Exception as e:
                retry_count += 1
                error_str = str(e)
                if retry_count >= 3:
                    print(f"  Fatal Error: Failed to process sample after {retry_count} attempts. Error: {e}. Terminating program.")
                    sys.exit(1)
                
                # Check if it's a rate limit error (429 or quota exceeded)
                if "429" in error_str or "quota" in error_str.lower() or "resourceexhausted" in error_str.lower():
                    wait_time = 65  # Sleep for a full minute + padding to clear the RPM window
                    print(f"  Rate limit (429) hit: {error_str.strip()}. Waiting {wait_time}s for window to reset...")
                else:
                    wait_time = retry_count * 5
                    print(f"  Error calling Gemini API: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            
        # Format the review responses
        chain_of_thought = response_json.get('chain_of_thought', '')
        decision_data = {k: v for k, v in response_json.items() if k != 'chain_of_thought'}
        json_block = json.dumps(decision_data, indent=2)
        
        assistant_content = f"{chain_of_thought}\n\n```json\n{json_block}\n```"
        user_content = f"Patient Narrative:\n{narrative}\n\nReference Safety Information (RSI):\n{rsi_text}"
        
        # Format in ChatML
        chatml_record = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        # Save progressively row-by-row
        with open(output_path, 'a', encoding='utf-8') as f_out:
            f_out.write(json.dumps(chatml_record, ensure_ascii=False) + '\n')
            
        count_processed += 1
        processed_keys.add(key)
        
        # Collect test samples to display to the user
        if limit is not None:
            samples_shown.append(chatml_record)
            
        # Rate Limiting: Capped at 5 RPM on Gemini 3.5 Flash Free Tier.
        # Sleep 12.5 seconds to ensure ~4.8 RPM.
        time.sleep(12.5)
        
    return count_processed, samples_shown



def main():
    parser = argparse.ArgumentParser(description="PV Fine-Tuning Dataset Creator using openFDA and Gemini API.")
    parser.add_argument('--limit', type=int, default=10,
                        help="Total samples to process during validation. Default is 10.")
    parser.add_argument('--full-run', action='store_true',
                        help="If set, runs the full datasets (ignores --limit).")
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                        help="Gemini model to use. Default is gemini-2.5-flash.")
    args = parser.parse_args()
    
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
        print(f"Beginning FULL pipeline execution. BioDEX size: {len(df_bio)} rows, openFDA size: {len(df_fda)} rows.")
    else:
        # Balanced 10-sample limit (5 from each)
        bio_limit = args.limit // 2
        fda_limit = args.limit - bio_limit
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
    bio_count, bio_samples = run_dataset_pipeline(df_bio, 'biodex', rsi_mapping, limit=bio_limit, model_name=args.model)
    
    # Process openFDA
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

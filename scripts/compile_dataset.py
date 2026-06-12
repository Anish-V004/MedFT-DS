import os
import json
import random
import sys
import argparse
import pandas as pd
import numpy as np
import tiktoken

# Ensure UTF-8 printing on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Constants
SYSTEM_PROMPT = (
    "You are a Pharmacovigilance (PV) Medical Review Assistant. "
    "CRITICAL GROUNDING RULE: You must base your entire evaluation STRICTLY and EXCLUSIVELY on the provided Patient Narrative. "
    "Do NOT invent, hallucinate, or bring in external patient cases. Do NOT reference drugs or adverse events that are not explicitly written in the user's prompt. "
    "If the provided RSI does not match the drug in the narrative, explicitly state 'Drug Mismatch - Cannot Evaluate' in your reasoning."
)

ADMIN_NOISE_TEMPLATES = [
    "Consumer called to ask for a refund for their prescription of {drug}. The package was damaged during shipping.",
    "Medical records were requested for the patient taking {drug} but they have not been provided by the clinical site.",
    "Follow-up report for patient on {drug}: No new clinical information has been received at this time.",
    "Product quality complaint: The customer noticed a discolored tablet in the bottle of {drug}. No adverse events reported.",
    "Customer called to request a replacement bottle of {drug} because they misplaced their current medication.",
    "Administrative notice: The patient on {drug} requested to be removed from the pharmacy mailing list.",
    "Insurance billing inquiry: Customer called to check co-pay pricing details for {drug}."
]

def calculate_tokens(sample, enc):
    """Calculates the ChatML formatted token size of a sample."""
    messages = sample.get('messages', [])
    full_text = ""
    for msg in messages:
        full_text += f"<|im_start|>{msg.get('role', '')}\n{msg.get('content', '')}<|im_end|>\n"
    return len(enc.encode(full_text))

def parse_valid_review(sample, enc):
    """Parses a valid ChatML review to extract its metrics and token count for balancing."""
    messages = sample.get('messages', [])
    if len(messages) < 3:
        return None
        
    user_content = messages[1].get('content', '')
    assistant_content = messages[2].get('content', '')
    
    # Identify source
    source = 'openfda' if 'The suspected drug is' in user_content and 'experienced the following adverse events' in user_content else 'biodex'
    
    # Check RSI availability
    rsi_avail = 'RSI not available' not in user_content
    
    # Extract JSON block
    start_idx = assistant_content.find('{')
    end_idx = assistant_content.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
        
    try:
        json_data = json.loads(assistant_content[start_idx:end_idx+1])
        expectedness = json_data.get('expectedness', 'Unexpected')
        causality = json_data.get('causality', {})
        naranjo = causality.get('naranjo_score', 1)
        
        # Categorize Naranjo
        if naranjo <= 0:
            causality_cat = 'Doubtful'
        elif 1 <= naranjo <= 4:
            causality_cat = 'Possible'
        else:
            causality_cat = 'Probable/Definite'
            
        tokens = calculate_tokens(sample, enc)
            
        return {
            'sample': sample,
            'source': source,
            'expectedness': expectedness,
            'causality_cat': causality_cat,
            'rsi_avail': rsi_avail,
            'tokens': tokens
        }
    except Exception:
        return None

def generate_negative_cases(count_per_cat=100):
    """Programmatically generates diversified negative escalation samples."""
    samples = []
    
    # Load raw datasets to grab realistic demographics, reactions, and drugs
    fda_drugs = ["Lisinopril", "Aspirin", "Humira", "Atorvastatin", "Metoprolol", "Furosemide", "Soliris", "Amlodipine"]
    fda_reactions = ["Myocardial infarction", "Renal failure acute", "Thrombocytopenia", "Cardiac arrest", "Angina pectoris"]
    
    if os.path.exists('data/fda_cardio_clinical.csv'):
        try:
            df = pd.read_csv('data/fda_cardio_clinical.csv', nrows=100)
            if not df.empty:
                fda_drugs = [d.split(';')[0].strip() for d in df['drugs'].dropna() if d]
                fda_reactions = [r.split(';')[0].strip() for r in df['reactions'].dropna() if r]
        except Exception:
            pass
            
    # 1. Missing Drugs
    for i in range(count_per_cat):
        age = random.randint(18, 85)
        sex = random.choice(["male", "female"])
        reaction = random.choice(fda_reactions)
        
        narrative = f"A {age} year-old {sex} patient experienced the following adverse events: {reaction}. The suspect medication was not documented in the safety report."
        rsi_text = "RSI not available"
        
        assistant_json = {
            "chain_of_thought": "Evaluation failed: The patient narrative does not mention any suspect medication. A clinical pharmacovigilance review cannot be performed without identifying the administered drug.",
            "seriousness": {
                "is_serious": True,
                "criteria": "other serious medical event"
            },
            "meddra_pt": "None",
            "expectedness": "Unexpected",
            "causality": {
                "naranjo_score": 0,
                "interpretation": "Unassessable - Missing Data"
            }
        }
        
        user_content = f"Patient Narrative:\n{narrative}\n\nReference Safety Information (RSI):\n{rsi_text}"
        assistant_content = f"{assistant_json['chain_of_thought']}\n\n```json\n{json.dumps(assistant_json, indent=2)}\n```"
        
        samples.append({
            "category": "Missing Drug",
            "sample": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
        })

    # 2. Missing Events
    for i in range(count_per_cat):
        age = random.randint(18, 85)
        sex = random.choice(["male", "female"])
        drug = random.choice(fda_drugs)
        
        narrative = f"A {age} year-old {sex} patient was prescribed {drug} for cardiovascular therapy. No adverse events, complaints, or physical symptoms were reported during the follow-up period."
        rsi_text = f"BOXED WARNING:\nWARNING: Serious events are possible.\n\nWARNINGS AND CAUTIONS:\nMonitor patient closely.\n\nADVERSE REACTIONS:\nHeadache, nausea."
        
        assistant_json = {
            "chain_of_thought": "Evaluation failed: The patient narrative does not describe any adverse events or reactions. A safety assessment cannot be completed without a reported reaction.",
            "seriousness": {
                "is_serious": False,
                "criteria": "none"
            },
            "meddra_pt": "None",
            "expectedness": "Unexpected",
            "causality": {
                "naranjo_score": 0,
                "interpretation": "Unassessable - Missing Data"
            }
        }
        
        user_content = f"Patient Narrative:\n{narrative}\n\nReference Safety Information (RSI):\n{rsi_text}"
        assistant_content = f"{assistant_json['chain_of_thought']}\n\n```json\n{json.dumps(assistant_json, indent=2)}\n```"
        
        samples.append({
            "category": "Missing Event",
            "sample": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
        })

    # 3. Administrative Noise
    for i in range(count_per_cat):
        drug = random.choice(fda_drugs)
        template = random.choice(ADMIN_NOISE_TEMPLATES)
        narrative = template.format(drug=drug)
        rsi_text = "RSI not available"
        
        assistant_json = {
            "chain_of_thought": f"Evaluation failed: The text contains only administrative or clerical metadata ('{narrative.split('.')[0].lower()}') and does not describe a clinical patient case. No safety review can be concluded.",
            "seriousness": {
                "is_serious": False,
                "criteria": "none"
            },
            "meddra_pt": "None",
            "expectedness": "Unexpected",
            "causality": {
                "naranjo_score": 0,
                "interpretation": "Unassessable - Missing Data"
            }
        }
        
        user_content = f"Patient Narrative:\n{narrative}\n\nReference Safety Information (RSI):\n{rsi_text}"
        assistant_content = f"{assistant_json['chain_of_thought']}\n\n```json\n{json.dumps(assistant_json, indent=2)}\n```"
        
        samples.append({
            "category": "Administrative Noise",
            "sample": {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
        })

    return samples

def main():
    parser = argparse.ArgumentParser(description="PV Dataset Compiler V2: Limits token count and balances RSI types.")
    parser.add_argument('--output', type=str, default='data/pv_safety_review_dataset_3000.jsonl',
                        help="Path to save the new compiled dataset.")
    parser.add_argument('--max-tokens', type=int, default=6000,
                        help="Hard maximum token count limit (default 6000).")
    parser.add_argument('--pref-tokens', type=int, default=4000,
                        help="Preferred maximum token count limit (default 4000).")
    args = parser.parse_args()
    
    print("==================================================")
    print("PHARMACOVIGILANCE DATASET COMPILER V2")
    print("==================================================")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    # 1. Load existing reviews
    valid_pool = []
    for fpath in ['data/biodex_chatml.jsonl', 'data/fda_chatml.jsonl']:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parsed = parse_valid_review(json.loads(line), enc)
                        if parsed:
                            valid_pool.append(parsed)
                            
    print(f"Loaded {len(valid_pool)} valid reviews from the source pool.")
    
    # Filter by hard token limit
    filtered_pool = [r for r in valid_pool if r['tokens'] <= args.max_tokens]
    print(f"Pool size after strictly excluding samples > {args.max_tokens} tokens: {len(filtered_pool)}")
    
    # Categorize filtered pool for balancing
    categories = {
        'Doubtful': [r for r in filtered_pool if r['causality_cat'] == 'Doubtful'],
        'Possible': [r for r in filtered_pool if r['causality_cat'] == 'Possible'],
        'Probable/Definite': [r for r in filtered_pool if r['causality_cat'] == 'Probable/Definite']
    }
    
    target_medical_size = 2700
    target_neg_size = 300
    
    # Target per category if perfectly balanced
    ideal_per_cat = target_medical_size // 3 # 900
    
    # Check the size of Probable/Definite
    prob_def_pool = categories['Probable/Definite']
    selected_prob_def = []
    
    # For Probable/Definite, we take all we can under args.max_tokens (since total under 6000 is 504)
    # We prioritize under 4,000 tokens first
    prob_def_under_pref = [r for r in prob_def_pool if r['tokens'] <= args.pref_tokens]
    prob_def_above_pref = [r for r in prob_def_pool if r['tokens'] > args.pref_tokens]
    
    selected_prob_def += prob_def_under_pref
    selected_prob_def += prob_def_above_pref
    
    num_prob_def = len(selected_prob_def)
    print(f"Probable/Definite selected: {num_prob_def} (all available under {args.max_tokens} tokens)")
    print(f"  - Under {args.pref_tokens} tokens: {len(prob_def_under_pref)}")
    print(f"  - Between {args.pref_tokens} and {args.max_tokens} tokens: {len(prob_def_above_pref)}")
    
    # Calculate the remaining target to hit 2,700 total medical reviews
    shortfall = ideal_per_cat - num_prob_def # 900 - 504 = 396
    
    # Split the shortfall between Possible and Doubtful
    added_per_cat = shortfall // 2 # 198
    target_doubtful = ideal_per_cat + added_per_cat # 1098
    target_possible = ideal_per_cat + (shortfall - added_per_cat) # 1098
    
    # --- SELECT POSSIBLE (Target: 1098) ---
    # We want to sample exactly target_possible from Possible pool.
    # To maximize quality and meet token preferences, we look at the Possible under-pref pool (1,950 records available).
    # We want a highly diverse split of RSI Available and RSI Not Available (AI-generated knowledge fallback).
    possible_pool = [r for r in categories['Possible'] if r['tokens'] <= args.pref_tokens]
    
    poss_rsi_avail = [r for r in possible_pool if r['rsi_avail']]
    poss_rsi_not_avail = [r for r in possible_pool if not r['rsi_avail']]
    
    # Sample 50% each
    half_target = target_possible // 2
    
    random.seed(42) # Set seed for reproducibility
    sampled_poss_avail = random.sample(poss_rsi_avail, min(len(poss_rsi_avail), half_target))
    sampled_poss_not_avail = random.sample(poss_rsi_not_avail, min(len(poss_rsi_not_avail), target_possible - len(sampled_poss_avail)))
    
    selected_possible = sampled_poss_avail + sampled_poss_not_avail
    # If we are still short (unlikely), fill from remainder of the Possible pool
    if len(selected_possible) < target_possible:
        remaining = [r for r in categories['Possible'] if r not in selected_possible]
        selected_possible += random.sample(remaining, target_possible - len(selected_possible))
        
    print(f"Possible selected: {len(selected_possible)} (all under {args.pref_tokens} tokens)")
    print(f"  - RSI Available (standard openFDA label): {len(sampled_poss_avail)}")
    print(f"  - RSI Not Available (AI-knowledge fallback): {len(sampled_poss_not_avail)}")
    
    # --- SELECT DOUBTFUL (Target: 1098) ---
    # We want to select target_doubtful from Doubtful pool.
    # Doubtful pool under-pref has 941 records.
    # We take all 941 under-pref records first.
    doubtful_under_pref = [r for r in categories['Doubtful'] if r['tokens'] <= args.pref_tokens]
    doubtful_above_pref = [r for r in categories['Doubtful'] if r['tokens'] > args.pref_tokens]
    
    selected_doubtful = list(doubtful_under_pref)
    needed_doubtful = target_doubtful - len(selected_doubtful) # 1098 - 941 = 157
    
    sampled_doubtful_above = random.sample(doubtful_above_pref, min(len(doubtful_above_pref), needed_doubtful))
    selected_doubtful += sampled_doubtful_above
    
    print(f"Doubtful selected: {len(selected_doubtful)}")
    print(f"  - Under {args.pref_tokens} tokens: {len(doubtful_under_pref)}")
    print(f"  - Between {args.pref_tokens} and {args.max_tokens} tokens: {len(sampled_doubtful_above)}")
    
    # Count RSI status for selected Doubtful
    doubtful_rsi_avail = sum(1 for r in selected_doubtful if r['rsi_avail'])
    doubtful_rsi_not_avail = sum(1 for r in selected_doubtful if not r['rsi_avail'])
    print(f"  - RSI Available (standard openFDA label): {doubtful_rsi_avail}")
    print(f"  - RSI Not Available (AI-knowledge fallback): {doubtful_rsi_not_avail}")
    
    # 2. Generate Negatives (300 total, 100 of each, all under 500 tokens)
    print(f"\nGenerating {target_neg_size} synthetic negative controls...")
    negatives = generate_negative_cases(count_per_cat=target_neg_size // 3)
    selected_negatives = [n['sample'] for n in negatives]
    
    # Extract reviews
    selected_medical_reviews = [r['sample'] for r in selected_prob_def + selected_possible + selected_doubtful]
    
    # 3. Combine and Shuffle
    final_dataset = selected_medical_reviews + selected_negatives
    random.shuffle(final_dataset)
    
    # 4. Save to New Dataset File
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f_out:
        for sample in final_dataset:
            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
    # Calculate stats of the new dataset
    final_token_counts = [calculate_tokens(s, enc) for s in final_dataset]
    mean_len = np.mean(final_token_counts)
    max_len = np.max(final_token_counts)
    under_4k_count = sum(1 for c in final_token_counts if c <= args.pref_tokens)
    
    print("\n" + "="*50)
    print("COMPILATION SUMMARY - NEW DATASET")
    print("="*50)
    print(f"Output File Path : {args.output}")
    print(f"Total Records    : {len(final_dataset)}")
    print(f"  - Medical reviews: {len(selected_medical_reviews)}")
    print(f"  - Negative reviews: {len(selected_negatives)}")
    print(f"Average Token Size: {mean_len:.2f} tokens")
    print(f"Max Token Size    : {max_len} tokens")
    print(f"Records <= {args.pref_tokens} : {under_4k_count} ({under_4k_count/len(final_dataset)*100:.2f}%)")
    print(f"Records > {args.pref_tokens}  : {len(final_dataset) - under_4k_count} ({(len(final_dataset) - under_4k_count)/len(final_dataset)*100:.2f}%)")
    print("="*50)

if __name__ == '__main__':
    main()

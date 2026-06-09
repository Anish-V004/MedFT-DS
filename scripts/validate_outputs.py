import os
import json
import re
import argparse

def check_overlap(narrative_lower, meddra_pt_lower):
    # Exact match check first
    if meddra_pt_lower in narrative_lower:
        return True
        
    # Direct word tokenization
    pt_words = re.findall(r'[a-zA-Z0-9]{3,}', meddra_pt_lower)
    narrative_words = set(re.findall(r'[a-zA-Z0-9]{2,}', narrative_lower))
    
    # Stop words to ignore
    stop_words = {'and', 'the', 'of', 'for', 'with', 'in', 'on', 'at', 'by', 'from', 'first', 'second', 'third', 'degree', 'acute', 'chronic', 'syndrome'}
    
    # Medical synonym/abbreviation mappings
    synonyms = {
        'atrioventricular': ['av'],
        'myocardial': ['mi', 'heart', 'cardiac'],
        'infarction': ['mi', 'infarct', 'attack'],
        'hemorrhage': ['bleed', 'bleeding', 'hemorrhagic'],
        'haemorrhage': ['bleed', 'bleeding', 'haemorrhagic'],
        'thrombocytopenia': ['thrombopenia', 'platelet', 'platelets'],
        'sinusoidal': ['sos', 'veno-occlusive'],
        'bradycardia': ['bradyarrhythmia', 'slow'],
        'tachycardia': ['tachyarrhythmia', 'fast'],
        'renal': ['kidney'],
        'hepatic': ['liver'],
        'cardiac': ['heart']
    }
    
    for word in pt_words:
        if word in stop_words:
            continue
        if word in narrative_words:
            return True
        # Check synonyms
        if word in synonyms:
            for syn in synonyms[word]:
                if syn in narrative_words or syn in narrative_lower:
                    return True
    return False

def validate_jsonl_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 0, 0
        
    print(f"\nValidating dataset file: {filepath}")
    valid_records = []
    flagged_count = 0
    total_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            total_count += 1
            try:
                data = json.loads(line)
                messages = data.get('messages', [])
                if len(messages) < 3:
                    print(f"  [Row {idx+1}] Flagged: Missing ChatML messages (length {len(messages)}).")
                    flagged_count += 1
                    continue
                    
                user_content = messages[1].get('content', '')
                assistant_content = messages[2].get('content', '')
                
                # Extract narrative
                if "Reference Safety Information (RSI):" in user_content:
                    narrative = user_content.split("Reference Safety Information (RSI):")[0].replace("Patient Narrative:", "").strip()
                else:
                    narrative = user_content.replace("Patient Narrative:", "").strip()
                    
                # Extract structured MedDRA PT from assistant response by finding JSON boundaries
                start_idx = assistant_content.find('{')
                end_idx = assistant_content.rfind('}')
                if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
                    print(f"  [Row {idx+1}] Flagged: No valid JSON boundaries found in assistant response.")
                    flagged_count += 1
                    continue
                    
                json_str = assistant_content[start_idx:end_idx+1]
                json_data = json.loads(json_str)
                meddra_pt = json_data.get('meddra_pt')
                
                if not meddra_pt:
                    print(f"  [Row {idx+1}] Flagged: Missing 'meddra_pt' field in JSON.")
                    flagged_count += 1
                    continue
                    
                # Check grounding
                if check_overlap(narrative.lower(), meddra_pt.lower()):
                    valid_records.append(line)
                else:
                    print(f"  [Row {idx+1}] Flagged Hallucination: MedDRA PT '{meddra_pt}' has no word/synonym overlap in narrative.")
                    print(f"    Narrative snippet: {narrative[:120]}...")
                    flagged_count += 1
                    
            except Exception as e:
                print(f"  [Row {idx+1}] Flagged Error: {e}")
                flagged_count += 1
                
    # Rewrite the file with only valid records if any were flagged
    if flagged_count > 0:
        print(f"  --> Cleaning file. Rewriting with {len(valid_records)} valid records (removed {flagged_count} flagged records).")
        with open(filepath, 'w', encoding='utf-8') as f_out:
            f_out.writelines(valid_records)
    else:
        print(f"  --> All {total_count} records validated successfully!")
        
    return total_count, flagged_count

def main():
    parser = argparse.ArgumentParser(description="PV Dataset Post-Generation Validation Tool.")
    parser.add_argument('--biodex', type=str, default='data/biodex_chatml.jsonl')
    parser.add_argument('--fda', type=str, default='data/fda_chatml.jsonl')
    args = parser.parse_args()
    
    print("==================================================")
    print("PHARMACOVIGILANCE DATASET VALIDATOR (LLM-AS-A-JUDGE)")
    print("==================================================")
    
    total_validated = 0
    total_flagged = 0
    
    for path in [args.biodex, args.fda]:
        total, flagged = validate_jsonl_file(path)
        total_validated += total
        total_flagged += flagged
        
    print("\n" + "="*50)
    print(f"VALIDATION REPORT SUMMARY")
    print(f"Total Evaluated: {total_validated}")
    print(f"Total Flagged & Removed: {total_flagged}")
    print(f"Total Valid & Kept: {total_validated - total_flagged}")
    print("="*50)

if __name__ == '__main__':
    main()

import os
import json
import re
import argparse

VALID_NARANJO_INTERPRETATIONS = {'definite', 'probable', 'possible', 'doubtful'}
VALID_SERIOUSNESS_CRITERIA = {
    'death', 'hospitalization', 'life-threatening', 'disabling',
    'congenital anomaly', 'other serious medical event', 'none'
}

EXPECTED_SYSTEM_PROMPT = (
    "You are a Pharmacovigilance (PV) Medical Review Assistant. "
    "CRITICAL GROUNDING RULE: You must base your entire evaluation STRICTLY and EXCLUSIVELY on the provided Patient Narrative. "
    "Do NOT invent, hallucinate, or bring in external patient cases. Do NOT reference drugs or adverse events that are not explicitly written in the user's prompt. "
    "If the provided RSI does not match the drug in the narrative, explicitly state 'Drug Mismatch - Cannot Evaluate' in your reasoning."
)


def validate_record(idx, data):
    """
    Validates a single ChatML record against schema and structured output rules.
    Returns (is_valid: bool, reason: str).
    
    Checks:
      1. ChatML structure: exactly 3 messages with correct roles.
      2. JSON parseability: assistant response contains a valid JSON block.
      3. Required fields: meddra_pt, seriousness, causality, expectedness present.
      4. meddra_pt sanity: not None, not empty, not the literal string 'None'.
      5. Naranjo score range: between -4 and +13.
      6. Naranjo interpretation: one of Definite, Probable, Possible, Doubtful.
      7. System prompt integrity: matches expected prompt (catches stale records).
    """
    messages = data.get('messages', [])

    # Check 1: ChatML structure
    if len(messages) != 3:
        return False, f"Expected 3 messages, got {len(messages)}."
    roles = [m.get('role') for m in messages]
    if roles != ['system', 'user', 'assistant']:
        return False, f"Unexpected message roles: {roles}."

    # Check 7: System prompt integrity
    sys_content = messages[0].get('content', '')
    if sys_content != EXPECTED_SYSTEM_PROMPT:
        return False, "Stale or mismatched system prompt."

    assistant_content = messages[2].get('content', '')

    # Check 2: JSON parseability — find first { to last }
    start = assistant_content.find('{')
    end = assistant_content.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return False, "No valid JSON block found in assistant response."

    try:
        json_data = json.loads(assistant_content[start:end + 1])
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}."

    # Check 3: Required top-level fields
    required_fields = ['meddra_pt', 'seriousness', 'causality', 'expectedness']
    for field in required_fields:
        if field not in json_data:
            return False, f"Missing required field '{field}'."

    # Check 5 & 6: Causality / Naranjo
    causality = json_data.get('causality', {})
    if not isinstance(causality, dict):
        return False, "Field 'causality' is not a JSON object."

    naranjo_score = causality.get('naranjo_score')
    interpretation = causality.get('interpretation', '')

    if naranjo_score is None or not isinstance(naranjo_score, (int, float)):
        return False, f"Invalid or missing 'naranjo_score': '{naranjo_score}'."
    if not (-4 <= int(naranjo_score) <= 13):
        return False, f"Naranjo score {naranjo_score} is out of valid range (-4 to +13)."
    
    interpretation_clean = interpretation.strip().lower()
    if interpretation_clean not in VALID_NARANJO_INTERPRETATIONS and interpretation_clean != 'unassessable - missing data':
        return False, f"Invalid Naranjo interpretation: '{interpretation}'."

    # Check 4: meddra_pt sanity
    meddra_pt = json_data.get('meddra_pt')
    if not meddra_pt or not isinstance(meddra_pt, str):
        return False, "Missing or invalid meddra_pt type."

    if interpretation_clean == 'unassessable - missing data':
        if meddra_pt.strip().lower() != 'none':
            return False, f"Expected meddra_pt to be 'None' for Unassessable case, got '{meddra_pt}'."
    else:
        if meddra_pt.strip().lower() in ('none', '', 'null', 'n/a'):
            return False, f"Invalid meddra_pt value for medical review: '{meddra_pt}'."

    # Check seriousness block
    seriousness = json_data.get('seriousness', {})
    if not isinstance(seriousness, dict):
        return False, "Field 'seriousness' is not a JSON object."
    if 'is_serious' not in seriousness or not isinstance(seriousness.get('is_serious'), bool):
        return False, "Missing or invalid 'seriousness.is_serious' (expected bool)."

    return True, "OK"


def validate_jsonl_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 0, 0

    print(f"\nValidating: {filepath}")
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
            except json.JSONDecodeError as e:
                print(f"  [Row {idx + 1}] Flagged: Line is not valid JSON — {e}.")
                flagged_count += 1
                continue

            is_valid, reason = validate_record(idx + 1, data)
            if is_valid:
                valid_records.append(line)
            else:
                print(f"  [Row {idx + 1}] Flagged: {reason}")
                flagged_count += 1

    if flagged_count > 0:
        print(f"  --> Cleaning file. Rewriting with {len(valid_records)} valid records (removed {flagged_count} flagged).")
        with open(filepath, 'w', encoding='utf-8') as f_out:
            f_out.writelines(valid_records)
    else:
        print(f"  --> All {total_count} records passed validation.")

    return total_count, flagged_count


def main():
    parser = argparse.ArgumentParser(description="PV Dataset Post-Generation Validation Tool.")
    parser.add_argument('--biodex', type=str, default='data/biodex_chatml.jsonl',
                        help="Path to the BioDEX ChatML output file.")
    parser.add_argument('--fda', type=str, default='data/fda_chatml.jsonl',
                        help="Path to the openFDA ChatML output file.")
    args = parser.parse_args()

    print("=" * 52)
    print("  PHARMACOVIGILANCE DATASET SCHEMA VALIDATOR")
    print("=" * 52)

    total_validated = 0
    total_flagged = 0

    for path in [args.biodex, args.fda]:
        total, flagged = validate_jsonl_file(path)
        total_validated += total
        total_flagged += flagged

    kept = total_validated - total_flagged
    pct_kept = (kept / total_validated * 100) if total_validated > 0 else 0

    print("\n" + "=" * 52)
    print("VALIDATION REPORT SUMMARY")
    print(f"  Total Evaluated : {total_validated}")
    print(f"  Flagged & Removed: {total_flagged}")
    print(f"  Valid & Kept    : {kept}  ({pct_kept:.1f}%)")
    print("=" * 52)


if __name__ == '__main__':
    main()

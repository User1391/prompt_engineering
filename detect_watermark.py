#!/usr/bin/env python3

import sys
from watermark_data import SYNONYM_MAPPING, SIGNATURE_BITS, FORCED_WORDS

def main():
    """
    Reads text from stdin, checks for forced synonyms,
    and prints whether the text is watermarked and which bits were detected.
    """
    # 1) Read all incoming text from stdin
    text = sys.stdin.read().lower()

    # 2) Build the "expected synonyms" list in the same order as insertion
    forced_synonyms = []
    for i, word in enumerate(FORCED_WORDS):
        bit = SIGNATURE_BITS[i]
        synonyms = SYNONYM_MAPPING[word]
        if bit >= len(synonyms):
            bit = len(synonyms) - 1
        forced_synonym = synonyms[bit]
        forced_synonyms.append((word, forced_synonym))

    # 3) Check whether each forced synonym appears in the text
    found_bits = []
    for i, (orig_word, forced_syn) in enumerate(forced_synonyms):
        if forced_syn in text:
            # If we find this forced synonym, we assume the model used that rank => decode the bit
            bit = SIGNATURE_BITS[i]
            found_bits.append(bit)

    # 4) Summarize results
    if len(found_bits) == len(forced_synonyms):
        print("Watermarked text detected!")
        print(f"Found all forced synonyms: {[syn for (_, syn) in forced_synonyms]}")
        print(f"Decoded bits: {found_bits}")
    else:
        print("This text does not appear to be fully watermarked.")
        if len(found_bits) > 0:
            print(f"Found partial synonyms => bits: {found_bits}")
        else:
            print("No forced synonyms recognized.")

def detect_watermark(text):
    """Enhanced watermark detection with redundancy checking"""
    # Original detection
    primary_bits = detect_primary_signature(text)
    
    # Check for redundant patterns
    redundant_bits = detect_redundant_patterns(text)
    
    # Check for synonym patterns
    synonym_consistency = check_synonym_consistency(text)
    
    # Combine evidence
    confidence_score = calculate_confidence(
        primary_bits, 
        redundant_bits,
        synonym_consistency
    )
    
    return confidence_score > CONFIDENCE_THRESHOLD

if __name__ == "__main__":
    main()


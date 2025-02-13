#!/usr/bin/env python3

import sys
from watermark_data import (
    SYNONYM_MAPPING, SIGNATURE_BITS, FORCED_WORDS,
    STANDARD_SIGNATURE_BITS, STANDARD_FORCED_WORDS,
    TECHNICAL_SIGNATURE_BITS, TECHNICAL_FORCED_WORDS
)

def detect_watermark_pattern(text, pattern_bits, pattern_words):
    """
    Check for a specific watermark pattern in the text.
    Returns (is_watermarked, found_bits, found_synonyms)
    """
    text = text.lower()
    forced_synonyms = []
    
    # Build expected synonyms list
    for i, word in enumerate(pattern_words):
        bit = pattern_bits[i]
        synonyms = SYNONYM_MAPPING[word]
        if bit >= len(synonyms):
            bit = len(synonyms) - 1
        forced_synonym = synonyms[bit]
        forced_synonyms.append((word, forced_synonym))

    # Check for synonyms and decode bits
    found_bits = []
    found_synonyms = []
    for i, (orig_word, forced_syn) in enumerate(forced_synonyms):
        if forced_syn in text:
            bit = pattern_bits[i]
            found_bits.append(bit)
            found_synonyms.append(forced_syn)

    is_watermarked = len(found_bits) == len(pattern_bits)
    return is_watermarked, found_bits, found_synonyms

def main():
    """
    Reads text from stdin and checks for watermarks using different patterns.
    """
    # Read input text
    text = sys.stdin.read()
    
    # Try standard pattern first
    is_watermarked, found_bits, found_synonyms = detect_watermark_pattern(
        text, STANDARD_SIGNATURE_BITS, STANDARD_FORCED_WORDS
    )
    
    if is_watermarked:
        print("Watermarked text detected!")
        print(f"Found all forced synonyms: {found_synonyms}")
        print(f"Decoded bits: {found_bits}")
        return
    
    # Try technical pattern
    is_watermarked, found_bits, found_synonyms = detect_watermark_pattern(
        text, TECHNICAL_SIGNATURE_BITS, TECHNICAL_FORCED_WORDS
    )
    
    if is_watermarked:
        print("Watermarked text detected! (Technical pattern)")
        print(f"Found all forced synonyms: {found_synonyms}")
        print(f"Decoded bits: {found_bits}")
        return
    
    # No complete watermark found
    print("This text does not appear to be fully watermarked.")
    if found_bits:  # Use results from last attempt
        print(f"Found partial synonyms => bits: {found_bits}")
    else:
        print("No forced synonyms recognized.")

if __name__ == "__main__":
    main()


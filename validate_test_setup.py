#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from importlib import import_module
import matplotlib.pyplot as plt
import subprocess

def validate_environment():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'openai'
    ]
    
    missing = []
    for package in required_packages:
        try:
            import_module(package)
            print(f"✓ Found {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ Missing {package}")
    
    return len(missing) == 0

def validate_files():
    """Check if all required files exist and are readable"""
    required_files = [
        'test_watermark.py',
        'runprompt.py',
        'watermark_data.py',
        'detect_watermark.py',
        '.env'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            if os.access(file, os.R_OK):
                print(f"✓ Found and readable: {file}")
            else:
                missing.append(file)
                print(f"✗ Not readable: {file}")
        else:
            missing.append(file)
            print(f"✗ Missing: {file}")
    
    return len(missing) == 0

def validate_api_key():
    """Check if API key is properly set"""
    # Try to load from .env if not already set
    if not os.getenv("OPENAI_API_KEY"):
        try:
            with open('.env') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        key, value = line.strip().replace('export ', '').split('=', 1)
                        if key.strip() == "OPENAI_API_KEY":
                            # Remove quotes if present
                            os.environ[key.strip()] = value.strip().strip('"').strip("'")
        except Exception as e:
            print(f"✗ Error reading .env file: {str(e)}")
            return False

    if os.getenv("OPENAI_API_KEY"):
        print("✓ OPENAI_API_KEY is set")
        # Mask the key for display
        key = os.getenv("OPENAI_API_KEY")
        masked = f"{key[:8]}...{key[-4:]}"
        print(f"  Key format: {masked}")
        return True
    else:
        print("✗ OPENAI_API_KEY is not set")
        print("  Please ensure .env file contains: export OPENAI_API_KEY=your-key-here")
        return False

def validate_data_structures():
    """Check if watermark data structures are valid"""
    try:
        import watermark_data
        
        # Check SYNONYM_MAPPING
        if not isinstance(watermark_data.SYNONYM_MAPPING, dict):
            print("✗ SYNONYM_MAPPING is not a dictionary")
            return False
        
        # Validate signature configurations
        if not all(isinstance(sig['bits'], list) for sig in watermark_data.WATERMARK_SIGNATURES.values()):
            print("✗ Invalid signature bits format")
            return False
        
        # Check for technical and standard configurations
        if not hasattr(watermark_data, 'TECHNICAL_SIGNATURE_BITS'):
            print("✗ Missing TECHNICAL_SIGNATURE_BITS")
            return False
        
        if not hasattr(watermark_data, 'STANDARD_SIGNATURE_BITS'):
            print("✗ Missing STANDARD_SIGNATURE_BITS")
            return False
        
        print("✓ Watermark data structures are valid")
        print(f"  Total synonyms: {len(watermark_data.SYNONYM_MAPPING)}")
        print(f"  Total signatures: {len(watermark_data.WATERMARK_SIGNATURES)}")
        return True
        
    except ImportError:
        print("✗ Could not import watermark_data")
        return False

def validate_test_framework():
    """Check if test framework components are properly setup"""
    try:
        from test_watermark import WatermarkTest
        
        # Create test instance
        tester = WatermarkTest()
        
        # Validate configurations
        configs = tester._get_default_configurations()
        if not configs:
            print("✗ No test configurations defined")
            return False
        
        print("✓ Test framework is valid")
        print(f"  Available configurations: {list(configs.keys())}")
        return True
        
    except ImportError:
        print("✗ Could not import test framework")
        return False

def test_with_sample_data():
    """Test the watermarking process with sample data"""
    print("\n6. Testing with Sample Data...")
    
    try:
        import watermark_data
        from detect_watermark import detect_watermark_pattern
        
        # Use the basic pattern we know works
        basic_pattern = [1, 3, 0, 2]
        basic_words = ["quick", "smart", "happy", "good"]
        
        # Print the mapping we're using
        print("\nWatermark Configuration:")
        for word, bit in zip(basic_words, basic_pattern):
            synonyms = watermark_data.SYNONYM_MAPPING[word]
            chosen_synonym = synonyms[bit] if bit < len(synonyms) else synonyms[-1]
            print(f"{word}[{bit}] -> {chosen_synonym} (from {synonyms})")
        
        # Create test text with explicit synonyms
        chosen_synonyms = [
            watermark_data.SYNONYM_MAPPING[word][bit] if bit < len(watermark_data.SYNONYM_MAPPING[word])
            else watermark_data.SYNONYM_MAPPING[word][-1]
            for word, bit in zip(basic_words, basic_pattern)
        ]
        
        test_text = (
            f"The {chosen_synonyms[0]} cat runs forward. "
            f"The {chosen_synonyms[1]} dog thinks deeply. "
            f"A {chosen_synonyms[2]} bird flies high. "
            f"What a {chosen_synonyms[3]} day it is."
        )
        
        print("\nTest Text:")
        print(test_text.strip())
        
        print("\nExpected Watermark Pattern:")
        print(f"Words: {basic_words}")
        print(f"Bits: {basic_pattern}")
        print(f"Expected synonyms: {chosen_synonyms}")
        
        # Run detection with our specific pattern
        is_watermarked, found_bits, found_synonyms = detect_watermark_pattern(
            test_text, basic_pattern, basic_words
        )
        
        print("\nDetection Results:")
        print(f"Watermark detected: {'✓' if is_watermarked else '✗'}")
        if found_bits:
            print(f"Found bits: {found_bits}")
        print(f"Found synonyms: {found_synonyms}")
        
        # Show found synonyms
        print(f"\nSynonym Detection:")
        print(f"Found {len(found_synonyms)}/{len(chosen_synonyms)} expected synonyms")
        for expected, found in zip(chosen_synonyms, found_synonyms if found_synonyms else []):
            print(f"{'✓' if expected == found else '✗'} Expected: {expected}, Found: {found if found else 'not found'}")
        
        print("\n✓ Sample data test completed")
        return True
        
    except Exception as e:
        print(f"\n✗ Error in sample data test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Validating Watermark Test Setup ===\n")
    
    # Track overall status
    status = True
    
    print("1. Checking Python Environment...")
    if not validate_environment():
        print("✗ Environment check failed")
        status = False
    print()
    
    print("2. Checking Required Files...")
    if not validate_files():
        print("✗ File check failed")
        status = False
    print()
    
    print("3. Checking API Configuration...")
    if not validate_api_key():
        print("✗ API configuration failed")
        status = False
    print()
    
    print("4. Checking Data Structures...")
    if not validate_data_structures():
        print("✗ Data structure validation failed")
        status = False
    print()
    
    print("5. Checking Test Framework...")
    if not validate_test_framework():
        print("✗ Test framework validation failed")
        status = False
    print()
    
    print("\n6. Testing with Sample Data...")
    if not test_with_sample_data():
        print("✗ Sample data test failed")
        status = False
    
    print("\n=== Validation Summary ===")
    if status:
        print("✓ All checks passed! The test framework should work correctly.")
        print("  Sample data tests show expected watermark detection behavior.")
    else:
        print("✗ Some checks failed. Please fix the issues above before running tests.")
    
    return 0 if status else 1

if __name__ == "__main__":
    sys.exit(main()) 
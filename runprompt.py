import os
import argparse
from openai import OpenAI
from watermark_data import (
    SYNONYM_MAPPING, SIGNATURE_BITS, FORCED_WORDS, 
    WATERMARK_SIGNATURES, update_active_signature, 
    WATERMARK_METADATA, TECHNICAL_SIGNATURE_BITS, TECHNICAL_FORCED_WORDS,
    STANDARD_SIGNATURE_BITS, STANDARD_FORCED_WORDS
)

##############################################################################
# 1) CONFIGURATION: API Key & Watermark Data
##############################################################################

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=api_key)

##############################################################################
# 2) HELPER FUNCTION: Build Watermark Instructions
##############################################################################

def print_verbose(message: str, verbose: bool = False):
    """Helper function to print debug information when verbose mode is enabled"""
    if verbose:
        print(f"[DEBUG] {message}")

def analyze_topic_context(topic):
    """Analyze topic to determine best watermarking strategy"""
    return {
        "formality": "formal" if any(w in topic.lower() for w in 
            ["research", "analysis", "professional"]) else "neutral",
        "length_target": "long" if any(w in topic.lower() for w in 
            ["explain", "describe", "analyze"]) else "medium",
        "complexity": "high" if any(w in topic.lower() for w in 
            ["technical", "advanced", "complex"]) else "normal"
    }

def build_watermark_instructions(topic: str, verbose: bool = False) -> str:
    """
    Construct a string instructing the model to embed synonyms.
    The model is asked to produce text on `topic` while using the forced synonyms.
    """
    context = analyze_topic_context(topic)
    # Adjust signature and synonyms based on context
    if context["complexity"] == "high":
        signature_bits = TECHNICAL_SIGNATURE_BITS
        forced_words = TECHNICAL_FORCED_WORDS
        description = "Technical pattern"
    else:
        signature_bits = STANDARD_SIGNATURE_BITS
        forced_words = STANDARD_FORCED_WORDS
        description = "Standard pattern"
    
    if verbose:
        print_verbose(f"Selected pattern: {'technical' if context['complexity'] == 'high' else 'standard'}", verbose)
        print_verbose(f"Signature bits: {signature_bits}", verbose)
        print_verbose(f"Forced words: {forced_words}", verbose)
        print_verbose(f"Description: {description}", verbose)
    
    lines = []
    lines.append(
        "You will produce a short text on a topic, embedding a hidden watermark.\n"
        "To do this, you MUST replace certain words with specific synonyms. Here are the rules:\n\n"
        f"RULES FOR WATERMARK (using {description}):"
    )

    for i, word in enumerate(forced_words):
        bit = signature_bits[i]
        possible_syns = SYNONYM_MAPPING[word]
        # Clamp if out of range
        if bit >= len(possible_syns):
            bit = len(possible_syns) - 1
        forced_synonym = possible_syns[bit]
        lines.append(
            f"- Whenever you want to use '{word}', use '{forced_synonym}'. (bit = {bit})"
        )
        if verbose:
            print_verbose(f"Word mapping: {word} -> {forced_synonym} (bit {bit})", verbose)

    lines.append(
        "\nImportant: use each forced synonym at least once so the watermark is embedded.\n"
        "If you mention the same concept again, keep using the same forced synonym. "
        "Ensure coherence.\n"
    )

    lines.append(f"TOPIC: {topic}")
    lines.append("Now produce your watermarked text:\n")

    return "\n".join(lines)


##############################################################################
# 3) MAIN SCRIPT: Prompt the Model
##############################################################################

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Generate watermarked text using GPT')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--topic', '-t', type=str, default="the importance of adapting to change",
                       help='Topic for text generation')
    args = parser.parse_args()

    if args.verbose:
        print_verbose("=== Watermark System Information ===", True)
        print_verbose(f"Version: {WATERMARK_METADATA['version']}", True)
        print_verbose(f"Total patterns: {WATERMARK_METADATA['total_patterns']}", True)
        print_verbose(f"Total words: {WATERMARK_METADATA['total_words']}", True)
        print_verbose(f"Avg synonyms per word: {WATERMARK_METADATA['average_synonyms_per_word']:.2f}", True)
        print_verbose("================================\n", True)

    # Build the watermark instructions
    prompt_instructions = build_watermark_instructions(args.topic, args.verbose)

    if args.verbose:
        print_verbose("=== Generated Instructions ===", True)
        print_verbose(prompt_instructions, True)
        print_verbose("============================\n", True)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that follows special watermark instructions."
        },
        {
            "role": "user",
            "content": prompt_instructions
        }
    ]

    if args.verbose:
        print_verbose("Sending request to OpenAI API...", True)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7
    )

    watermarked_text = response.choices[0].message.content

    print("\n=== Watermarked Output ===")
    print(watermarked_text)


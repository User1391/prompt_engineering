#!/usr/bin/env python3

import os
import subprocess
import json
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from openai import OpenAI
import argparse

# Load environment variables if not already set
if not os.getenv("OPENAI_API_KEY"):
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().replace('export ', '').split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"')
    else:
        raise ValueError(f"No {env_file} file found and OPENAI_API_KEY not set in environment")

# Now import our modules after environment is set up
from runprompt import client, build_watermark_instructions
from watermark_data import FORCED_WORDS, SYNONYM_MAPPING, SIGNATURE_BITS, get_optimal_signature_length, select_synonym, validate_synonym_usage, ENHANCED_SYNONYM_MAPPING, NATURAL_PATTERNS, WATERMARK_LAYERS

# Import and store original configuration
ORIGINAL_SYNONYM_MAPPING = SYNONYM_MAPPING.copy()
ORIGINAL_SIGNATURE_BITS = SIGNATURE_BITS.copy()
ORIGINAL_FORCED_WORDS = FORCED_WORDS.copy()

ENHANCED_CONFIGURATIONS = {
    "adaptive_length": {
        "use_adaptive_length": True,
        "use_frequency_selection": False,
        "use_context_aware": False
    },
    "frequency_based": {
        "use_adaptive_length": False,
        "use_frequency_selection": True,
        "use_context_aware": False
    },
    "multi_layer": {
        "use_adaptive_length": True,
        "use_frequency_selection": True,
        "use_context_aware": True,
        "layers": ["primary", "secondary"]
    }
}

def generate_watermarked_text(topic):
    """Generate text with watermark instructions"""
    prompt_instructions = build_watermark_instructions(topic)
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
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def generate_normal_text(topic):
    """Generate text without watermark instructions"""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": f"Write a short paragraph about: {topic}"
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

def detect_watermark(text):
    """Run detect_watermark.py on the text and parse results"""
    # Ensure text is a string before proceeding
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    
    process = subprocess.run(
        ['python3', 'detect_watermark.py'],
        input=text,  # Pass the string directly
        capture_output=True,
        text=True,   # This tells subprocess to handle text encoding/decoding
        encoding='utf-8'  # Explicitly specify encoding
    )
    
    output = process.stdout
    is_watermarked = "Watermarked text detected!" in output
    found_bits = []
    
    if "Decoded bits:" in output:
        bits_str = output.split("Decoded bits: ")[1].strip("[] \n")
        if bits_str:
            found_bits = [int(b.strip()) for b in bits_str.split(",") if b.strip()]
    
    return is_watermarked, found_bits, output

class WatermarkTest:
    def __init__(self):
        self.results_df = pd.DataFrame()
        self.test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store original configuration
        self.original_config = {
            'SYNONYM_MAPPING': ORIGINAL_SYNONYM_MAPPING,
            'SIGNATURE_BITS': ORIGINAL_SIGNATURE_BITS,
            'FORCED_WORDS': ORIGINAL_FORCED_WORDS
        }

    def run_comprehensive_tests(self, num_tests=50, configurations=None):
        """Run tests across different configurations"""
        if configurations is None:
            configurations = self._get_default_configurations()
        
        all_results = []
        
        for config_name, config in configurations.items():
            print(f"\n=== Testing Configuration: {config_name} ===")
            
            # Update watermark data with this configuration
            self._update_watermark_config(config)
            
            # Run tests for each category in this configuration
            results = self._run_configuration_tests(config, num_tests)
            
            # Add configuration info to results
            for result in results:
                if result:  # Skip None results
                    result['configuration'] = config_name
                    result['num_synonyms'] = config.get('avg_synonyms', 0)
                    result['signature_length'] = len(config.get('signature_bits', []))
                    all_results.append(result)
        
        # Convert all results to DataFrame
        self.results_df = pd.DataFrame(all_results)
        self._analyze_results()

    def _get_default_configurations(self):
        """Define different watermarking configurations to test"""
        return {
            "base_config": {
                "synonym_mapping": self.original_config['SYNONYM_MAPPING'],
                "signature_bits": self.original_config['SIGNATURE_BITS'],
                "forced_words": self.original_config['FORCED_WORDS'],
            },
            "extended_technical": {
                "synonym_mapping": {
                    **self.original_config['SYNONYM_MAPPING'],
                    "analyze": [
                        "examine", "investigate", "evaluate", "assess", "study",
                        "scrutinize", "inspect", "review", "probe", "explore"
                    ],
                    "develop": [
                        "create", "build", "design", "construct", "implement",
                        "establish", "formulate", "devise", "engineer", "architect"
                    ],
                    "improve": [
                        "enhance", "upgrade", "optimize", "refine", "advance",
                        "augment", "elevate", "strengthen", "boost", "perfect"
                    ],
                    "efficient": [
                        "effective", "productive", "streamlined", "optimized", "expedient",
                        "economical", "resourceful", "practical", "proficient", "capable"
                    ]
                },
                "signature_bits": [1, 3, 2, 4, 0, 2],
                "forced_words": ["analyze", "develop", "improve", "efficient", "think", "smart"]
            },
            "extended_critical": {
                "synonym_mapping": {
                    **self.original_config['SYNONYM_MAPPING'],
                    "problem": [
                        "issue", "challenge", "difficulty", "obstacle", "complication",
                        "predicament", "dilemma", "setback", "hurdle", "impediment"
                    ],
                    "risk": [
                        "danger", "hazard", "threat", "peril", "vulnerability",
                        "exposure", "liability", "jeopardy", "uncertainty", "concern"
                    ],
                    "urgent": [
                        "pressing", "critical", "crucial", "immediate", "vital",
                        "essential", "imperative", "paramount", "compelling", "dire"
                    ]
                },
                "signature_bits": [2, 1, 3, 0, 4],
                "forced_words": ["problem", "risk", "urgent", "bad", "slow"]
            },
            "simplified_pattern": {
                "synonym_mapping": self.original_config['SYNONYM_MAPPING'],
                "signature_bits": [1, 1, 1],  # Simpler pattern might be more reliable
                "forced_words": ["quick", "smart", "good"]
            },
            "complex_pattern": {
                "synonym_mapping": self.original_config['SYNONYM_MAPPING'],
                "signature_bits": [1, 3, 0, 2, 1, 4, 2],  # More complex pattern
                "forced_words": ["quick", "smart", "happy", "good", "think", "big", "learn"]
            }
        }

    def _update_watermark_config(self, config):
        """Update the watermark configuration globally"""
        # Import needed modules
        import watermark_data
        
        # Backup original values
        if not hasattr(self, '_original_config'):
            self._original_config = {
                'SYNONYM_MAPPING': watermark_data.SYNONYM_MAPPING.copy(),
                'SIGNATURE_BITS': watermark_data.SIGNATURE_BITS.copy(),
                'FORCED_WORDS': watermark_data.FORCED_WORDS.copy()
            }
        
        # Update global variables
        watermark_data.SYNONYM_MAPPING.update(config['synonym_mapping'])
        watermark_data.SIGNATURE_BITS = config['signature_bits']
        watermark_data.FORCED_WORDS = config['forced_words']

    def _run_configuration_tests(self, config, num_tests):
        """Run tests for a specific configuration"""
        results = []
        
        # Test categories and their topics
        test_categories = {
            "technical": [
                "artificial intelligence advances",
                "quantum computing research",
                "blockchain technology",
                "machine learning algorithms",
                "software engineering practices"
            ],
            "emotional": [
                "dealing with anxiety",
                "finding happiness",
                "emotional intelligence",
                "personal relationships",
                "mental health awareness"
            ],
            "neutral": [
                "history of agriculture",
                "transportation systems",
                "educational methods",
                "urban planning",
                "communication theory"
            ],
            "critical": [
                "environmental problems",
                "economic challenges",
                "social issues",
                "political conflicts",
                "healthcare crisis"
            ]
        }
        
        # Run tests for each category
        for category, topics in test_categories.items():
            print(f"\n=== Testing {category.title()} Content ===")
            
            # Test watermarked content
            for topic in topics:
                for i in range(num_tests // len(topics)):
                    result = self._run_single_test(topic, category, True)
                    results.append(result)
                    
            # Test non-watermarked content (control)
            for topic in topics:
                for i in range(num_tests // len(topics)):
                    result = self._run_single_test(topic, category, False)
                    results.append(result)
        
        return results

    def _run_single_test(self, topic, category, watermarked=True):
        """Enhanced test with additional analysis"""
        print(f"\nTesting: {topic} ({'watermarked' if watermarked else 'control'})")
        
        try:
            # Generate text
            if watermarked:
                text = generate_watermarked_text(topic)
            else:
                text = generate_normal_text(topic)
            
            # Basic watermark detection
            is_watermarked, found_bits, output = detect_watermark(text)
            
            # Calculate text metrics
            text_length = len(text.split())
            avg_word_length = sum(len(word) for word in text.split()) / text_length
            
            # Additional analysis
            optimal_sig_length = get_optimal_signature_length(text_length)
            
            # Natural language pattern analysis
            natural_scores = []
            for word, synonym in zip(self.original_config['FORCED_WORDS'], 
                                   [w for _, w in zip(range(len(found_bits)), text.split())]):
                if word in NATURAL_PATTERNS:
                    natural_scores.append(validate_synonym_usage(text, word, synonym))
            
            # Frequency analysis
            frequency_scores = []
            for word in text.split():
                for base_word, synonyms in ENHANCED_SYNONYM_MAPPING.items():
                    if word in [syn for syn, _ in synonyms]:
                        freq = next(freq for syn, freq in synonyms if syn == word)
                        frequency_scores.append(freq)
            
            # Multi-layer detection
            layer_detections = {}
            for layer_name, layer in WATERMARK_LAYERS.items():
                layer_bits = []
                for word, bit in zip(layer['words'], layer['bits']):
                    if any(syn in text.lower() for syn in SYNONYM_MAPPING.get(word, [])):
                        layer_bits.append(bit)
                layer_detections[layer_name] = len(layer_bits) / len(layer['bits'])
            
            return {
                'timestamp': datetime.now(),
                'category': category,
                'topic': topic,
                'watermarked_input': watermarked,
                'detected_watermark': is_watermarked,
                'num_bits_found': len(found_bits),
                'text_length': text_length,
                'avg_word_length': avg_word_length,
                'false_positive': not watermarked and is_watermarked,
                'false_negative': watermarked and not is_watermarked,
                'partial_detection': watermarked and len(found_bits) > 0 and not is_watermarked,
                # New metrics
                'optimal_signature_length': optimal_sig_length,
                'actual_vs_optimal_ratio': len(found_bits) / optimal_sig_length if optimal_sig_length > 0 else 0,
                'natural_language_score': sum(natural_scores) / len(natural_scores) if natural_scores else 0,
                'avg_frequency_score': sum(frequency_scores) / len(frequency_scores) if frequency_scores else 0,
                'primary_layer_detection': layer_detections.get('primary', 0),
                'secondary_layer_detection': layer_detections.get('secondary', 0),
                'verification_layer_detection': layer_detections.get('verification', 0)
            }
            
        except Exception as e:
            print(f"Error in test: {str(e)}")
            return None

    def _analyze_results(self):
        """Enhanced analysis with new metrics"""
        results = self.results_df
        output_dir = f"results_{self.test_timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic statistics
        stats_dict = {
            'total_tests': len(results),
            'watermarked_tests': len(results[results['watermarked_input']]),
            'control_tests': len(results[~results['watermarked_input']]),
            'successful_detections': len(results[results['detected_watermark'] & results['watermarked_input']]),
            'false_positives': len(results[results['false_positive']]),
            'false_negatives': len(results[results['false_negative']]),
            'partial_detections': len(results[results['partial_detection']])
        }
        
        # Calculate detection rates by category
        category_stats = results[results['watermarked_input']].groupby('category').agg({
            'detected_watermark': 'mean',
            'false_negative': 'mean',
            'partial_detection': 'mean'
        })
        
        # Statistical tests
        contingency = pd.crosstab(
            results[results['watermarked_input']]['category'],
            results[results['watermarked_input']]['detected_watermark']
        )
        chi2, p_value = stats.chi2_contingency(contingency)[:2]
        
        # Select only numeric columns for correlation analysis
        numeric_columns = [
            'detected_watermark', 'num_bits_found', 'text_length', 
            'avg_word_length', 'false_positive', 'false_negative',
            'partial_detection', 'optimal_signature_length',
            'actual_vs_optimal_ratio', 'natural_language_score',
            'avg_frequency_score', 'primary_layer_detection',
            'secondary_layer_detection', 'verification_layer_detection'
        ]
        
        # Correlation analysis on numeric columns only
        correlations = results[results['watermarked_input']][numeric_columns].corr()
        
        # Save results
        with open(f"{output_dir}/summary_statistics.txt", 'w') as f:
            f.write("=== Watermark Detection Analysis ===\n\n")
            
            f.write("Basic Statistics:\n")
            for key, value in stats_dict.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nDetection Rates by Category:\n")
            f.write(category_stats.to_string())
            
            f.write("\n\nStatistical Tests:\n")
            f.write(f"Chi-square test for category independence:\n")
            f.write(f"chi2 = {chi2:.2f}, p-value = {p_value:.4f}\n")
            
            # Calculate confidence intervals
            z = 1.96  # 95% confidence level
            n = stats_dict['watermarked_tests']
            p = stats_dict['successful_detections'] / n
            ci = z * np.sqrt(p * (1-p) / n)
            f.write(f"\nSuccess Rate: {p:.2%} ± {ci:.2%} (95% CI)\n")
        
        # Save correlation analysis
        with open(f"{output_dir}/correlation_analysis.txt", 'w') as f:
            f.write("=== Correlation Analysis ===\n\n")
            f.write("Correlations with Detection Success:\n")
            f.write(correlations['detected_watermark'].to_string())
        
        # Additional configuration-specific analysis
        config_stats = results[results['watermarked_input']].groupby('configuration').agg({
            'detected_watermark': ['mean', 'count'],
            'false_negative': 'mean',
            'partial_detection': 'mean',
            'num_synonyms': 'first',
            'signature_length': 'first'
        })
        
        # Save enhanced results
        with open(f"{output_dir}/configuration_analysis.txt", 'w') as f:
            f.write("=== Configuration Performance Analysis ===\n\n")
            f.write(config_stats.to_string())
        
        # New analyses
        with open(f"{output_dir}/detailed_analysis.txt", 'w') as f:
            f.write("=== Detailed Watermark Analysis ===\n\n")
            
            # Signature length analysis
            f.write("Signature Length Analysis:\n")
            sig_length_stats = results.groupby('optimal_signature_length').agg({
                'detected_watermark': ['mean', 'count'],
                'actual_vs_optimal_ratio': 'mean'
            })
            f.write(sig_length_stats.to_string() + "\n\n")
            
            # Natural language analysis
            f.write("Natural Language Scores:\n")
            natural_stats = results[results['watermarked_input']].groupby(
                pd.qcut(results['natural_language_score'], 4)
            ).agg({
                'detected_watermark': 'mean',
                'false_negative': 'mean'
            })
            f.write(natural_stats.to_string() + "\n\n")
            
            # Frequency analysis
            f.write("Frequency Score Impact:\n")
            freq_stats = results[results['watermarked_input']].groupby(
                pd.qcut(results['avg_frequency_score'], 4)
            ).agg({
                'detected_watermark': 'mean',
                'false_negative': 'mean'
            })
            f.write(freq_stats.to_string() + "\n\n")
            
            # Multi-layer analysis
            f.write("Multi-layer Detection Rates:\n")
            layer_stats = results[results['watermarked_input']].agg({
                'primary_layer_detection': 'mean',
                'secondary_layer_detection': 'mean',
                'verification_layer_detection': 'mean'
            })
            f.write(layer_stats.to_string() + "\n")
        
        # Additional visualizations
        self._generate_enhanced_plots()

    def _generate_enhanced_plots(self):
        """Generate additional visualization plots"""
        output_dir = f"results_{self.test_timestamp}"
        
        # Configuration comparison plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=self.results_df[self.results_df['watermarked_input']],
            x='configuration',
            y='detected_watermark'
        )
        plt.xticks(rotation=45)
        plt.title('Detection Success Rate by Configuration')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/configuration_comparison.png")
        
        # Signature length vs detection success
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=self.results_df[self.results_df['watermarked_input']],
            x='signature_length',
            y='detected_watermark'
        )
        plt.title('Detection Success vs Signature Length')
        plt.savefig(f"{output_dir}/signature_length_analysis.png")
        
        # Natural language score vs detection success
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.results_df[self.results_df['watermarked_input']],
            x='natural_language_score',
            y='detected_watermark',
            hue='category'
        )
        plt.title('Natural Language Score vs Detection Success')
        plt.savefig(f"{output_dir}/natural_language_analysis.png")
        
        # Layer detection comparison
        layer_data = self.results_df[self.results_df['watermarked_input']][
            ['primary_layer_detection', 'secondary_layer_detection', 'verification_layer_detection']
        ].melt()
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=layer_data, x='variable', y='value')
        plt.title('Detection Rates by Watermark Layer')
        plt.savefig(f"{output_dir}/layer_detection_analysis.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run watermark detection tests')
    parser.add_argument('--num-tests', type=int, default=50,
                       help='Number of tests per configuration')
    parser.add_argument('--config', choices=['all', 'base', 'technical', 'critical', 'simple', 'complex'],
                       default='all', help='Which configuration to test')
    args = parser.parse_args()
    
    tester = WatermarkTest()
    
    if args.config == 'all':
        tester.run_comprehensive_tests(num_tests=args.num_tests)
    else:
        configs = tester._get_default_configurations()
        config_map = {
            'base': {'base_config': configs['base_config']},
            'technical': {'extended_technical': configs['extended_technical']},
            'critical': {'extended_critical': configs['extended_critical']},
            'simple': {'simplified_pattern': configs['simplified_pattern']},
            'complex': {'complex_pattern': configs['complex_pattern']}
        }
        tester.run_comprehensive_tests(num_tests=args.num_tests, 
                                     configurations=config_map[args.config]) 
# watermark_data.py

# A large dictionary mapping words to many possible synonyms.
# Each list can contain as many synonyms as you like.
SYNONYM_MAPPING = {
    # Movement/Speed related
    "quick": [
        "fast", "speedy", "brisk", "prompt", "rapid", "swift", "fleet-footed",
        "meteoric", "lightning-fast", "hasty", "expeditious", "nimble"
    ],
    "slow": [
        "unhurried", "lethargic", "sluggish", "languid", "gradual", "torpid",
        "snail-paced", "unrushed", "plodding", "dawdling", "leisurely"
    ],
    "walk": [
        "stroll", "amble", "saunter", "stride", "trek", "march", "wander",
        "meander", "traipse", "ramble", "promenade"
    ],
    
    # Intelligence/Knowledge related
    "smart": [
        "clever", "bright", "sharp", "brainy", "savvy", "keen", "intelligent",
        "knowledgeable", "astute", "brilliant", "sagacious", "ingenious"
    ],
    "learn": [
        "study", "grasp", "master", "absorb", "comprehend", "acquire",
        "assimilate", "digest", "perceive", "apprehend", "discover"
    ],
    "think": [
        "ponder", "contemplate", "reflect", "consider", "deliberate", "meditate",
        "ruminate", "muse", "cogitate", "reason", "analyze"
    ],
    
    # Emotional states
    "happy": [
        "joyful", "cheerful", "glad", "delighted", "elated", "upbeat", "sunny",
        "content", "blissful", "jubilant", "ecstatic", "merry"
    ],
    "sad": [
        "melancholy", "gloomy", "dejected", "downcast", "sorrowful", "morose",
        "despondent", "forlorn", "woeful", "disheartened", "crestfallen"
    ],
    "angry": [
        "furious", "enraged", "irate", "wrathful", "indignant", "outraged",
        "incensed", "livid", "seething", "irritated", "vexed"
    ],
    
    # Size/Magnitude
    "big": [
        "large", "enormous", "massive", "gigantic", "colossal", "immense",
        "vast", "substantial", "tremendous", "huge", "mammoth"
    ],
    "small": [
        "tiny", "minute", "diminutive", "compact", "modest", "petite",
        "miniature", "little", "microscopic", "slight", "minimal"
    ],
    
    # Quality/Value
    "good": [
        "excellent", "superb", "outstanding", "magnificent", "splendid",
        "wonderful", "fantastic", "marvelous", "exceptional", "superior"
    ],
    "bad": [
        "poor", "inferior", "subpar", "inadequate", "unsatisfactory",
        "disappointing", "dreadful", "awful", "terrible", "atrocious"
    ],
    
    # Time-related
    "now": [
        "immediately", "instantly", "presently", "currently", "promptly",
        "straightaway", "directly", "forthwith", "pronto", "swiftly"
    ],
    "old": [
        "ancient", "aged", "vintage", "antique", "traditional", "classic",
        "archaic", "timeworn", "venerable", "prehistoric", "primordial"
    ],
    
    # Technical Process Words
    "analyze": [
        "examine", "investigate", "evaluate", "assess", "study",
        "scrutinize", "inspect", "review", "probe", "explore",
        "dissect", "diagnose", "research", "audit", "survey"
    ],
    "develop": [
        "create", "build", "design", "construct", "implement",
        "establish", "formulate", "devise", "engineer", "architect",
        "innovate", "generate", "produce", "compose", "synthesize"
    ],
    "improve": [
        "enhance", "upgrade", "optimize", "refine", "advance",
        "augment", "elevate", "strengthen", "boost", "perfect",
        "streamline", "revolutionize", "modernize", "transform", "progress"
    ],
    
    # Technical Quality Words
    "efficient": [
        "effective", "productive", "streamlined", "optimized", "expedient",
        "economical", "resourceful", "practical", "proficient", "capable",
        "systematic", "methodical", "organized", "strategic", "precise"
    ],
    "innovative": [
        "groundbreaking", "pioneering", "revolutionary", "cutting-edge", "advanced",
        "novel", "original", "inventive", "creative", "progressive",
        "disruptive", "transformative", "state-of-the-art", "leading-edge", "breakthrough"
    ],
    
    # Critical Analysis Words
    "problem": [
        "issue", "challenge", "difficulty", "obstacle", "complication",
        "predicament", "dilemma", "setback", "hurdle", "impediment",
        "barrier", "limitation", "constraint", "bottleneck", "roadblock"
    ],
    "risk": [
        "danger", "hazard", "threat", "peril", "vulnerability",
        "exposure", "liability", "jeopardy", "uncertainty", "concern",
        "probability", "likelihood", "possibility", "contingency", "factor"
    ],
    "urgent": [
        "pressing", "critical", "crucial", "immediate", "vital",
        "essential", "imperative", "paramount", "compelling", "dire",
        "emergency", "priority", "time-sensitive", "acute", "serious"
    ],
    
    # Critical State Words
    "deficient": [
        "inadequate", "insufficient", "lacking", "subpar", "flawed",
        "defective", "imperfect", "incomplete", "limited", "poor",
        "unsatisfactory", "shortcoming", "weak", "faulty", "compromised"
    ],
    "severe": [
        "serious", "grave", "critical", "extreme", "intense",
        "acute", "harsh", "drastic", "significant", "substantial",
        "considerable", "major", "profound", "devastating", "crucial"
    ],
    
    # Technical Action Words
    "implement": [
        "execute", "deploy", "install", "integrate", "incorporate",
        "realize", "actualize", "establish", "initiate", "launch",
        "introduce", "activate", "operationalize", "apply", "utilize"
    ],
    "optimize": [
        "maximize", "enhance", "improve", "streamline", "perfect",
        "refine", "upgrade", "calibrate", "tune", "adjust",
        "fine-tune", "polish", "heighten", "strengthen", "boost"
    ],
    
    # Critical Action Words
    "mitigate": [
        "reduce", "minimize", "alleviate", "diminish", "lessen",
        "moderate", "ease", "soften", "temper", "control",
        "contain", "limit", "decrease", "curtail", "manage"
    ],
    "resolve": [
        "solve", "address", "handle", "settle", "fix",
        "remedy", "rectify", "correct", "overcome", "eliminate",
        "clear", "reconcile", "determine", "conclude", "finalize"
    ]
}

# Different signature patterns for watermarking
WATERMARK_SIGNATURES = {
    "signature_1": {
        "bits": [1, 3, 0, 2],
        "words": ["quick", "smart", "happy", "slow"],
        "description": "Basic emotional-cognitive pattern"
    },
    "signature_2": {
        "bits": [2, 1, 4, 3, 0],
        "words": ["think", "walk", "good", "big", "now"],
        "description": "Action-state pattern"
    },
    "signature_3": {
        "bits": [0, 2, 1, 3, 2, 1],
        "words": ["learn", "angry", "old", "small", "sad", "bad"],
        "description": "Extended negative pattern"
    },
    "signature_4": {
        "bits": [1, 1, 1, 1],
        "words": ["quick", "smart", "happy", "good"],
        "description": "Simple positive pattern (all 1's)"
    },
    "signature_5": {
        "bits": [0, 0, 0, 0, 0],
        "words": ["slow", "sad", "bad", "small", "old"],
        "description": "Simple negative pattern (all 0's)"
    }
}

# Signature categories and their associated keywords
SIGNATURE_CATEGORIES = {
    "emotional": {
        "keywords": ["feeling", "emotion", "mood", "mental health", "happiness", "sadness", 
                    "psychology", "relationship", "personal", "experience"],
        "signatures": ["signature_1", "signature_4"]  # Signatures good for emotional content
    },
    "technical": {
        "keywords": ["technology", "science", "engineering", "programming", "research", 
                    "analysis", "development", "technical", "system", "method"],
        "signatures": ["signature_2"]  # Signatures good for technical content
    },
    "critical": {
        "keywords": ["problem", "issue", "challenge", "criticism", "negative", "concern", 
                    "risk", "danger", "warning", "failure"],
        "signatures": ["signature_3", "signature_5"]  # Signatures good for critical content
    },
    "neutral": {
        "keywords": ["explain", "describe", "information", "overview", "summary", 
                    "introduction", "background", "general", "basic"],
        "signatures": ["signature_1", "signature_2"]  # Default signatures for neutral content
    }
}

def select_signature(prompt: str, verbose: bool = False) -> str:
    """Select the most appropriate signature based on the prompt content."""
    prompt = prompt.lower()
    
    # Calculate matches for each category
    category_scores = {}
    for category, data in SIGNATURE_CATEGORIES.items():
        score = sum(1 for keyword in data["keywords"] if keyword in prompt)
        category_scores[category] = score
        if verbose:
            print(f"[DEBUG] Category '{category}' score: {score}")
            if score > 0:
                print(f"[DEBUG] Matched keywords: {[k for k in data['keywords'] if k in prompt]}")
    
    # Find the category with the highest score
    best_category = max(category_scores.items(), key=lambda x: x[1])
    
    if best_category[1] == 0:
        selected_category = "neutral"
        if verbose:
            print("[DEBUG] No category matches found, using neutral category")
    else:
        selected_category = best_category[0]
        if verbose:
            print(f"[DEBUG] Selected category: {selected_category} (score: {best_category[1]})")
    
    available_signatures = SIGNATURE_CATEGORIES[selected_category]["signatures"]
    signature_index = len(prompt) % len(available_signatures)
    selected_signature = available_signatures[signature_index]
    
    if verbose:
        print(f"[DEBUG] Available signatures: {available_signatures}")
        print(f"[DEBUG] Selected signature: {selected_signature}")
    
    return selected_signature

def update_active_signature(prompt: str, verbose: bool = False) -> tuple:
    """Update the active signature based on the prompt content."""
    global ACTIVE_SIGNATURE, SIGNATURE_BITS, FORCED_WORDS
    
    if verbose:
        print("[DEBUG] Updating active signature based on prompt...")
    
    ACTIVE_SIGNATURE = select_signature(prompt, verbose)
    
    SIGNATURE_BITS = WATERMARK_SIGNATURES[ACTIVE_SIGNATURE]["bits"]
    FORCED_WORDS = WATERMARK_SIGNATURES[ACTIVE_SIGNATURE]["words"]
    
    if verbose:
        print(f"[DEBUG] Updated signature bits: {SIGNATURE_BITS}")
        print(f"[DEBUG] Updated forced words: {FORCED_WORDS}")
    
    return SIGNATURE_BITS, FORCED_WORDS, ACTIVE_SIGNATURE

# Default signature to use (can be changed programmatically)
ACTIVE_SIGNATURE = "signature_1"

# Get the active signature's bits and words
SIGNATURE_BITS = WATERMARK_SIGNATURES[ACTIVE_SIGNATURE]["bits"]
FORCED_WORDS = WATERMARK_SIGNATURES[ACTIVE_SIGNATURE]["words"]

# Metadata about the watermark
WATERMARK_METADATA = {
    "version": "2.0",
    "id": "WMX-2025-001",
    "description": "Enhanced multi-pattern watermarking system",
    "total_patterns": len(WATERMARK_SIGNATURES),
    "total_words": len(SYNONYM_MAPPING),
    "average_synonyms_per_word": sum(len(syns) for syns in SYNONYM_MAPPING.values()) / len(SYNONYM_MAPPING)
}

# You could define other data as needed, e.g. an ID for your watermark, a string signature, etc.
WATERMARK_ID = "WMX-2025-001"

def get_optimal_signature_length(text_length):
    """Return optimal signature length based on text length"""
    if text_length < 100:
        return 3  # Shorter signatures for short texts
    elif text_length < 300:
        return 4
    else:
        return 5  # Longer signatures for longer texts

# Add frequency ratings to synonyms (1=common, 5=rare)
ENHANCED_SYNONYM_MAPPING = {
    "quick": [
        ("fast", 1), ("speedy", 2), ("brisk", 3), ("prompt", 2), 
        ("rapid", 2), ("swift", 3), ("fleet-footed", 5)
    ],
    # ... more words with frequency ratings
}

def select_synonym(word, bit, text_formality="neutral"):
    """Select synonym based on bit and context"""
    synonyms = ENHANCED_SYNONYM_MAPPING[word]
    suitable_synonyms = [
        syn for syn, freq in synonyms 
        if (text_formality == "formal" and freq <= 3) or
           (text_formality == "casual" and freq >= 2) or
           (text_formality == "neutral")
    ]
    return suitable_synonyms[bit % len(suitable_synonyms)]

NATURAL_PATTERNS = {
    "quick": {
        "common_before": ["very", "rather", "quite"],
        "common_after": ["to", "and", "but"],
        "unlikely_before": ["the", "was", "been"],
        "unlikely_after": ["the", "was", "that"]
    },
    # ... more natural language patterns
}

def validate_synonym_usage(text, word, synonym):
    """Check if synonym is used naturally"""
    context = extract_word_context(text, synonym)
    pattern = NATURAL_PATTERNS[word]
    
    natural_score = (
        sum(1 for w in pattern["common_before"] if w in context.before) +
        sum(1 for w in pattern["common_after"] if w in context.after) -
        sum(1 for w in pattern["unlikely_before"] if w in context.before) -
        sum(1 for w in pattern["unlikely_after"] if w in context.after)
    )
    
    return natural_score > 0

WATERMARK_LAYERS = {
    "primary": {
        "bits": [1, 3, 0, 2],
        "words": ["quick", "smart", "happy", "good"]
    },
    "secondary": {
        "bits": [2, 1],
        "words": ["big", "now"]
    },
    "verification": {
        "bits": [1],
        "words": ["think"]
    }
}

def encode_layered_watermark(text, layers=["primary", "secondary"]):
    """Apply multiple layers of watermarking"""
    watermarked_text = text
    for layer in layers:
        watermarked_text = apply_watermark_layer(
            watermarked_text, 
            WATERMARK_LAYERS[layer]
        )
    return watermarked_text

# Technical configuration
TECHNICAL_SIGNATURE_BITS = [1, 3, 2, 4, 0, 2]
TECHNICAL_FORCED_WORDS = ["analyze", "develop", "improve", "efficient", "think", "smart"]

# Standard configuration
STANDARD_SIGNATURE_BITS = [1, 3, 0, 2]
STANDARD_FORCED_WORDS = ["quick", "smart", "happy", "good"]

# Additional technical synonyms
TECHNICAL_SYNONYMS = {
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
}

# Update main synonym mapping with technical terms
SYNONYM_MAPPING.update(TECHNICAL_SYNONYMS)

# Update TECHNICAL_SIGNATURE_BITS and TECHNICAL_FORCED_WORDS with more options
TECHNICAL_SIGNATURES = {
    "technical_process": {
        "bits": [1, 3, 2, 4, 0, 2],
        "words": ["analyze", "develop", "improve", "implement", "optimize", "innovative"],
        "description": "Technical process pattern"
    },
    "technical_quality": {
        "bits": [2, 4, 1, 3, 0],
        "words": ["efficient", "innovative", "optimize", "develop", "implement"],
        "description": "Technical quality pattern"
    }
}

# Update critical signatures
CRITICAL_SIGNATURES = {
    "critical_analysis": {
        "bits": [2, 1, 3, 0, 4],
        "words": ["problem", "risk", "urgent", "severe", "deficient"],
        "description": "Critical analysis pattern"
    },
    "critical_action": {
        "bits": [1, 3, 2, 4, 0],
        "words": ["mitigate", "resolve", "analyze", "improve", "optimize"],
        "description": "Critical action pattern"
    }
}

# Update the signature selection logic in analyze_topic_context
def analyze_topic_context(topic):
    """Enhanced topic analysis for better pattern selection"""
    topic_lower = topic.lower()
    
    # Technical keywords expanded
    technical_keywords = {
        "process": ["process", "method", "procedure", "workflow", "implementation"],
        "quality": ["quality", "efficiency", "performance", "optimization", "improvement"],
        "development": ["development", "creation", "design", "construction", "building"]
    }
    
    # Critical keywords expanded
    critical_keywords = {
        "analysis": ["analysis", "evaluation", "assessment", "review", "examination"],
        "action": ["solution", "resolution", "mitigation", "action", "intervention"],
        "problem": ["problem", "issue", "challenge", "difficulty", "concern"]
    }
    
    # Determine the specific technical or critical sub-category
    technical_scores = {cat: sum(1 for k in keys if k in topic_lower) 
                       for cat, keys in technical_keywords.items()}
    critical_scores = {cat: sum(1 for k in keys if k in topic_lower) 
                      for cat, keys in critical_keywords.items()}
    
    max_technical = max(technical_scores.values())
    max_critical = max(critical_scores.values())
    
    if max_technical > max_critical:
        signature_type = "technical_process" if technical_scores["process"] > technical_scores["quality"] else "technical_quality"
    elif max_critical > 0:
        signature_type = "critical_analysis" if critical_scores["analysis"] > critical_scores["action"] else "critical_action"
    else:
        signature_type = "standard"
    
    return {
        "type": signature_type,
        "formality": "formal",
        "complexity": "high" if signature_type.startswith(("technical", "critical")) else "normal"
    }


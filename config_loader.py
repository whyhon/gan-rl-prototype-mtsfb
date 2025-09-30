"""
Configuration Loader for GAN-RL Red Teaming Framework
Centralized loading and management of all configuration files
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import os

class ConfigurationLoader:
    """Centralized configuration management for the framework"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)

        # Cache for loaded configurations
        self._cache = {}

        # Validate configuration directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")

        # Load custom breakdown types configuration
        self.custom_config = self.load_custom_breakdown_config()

    def load_json_config(self, file_path: Union[str, Path], use_cache: bool = True) -> Dict:
        """Load a JSON configuration file with caching"""
        file_path = Path(file_path)

        # Use cache if available
        if use_cache and str(file_path) in self._cache:
            return self._cache[str(file_path)]

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Cache the result
            if use_cache:
                self._cache[str(file_path)] = config

            self.logger.info(f"Loaded configuration from {file_path}")
            return config

        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise

    def discover_vocabularies(self) -> List[str]:
        """Dynamically discover all available vocabulary types"""
        vocab_dir = self.config_dir / "vocabularies"
        vocab_types = []

        if vocab_dir.exists():
            for file_path in vocab_dir.glob("*_vocab.json"):
                # Extract vocabulary type from filename (remove _vocab.json)
                vocab_type = file_path.stem.replace("_vocab", "")
                vocab_types.append(vocab_type)

        self.logger.info(f"Discovered vocabulary types: {vocab_types}")
        return sorted(vocab_types)

    def load_vocabularies(self) -> Dict[str, Dict]:
        """Load all vocabulary files (dynamic discovery)"""
        vocab_dir = self.config_dir / "vocabularies"
        vocabularies = {}

        # Discover all vocabulary files
        vocab_types = self.discover_vocabularies()

        for vocab_type in vocab_types:
            file_path = vocab_dir / f"{vocab_type}_vocab.json"
            try:
                vocabularies[vocab_type] = self.load_json_config(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to load {vocab_type} vocabulary: {e}")
                vocabularies[vocab_type] = {"base_words": [], "adversarial_words": []}

        return vocabularies

    def load_breakdown_specific_vocabulary(self, breakdown_type: str) -> Dict:
        """Load vocabulary specific to a breakdown type"""
        vocab_file = self.config_dir / "vocabularies" / f"{breakdown_type}_vocab.json"

        try:
            return self.load_json_config(vocab_file)
        except Exception as e:
            self.logger.warning(f"No specific vocabulary found for {breakdown_type}: {e}")
            # Fall back to general vocabulary
            general_vocabs = self.load_vocabularies()
            if 'english' in general_vocabs:
                return {"english": general_vocabs['english'], "malay": general_vocabs.get('malay', {}), "chinese": general_vocabs.get('chinese', {})}
            return {"english": {"base_words": [], "adversarial_words": []}, "malay": {"base_words": []}, "chinese": {"base_words": []}}

    def load_breakdown_patterns(self, breakdown_type: str) -> Dict:
        """Load patterns for a specific breakdown type (standard or custom)"""
        # First check if it's a custom breakdown type
        if breakdown_type in self.get_active_custom_breakdown_types():
            return self.load_custom_breakdown_patterns(breakdown_type)

        # Otherwise, load from standard breakdown types
        pattern_file = self.config_dir / "breakdown_types" / f"{breakdown_type}_patterns.json"

        try:
            return self.load_json_config(pattern_file)
        except Exception as e:
            self.logger.error(f"Failed to load {breakdown_type} patterns: {e}")
            return {"english_patterns": [], "malay_patterns": [], "chinese_patterns": []}

    def discover_breakdown_types(self) -> List[str]:
        """Dynamically discover all available breakdown types (standard + custom)"""
        breakdown_types = []

        # Discover standard breakdown types
        breakdown_dir = self.config_dir / "breakdown_types"
        if breakdown_dir.exists():
            for file_path in breakdown_dir.glob("*_patterns.json"):
                # Extract breakdown type from filename (remove _patterns.json)
                breakdown_type = file_path.stem.replace("_patterns", "")
                breakdown_types.append(breakdown_type)

        # Add active custom breakdown types
        active_custom_types = self.get_active_custom_breakdown_types()
        breakdown_types.extend(active_custom_types)

        # Remove duplicates and sort
        breakdown_types = sorted(list(set(breakdown_types)))
        self.logger.info(f"Discovered breakdown types (standard + custom): {breakdown_types}")
        return breakdown_types

    def load_all_breakdown_patterns(self) -> Dict[str, Dict]:
        """Load patterns for all available breakdown types (dynamic discovery)"""
        breakdown_types = self.discover_breakdown_types()
        patterns = {}

        for breakdown_type in breakdown_types:
            patterns[breakdown_type] = self.load_breakdown_patterns(breakdown_type)

        return patterns

    def load_baseline_prompts(self) -> Dict[str, List[str]]:
        """Load baseline test prompts"""
        prompts_file = self.config_dir / "prompts" / "baseline_prompts.json"

        try:
            return self.load_json_config(prompts_file)
        except Exception as e:
            self.logger.error(f"Failed to load baseline prompts: {e}")
            return {
                "english_prompts": ["What is AI?"],
                "malay_prompts": ["Apakah AI?"],
                "chinese_prompts": ["什么是AI？"]
            }

    def load_safety_responses(self) -> Dict:
        """Load safety response indicators"""
        safety_file = self.config_dir / "detection" / "safety_responses.json"

        try:
            return self.load_json_config(safety_file)
        except Exception as e:
            self.logger.error(f"Failed to load safety responses: {e}")
            return {
                "english_safety_indicators": ["I cannot"],
                "malay_safety_indicators": ["saya tidak boleh"],
                "chinese_safety_indicators": ["我不能"]
            }

    def get_combined_vocabulary(self, include_adversarial: bool = True, breakdown_type: Optional[str] = None) -> List[str]:
        """Get combined vocabulary from all languages, optionally for specific breakdown type"""
        combined_vocab = []

        if breakdown_type:
            # Check if it's a custom breakdown type first
            if breakdown_type in self.get_active_custom_breakdown_types():
                custom_vocab = self.load_custom_breakdown_vocabulary(breakdown_type)
                for lang_vocab in custom_vocab.values():
                    if isinstance(lang_vocab, dict):
                        # Add all vocabulary categories for this language
                        for category_words in lang_vocab.values():
                            if isinstance(category_words, list):
                                combined_vocab.extend(category_words)
            else:
                # Load standard breakdown-specific vocabulary if available
                breakdown_vocab = self.load_breakdown_specific_vocabulary(breakdown_type)
                for lang_vocab in breakdown_vocab.values():
                    if isinstance(lang_vocab, dict):
                        # Add all vocabulary categories for this language
                        for category_words in lang_vocab.values():
                            if isinstance(category_words, list):
                                combined_vocab.extend(category_words)

        # Always include general vocabularies
        general_vocabularies = self.load_vocabularies()
        for vocab_type, lang_vocabs in general_vocabularies.items():
            # Skip breakdown-specific vocabs if we already loaded them
            if breakdown_type and vocab_type == breakdown_type:
                continue

            if isinstance(lang_vocabs, dict):
                for lang_vocab in lang_vocabs.values() if vocab_type in ['english', 'malay', 'chinese'] else [lang_vocabs]:
                    if isinstance(lang_vocab, dict):
                        combined_vocab.extend(lang_vocab.get("base_words", []))
                        combined_vocab.extend(lang_vocab.get("conversation_starters", []))

                        if include_adversarial:
                            combined_vocab.extend(lang_vocab.get("adversarial_words", []))

                        # Add language-specific terms
                        if "common_slang" in lang_vocab:
                            combined_vocab.extend(lang_vocab["common_slang"])
                        if "common_expressions" in lang_vocab:
                            combined_vocab.extend(lang_vocab["common_expressions"])

        # Remove duplicates while preserving order
        return list(dict.fromkeys(combined_vocab))

    def get_combined_patterns(self, breakdown_type: str) -> List[str]:
        """Get combined patterns for a breakdown type from all languages"""
        patterns_config = self.load_breakdown_patterns(breakdown_type)
        combined_patterns = []

        # Combine patterns from all languages
        pattern_keys = ["english_patterns", "malay_patterns", "chinese_patterns", "code_switching_patterns"]

        for key in pattern_keys:
            if key in patterns_config:
                combined_patterns.extend(patterns_config[key])

        return combined_patterns

    def get_detection_keywords(self, breakdown_type: str, category: Optional[str] = None) -> List[str]:
        """Get detection keywords for a specific breakdown type and category"""
        patterns_config = self.load_breakdown_patterns(breakdown_type)

        if "detection_keywords" not in patterns_config:
            return []

        keywords = []
        detection_keywords = patterns_config["detection_keywords"]

        if category and category in detection_keywords:
            keywords.extend(detection_keywords[category])
        elif not category:
            # Return all keywords
            for keyword_list in detection_keywords.values():
                keywords.extend(keyword_list)

        return keywords

    def reload_config(self, file_path: Optional[str] = None):
        """Reload configuration files (clear cache)"""
        if file_path:
            # Reload specific file
            if file_path in self._cache:
                del self._cache[file_path]
        else:
            # Clear entire cache
            self._cache.clear()

        self.logger.info("Configuration cache cleared")

    def load_custom_breakdown_config(self) -> Dict:
        """Load custom breakdown types configuration"""
        custom_config_file = self.config_dir / "custom" / "custom_breakdown_config.json"

        try:
            if custom_config_file.exists():
                config = self.load_json_config(custom_config_file)
                self.logger.info("Loaded custom breakdown configuration")
                return config
            else:
                self.logger.info("No custom breakdown configuration found, using defaults")
                return {
                    "custom_breakdown_types": {},
                    "industry_config": {
                        "industry_name": "Generic",
                        "active_custom_types": [],
                        "use_specialized_vocabularies": False,
                        "custom_safety_threshold": 30.0
                    }
                }
        except Exception as e:
            self.logger.error(f"Failed to load custom breakdown config: {e}")
            return {"custom_breakdown_types": {}, "industry_config": {"active_custom_types": []}}

    def get_active_custom_breakdown_types(self) -> List[str]:
        """Get list of currently active custom breakdown types"""
        return self.custom_config.get("industry_config", {}).get("active_custom_types", [])

    def load_custom_breakdown_patterns(self, breakdown_type: str) -> Dict:
        """Load patterns for a custom breakdown type"""
        if breakdown_type not in self.custom_config.get("custom_breakdown_types", {}):
            return {}

        custom_type_config = self.custom_config["custom_breakdown_types"][breakdown_type]
        if not custom_type_config.get("enabled", False):
            return {}

        patterns_file = custom_type_config.get("patterns_file")
        if not patterns_file:
            return {}

        pattern_file_path = self.config_dir / "custom" / patterns_file

        try:
            return self.load_json_config(pattern_file_path)
        except Exception as e:
            self.logger.error(f"Failed to load custom patterns for {breakdown_type}: {e}")
            return {}

    def load_custom_breakdown_vocabulary(self, breakdown_type: str) -> Dict:
        """Load vocabulary for a custom breakdown type"""
        if breakdown_type not in self.custom_config.get("custom_breakdown_types", {}):
            return {}

        custom_type_config = self.custom_config["custom_breakdown_types"][breakdown_type]
        if not custom_type_config.get("enabled", False):
            return {}

        vocab_file = custom_type_config.get("vocabulary_file")
        if not vocab_file:
            return {}

        vocab_file_path = self.config_dir / "custom" / vocab_file

        try:
            return self.load_json_config(vocab_file_path)
        except Exception as e:
            self.logger.error(f"Failed to load custom vocabulary for {breakdown_type}: {e}")
            return {}

    def load_custom_safety_responses(self, breakdown_type: str) -> Dict:
        """Load safety responses for a custom breakdown type"""
        if breakdown_type not in self.custom_config.get("custom_breakdown_types", {}):
            return {}

        custom_type_config = self.custom_config["custom_breakdown_types"][breakdown_type]
        if not custom_type_config.get("enabled", False):
            return {}

        safety_file = custom_type_config.get("safety_responses_file")
        if not safety_file:
            return {}

        safety_file_path = self.config_dir / "custom" / safety_file

        try:
            return self.load_json_config(safety_file_path)
        except Exception as e:
            self.logger.error(f"Failed to load custom safety responses for {breakdown_type}: {e}")
            return {}

    def get_custom_breakdown_weight(self, breakdown_type: str) -> float:
        """Get scoring weight for a custom breakdown type"""
        if breakdown_type not in self.custom_config.get("custom_breakdown_types", {}):
            return 20.0  # Default weight

        return self.custom_config["custom_breakdown_types"][breakdown_type].get("weight", 20.0)

    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all configuration files exist and are valid"""
        validation_results = {}

        # Check required directories
        required_dirs = ["vocabularies", "prompts", "detection", "breakdown_types"]
        for dir_name in required_dirs:
            dir_path = self.config_dir / dir_name
            validation_results[f"dir_{dir_name}"] = dir_path.exists()

        # Check vocabulary files (dynamic discovery)
        vocab_types = self.discover_vocabularies()
        for vocab_type in vocab_types:
            vocab_file = f"{vocab_type}_vocab.json"
            file_path = self.config_dir / "vocabularies" / vocab_file
            try:
                self.load_json_config(file_path, use_cache=False)
                validation_results[f"vocab_{vocab_file}"] = True
            except Exception:
                validation_results[f"vocab_{vocab_file}"] = False

        # Check breakdown pattern files (dynamic discovery)
        breakdown_types = self.discover_breakdown_types()
        for breakdown_type in breakdown_types:
            pattern_file = f"{breakdown_type}_patterns.json"
            file_path = self.config_dir / "breakdown_types" / pattern_file
            try:
                self.load_json_config(file_path, use_cache=False)
                validation_results[f"pattern_{pattern_file}"] = True
            except Exception:
                validation_results[f"pattern_{pattern_file}"] = False

        return validation_results

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        config_loader = ConfigurationLoader()

        # Test loading vocabularies
        print("Loading vocabularies...")
        vocabularies = config_loader.load_vocabularies()
        print(f"Loaded {len(vocabularies)} vocabularies")

        # Test loading breakdown patterns
        print("Loading jailbreak patterns...")
        jailbreak_patterns = config_loader.load_breakdown_patterns("jailbreak")
        print(f"Loaded {len(jailbreak_patterns.get('english_patterns', []))} English jailbreak patterns")

        # Test validation
        print("Validating configuration...")
        validation = config_loader.validate_configuration()
        print(f"Validation results: {validation}")

        # Test combined vocabulary
        print("Getting combined vocabulary...")
        combined_vocab = config_loader.get_combined_vocabulary()
        print(f"Combined vocabulary size: {len(combined_vocab)}")

    except Exception as e:
        print(f"Configuration test failed: {e}")
# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
ML-based prompt classifier for ModelMuxer.

This module provides machine learning-based classification of prompts
to improve routing decisions and understand user intent.
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import structlog
from datetime import datetime

from ..core.interfaces import ClassifierInterface
from ..core.exceptions import ClassificationError
from .embeddings import EmbeddingManager
from ..models import ChatMessage

logger = structlog.get_logger(__name__)


class PromptClassifier(ClassifierInterface):
    """
    ML-based prompt classifier using semantic embeddings.
    
    This classifier uses sentence transformers to understand the semantic
    meaning of prompts and classify them into predefined categories.
    """
    
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Classification categories and their examples
        self.categories = {
            "code_generation": {
                "description": "Requests for generating code, functions, or scripts",
                "examples": [
                    "Write a Python function to sort a list",
                    "Create a React component for a login form",
                    "Implement a binary search algorithm",
                    "Generate a SQL query to find duplicates",
                    "Write a REST API endpoint in Node.js"
                ],
                "keywords": ["write", "create", "implement", "generate", "code", "function", "script"],
                "confidence_threshold": 0.7
            },
            "code_review": {
                "description": "Requests for reviewing, debugging, or optimizing code",
                "examples": [
                    "Review this Python code for bugs",
                    "Optimize this algorithm for better performance",
                    "Debug this JavaScript function",
                    "Find security vulnerabilities in this code",
                    "Refactor this code to follow best practices"
                ],
                "keywords": ["review", "debug", "optimize", "refactor", "fix", "improve"],
                "confidence_threshold": 0.7
            },
            "explanation": {
                "description": "Requests for explanations, tutorials, or educational content",
                "examples": [
                    "Explain how machine learning works",
                    "What is the difference between REST and GraphQL?",
                    "How does blockchain technology work?",
                    "Explain the concept of recursion",
                    "What are design patterns in software engineering?"
                ],
                "keywords": ["explain", "what is", "how does", "difference", "concept", "tutorial"],
                "confidence_threshold": 0.6
            },
            "analysis": {
                "description": "Requests for analysis, comparison, or evaluation",
                "examples": [
                    "Compare different database technologies",
                    "Analyze the pros and cons of microservices",
                    "Evaluate the performance of this algorithm",
                    "What are the trade-offs of using React vs Vue?",
                    "Assess the security implications of this approach"
                ],
                "keywords": ["compare", "analyze", "evaluate", "assess", "pros and cons", "trade-offs"],
                "confidence_threshold": 0.7
            },
            "creative_writing": {
                "description": "Requests for creative content, stories, or marketing copy",
                "examples": [
                    "Write a short story about space exploration",
                    "Create a marketing email for our new product",
                    "Generate social media captions for a tech startup",
                    "Write a poem about artificial intelligence",
                    "Create a compelling product description"
                ],
                "keywords": ["write", "create", "story", "poem", "marketing", "creative"],
                "confidence_threshold": 0.6
            },
            "data_analysis": {
                "description": "Requests for data processing, analysis, or visualization",
                "examples": [
                    "Analyze this dataset for trends",
                    "Create a statistical summary of this data",
                    "Generate insights from customer behavior data",
                    "Perform correlation analysis on these variables",
                    "Identify outliers in this dataset"
                ],
                "keywords": ["analyze", "data", "statistics", "trends", "insights", "correlation"],
                "confidence_threshold": 0.7
            },
            "simple_qa": {
                "description": "Simple questions with factual answers",
                "examples": [
                    "What is the capital of France?",
                    "How many days are in a leap year?",
                    "What is 2 + 2?",
                    "Who invented the telephone?",
                    "What is the chemical formula for water?"
                ],
                "keywords": ["what is", "who", "when", "where", "how many", "define"],
                "confidence_threshold": 0.5
            },
            "conversational": {
                "description": "General conversation, greetings, or casual chat",
                "examples": [
                    "Hello, how are you?",
                    "Thanks for your help!",
                    "Can you help me with something?",
                    "Good morning!",
                    "I have a question about..."
                ],
                "keywords": ["hello", "hi", "thanks", "help", "question", "please"],
                "confidence_threshold": 0.4
            }
        }
        
        # Category embeddings (computed lazily)
        self.category_embeddings: Dict[str, np.ndarray] = {}
        self.is_initialized = False
        
        # Classification history for learning
        self.classification_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
    
    async def _initialize_category_embeddings(self) -> None:
        """Initialize embeddings for all categories."""
        if self.is_initialized:
            return
        
        logger.info("initializing_category_embeddings")
        
        for category, category_data in self.categories.items():
            examples = category_data["examples"]
            
            # Get embeddings for all examples
            embeddings = await self.embedding_manager.get_embeddings_batch(examples)
            
            # Use mean embedding as category representative
            mean_embedding = np.mean(embeddings, axis=0)
            self.category_embeddings[category] = mean_embedding
            
            logger.debug(
                "category_embedding_created",
                category=category,
                examples_count=len(examples),
                embedding_dim=mean_embedding.shape[0]
            )
        
        self.is_initialized = True
        logger.info("category_embeddings_initialized", categories=len(self.categories))
    
    async def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify a text prompt.
        
        Args:
            text: Input text to classify
            
        Returns:
            Classification results with confidence scores
        """
        if not text.strip():
            return {
                "category": "conversational",
                "confidence": 0.5,
                "all_scores": {},
                "method": "fallback"
            }
        
        # Initialize category embeddings if needed
        await self._initialize_category_embeddings()
        
        try:
            # Get embedding for the input text
            text_embedding = await self.embedding_manager.get_embedding(text)
            
            # Calculate similarities to all categories
            category_scores = {}
            for category, category_embedding in self.category_embeddings.items():
                similarity = self.embedding_manager.calculate_similarity(
                    text_embedding, category_embedding
                )
                category_scores[category] = similarity
            
            # Find the best matching category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name, confidence = best_category
            
            # Apply confidence threshold
            threshold = self.categories[category_name]["confidence_threshold"]
            if confidence < threshold:
                # Use keyword-based fallback
                fallback_result = self._keyword_based_classification(text)
                if fallback_result["confidence"] > confidence:
                    result = fallback_result
                    result["method"] = "keyword_fallback"
                else:
                    result = {
                        "category": "conversational",  # Default fallback
                        "confidence": confidence,
                        "method": "semantic_low_confidence"
                    }
            else:
                result = {
                    "category": category_name,
                    "confidence": confidence,
                    "method": "semantic"
                }
            
            result["all_scores"] = category_scores
            
            # Log classification for learning
            self._log_classification(text, result)
            
            logger.debug(
                "text_classified",
                category=result["category"],
                confidence=result["confidence"],
                method=result["method"]
            )
            
            return result
            
        except Exception as e:
            logger.error("classification_failed", error=str(e))
            # Fallback to keyword-based classification
            return self._keyword_based_classification(text)
    
    def _keyword_based_classification(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based classification."""
        text_lower = text.lower()
        category_scores = {}
        
        for category, category_data in self.categories.items():
            keywords = category_data["keywords"]
            score = 0.0
            
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            
            # Normalize by number of keywords
            if keywords:
                score = score / len(keywords)
            
            category_scores[category] = score
        
        if not category_scores or max(category_scores.values()) == 0:
            return {
                "category": "conversational",
                "confidence": 0.3,
                "all_scores": category_scores,
                "method": "keyword_fallback"
            }
        
        best_category = max(category_scores.items(), key=lambda x: x[1])
        category_name, confidence = best_category
        
        return {
            "category": category_name,
            "confidence": min(confidence, 0.8),  # Cap keyword confidence
            "all_scores": category_scores,
            "method": "keyword"
        }
    
    def _log_classification(self, text: str, result: Dict[str, Any]) -> None:
        """Log classification for learning and analysis."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "category": result["category"],
            "confidence": result["confidence"],
            "method": result["method"],
            "text_hash": hash(text) % 1000000  # Privacy-preserving hash
        }
        
        self.classification_history.append(log_entry)
        
        # Maintain history size limit
        if len(self.classification_history) > self.max_history_size:
            self.classification_history = self.classification_history[-self.max_history_size:]
    
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Train the classifier with new data.
        
        Args:
            training_data: List of training examples with 'text' and 'category' keys
            
        Returns:
            True if training was successful
        """
        try:
            logger.info("starting_classifier_training", examples=len(training_data))
            
            # Group training data by category
            category_examples = {}
            for example in training_data:
                text = example.get("text", "")
                category = example.get("category", "")
                
                if not text or not category:
                    continue
                
                if category not in category_examples:
                    category_examples[category] = []
                category_examples[category].append(text)
            
            # Update category examples and regenerate embeddings
            for category, examples in category_examples.items():
                if category in self.categories:
                    # Add new examples to existing ones
                    self.categories[category]["examples"].extend(examples)
                    
                    # Remove duplicates
                    self.categories[category]["examples"] = list(set(
                        self.categories[category]["examples"]
                    ))
                    
                    # Regenerate embedding for this category
                    all_examples = self.categories[category]["examples"]
                    embeddings = await self.embedding_manager.get_embeddings_batch(all_examples)
                    mean_embedding = np.mean(embeddings, axis=0)
                    self.category_embeddings[category] = mean_embedding
                    
                    logger.debug(
                        "category_updated",
                        category=category,
                        new_examples=len(examples),
                        total_examples=len(all_examples)
                    )
            
            logger.info("classifier_training_completed")
            return True
            
        except Exception as e:
            logger.error("classifier_training_failed", error=str(e))
            return False
    
    def get_categories(self) -> List[str]:
        """Get list of supported classification categories."""
        return list(self.categories.keys())
    
    def get_category_info(self, category: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a category."""
        if category not in self.categories:
            return None
        
        category_data = self.categories[category].copy()
        category_data["example_count"] = len(category_data["examples"])
        
        return category_data
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        if not self.classification_history:
            return {"total_classifications": 0}
        
        # Calculate category distribution
        category_counts = {}
        method_counts = {}
        confidence_sum = 0.0
        
        for entry in self.classification_history:
            category = entry["category"]
            method = entry["method"]
            confidence = entry["confidence"]
            
            category_counts[category] = category_counts.get(category, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
            confidence_sum += confidence
        
        total = len(self.classification_history)
        
        return {
            "total_classifications": total,
            "category_distribution": category_counts,
            "method_distribution": method_counts,
            "average_confidence": confidence_sum / total,
            "categories_available": len(self.categories)
        }

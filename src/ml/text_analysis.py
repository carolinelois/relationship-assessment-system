from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from textblob import TextBlob
import spacy
from collections import defaultdict
import torch
from loguru import logger

class TextAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.lda = LatentDirichletAllocation(
            n_components=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        
        # Emotion categories and their associated terms
        self.emotion_lexicon = {
            "joy": ["happy", "excited", "delighted", "content", "satisfied"],
            "anger": ["angry", "frustrated", "annoyed", "irritated", "furious"],
            "fear": ["afraid", "anxious", "worried", "nervous", "scared"],
            "sadness": ["sad", "disappointed", "unhappy", "depressed", "hurt"],
            "trust": ["trust", "confident", "secure", "reliable", "dependable"],
            "love": ["love", "affection", "caring", "intimate", "attached"]
        }

    async def analyze_text_response(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Perform various analyses
            sentiment = await self._analyze_sentiment(text)
            emotions = await self._analyze_emotions(doc)
            themes = await self._extract_themes(doc)
            key_patterns = await self._identify_patterns(doc)
            relationship_indicators = await self._extract_relationship_indicators(
                doc, context
            )
            
            return {
                "sentiment": sentiment,
                "emotions": emotions,
                "themes": themes,
                "patterns": key_patterns,
                "relationship_indicators": relationship_indicators,
                "metadata": {
                    "text_length": len(text),
                    "complexity_score": self._calculate_complexity(doc),
                    "confidence_score": self._calculate_confidence(
                        sentiment, emotions, themes
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def analyze_response_batch(
        self, responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        try:
            # Extract text from responses
            texts = [r.get("response_text", "") for r in responses]
            
            # Perform batch analysis
            topics = await self._analyze_topics(texts)
            sentiment_trends = await self._analyze_sentiment_trends(texts)
            pattern_summary = await self._summarize_patterns(texts)
            
            return {
                "topics": topics,
                "sentiment_trends": sentiment_trends,
                "pattern_summary": pattern_summary,
                "metadata": {
                    "response_count": len(responses),
                    "average_length": np.mean([len(t) for t in texts]),
                    "confidence_score": self._calculate_batch_confidence(
                        topics, sentiment_trends
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing response batch: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        # Get transformer-based sentiment
        transformer_sentiment = self.sentiment_analyzer(text)[0]
        
        # Get TextBlob sentiment
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        # Combine and normalize scores
        sentiment_score = (
            (float(transformer_sentiment["score"]) if transformer_sentiment["label"] == "POSITIVE" else -float(transformer_sentiment["score"])) +
            textblob_sentiment.polarity
        ) / 2
        
        return {
            "score": sentiment_score,
            "magnitude": abs(sentiment_score),
            "label": "positive" if sentiment_score > 0 else "negative",
            "confidence": float(transformer_sentiment["score"]),
            "subjectivity": float(textblob_sentiment.subjectivity)
        }

    async def _analyze_emotions(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        emotion_scores = defaultdict(float)
        total_matches = 0
        
        # Analyze text for each emotion category
        text_lower = doc.text.lower()
        for emotion, terms in self.emotion_lexicon.items():
            matches = sum(1 for term in terms if term in text_lower)
            emotion_scores[emotion] = matches
            total_matches += matches
        
        # Normalize scores
        if total_matches > 0:
            for emotion in emotion_scores:
                emotion_scores[emotion] /= total_matches
        
        # Get primary and secondary emotions
        sorted_emotions = sorted(
            emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "primary_emotion": sorted_emotions[0][0] if sorted_emotions else None,
            "secondary_emotion": sorted_emotions[1][0] if len(sorted_emotions) > 1 else None,
            "emotion_scores": dict(emotion_scores),
            "intensity": sum(emotion_scores.values()) / len(emotion_scores) if emotion_scores else 0
        }

    async def _extract_themes(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        themes = []
        
        # Extract noun phrases as potential themes
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract key verbs and their objects
        verb_phrases = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_obj = []
                for child in token.children:
                    if child.dep_ in ["dobj", "pobj"]:
                        verb_obj.append((token.text, child.text))
        
        # Combine and analyze for themes
        for phrase in noun_phrases:
            if len(phrase.split()) > 1:  # Consider only multi-word phrases
                themes.append({
                    "text": phrase,
                    "type": "concept",
                    "confidence": 0.7
                })
        
        for entity, label in entities:
            themes.append({
                "text": entity,
                "type": label,
                "confidence": 0.8
            })
        
        for verb, obj in verb_phrases:
            themes.append({
                "text": f"{verb} {obj}",
                "type": "action",
                "confidence": 0.6
            })
        
        return themes

    async def _identify_patterns(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        patterns = []
        
        # Analyze sentence structure patterns
        sentence_patterns = defaultdict(int)
        for sent in doc.sents:
            pattern = self._get_sentence_pattern(sent)
            sentence_patterns[pattern] += 1
        
        # Analyze word usage patterns
        word_patterns = self._analyze_word_patterns(doc)
        
        # Analyze emotional patterns
        emotional_patterns = self._analyze_emotional_patterns(doc)
        
        # Combine patterns
        patterns.extend([
            {
                "type": "sentence_structure",
                "pattern": pattern,
                "frequency": count,
                "significance": count / len(list(doc.sents))
            }
            for pattern, count in sentence_patterns.items()
        ])
        
        patterns.extend(word_patterns)
        patterns.extend(emotional_patterns)
        
        return patterns

    async def _extract_relationship_indicators(
        self,
        doc: spacy.tokens.Doc,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        indicators = {
            "communication_style": self._analyze_communication_style(doc),
            "emotional_expression": self._analyze_emotional_expression(doc),
            "relationship_dynamics": self._analyze_relationship_dynamics(doc),
            "attachment_indicators": self._analyze_attachment_indicators(doc)
        }
        
        if context:
            indicators["contextual_analysis"] = self._analyze_context(doc, context)
        
        return indicators

    async def _analyze_topics(self, texts: List[str]) -> List[Dict[str, Any]]:
        # Create document-term matrix
        dtm = self.tfidf.fit_transform(texts)
        
        # Perform topic modeling
        topics = self.lda.fit_transform(dtm)
        
        # Get feature names
        feature_names = self.tfidf.get_feature_names_out()
        
        # Extract top words for each topic
        topics_data = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [
                feature_names[i]
                for i in topic.argsort()[:-10:-1]
            ]
            
            topics_data.append({
                "topic_id": topic_idx,
                "top_words": top_words,
                "coherence_score": self._calculate_topic_coherence(
                    topic, feature_names
                ),
                "prevalence": float(np.mean(topics[:, topic_idx]))
            })
        
        return topics_data

    async def _analyze_sentiment_trends(
        self, texts: List[str]
    ) -> Dict[str, Any]:
        sentiments = []
        for text in texts:
            sentiment = await self._analyze_sentiment(text)
            sentiments.append(sentiment)
        
        return {
            "average_sentiment": np.mean([s["score"] for s in sentiments]),
            "sentiment_variance": np.var([s["score"] for s in sentiments]),
            "trend": self._calculate_sentiment_trend(sentiments),
            "distribution": {
                "positive": sum(1 for s in sentiments if s["score"] > 0) / len(sentiments),
                "negative": sum(1 for s in sentiments if s["score"] < 0) / len(sentiments),
                "neutral": sum(1 for s in sentiments if s["score"] == 0) / len(sentiments)
            }
        }

    async def _summarize_patterns(self, texts: List[str]) -> Dict[str, Any]:
        all_patterns = []
        for text in texts:
            doc = self.nlp(text)
            patterns = await self._identify_patterns(doc)
            all_patterns.extend(patterns)
        
        # Aggregate patterns
        pattern_summary = defaultdict(list)
        for pattern in all_patterns:
            pattern_summary[pattern["type"]].append(pattern)
        
        # Calculate pattern statistics
        return {
            pattern_type: {
                "frequency": len(patterns),
                "significance": np.mean([p["significance"] for p in patterns]),
                "common_patterns": self._get_common_patterns(patterns)
            }
            for pattern_type, patterns in pattern_summary.items()
        }

    def _get_sentence_pattern(self, sent: spacy.tokens.Span) -> str:
        return " ".join([token.pos_ for token in sent])

    def _analyze_word_patterns(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        patterns = []
        
        # Analyze word repetition
        word_freq = defaultdict(int)
        for token in doc:
            if not token.is_stop and not token.is_punct:
                word_freq[token.text.lower()] += 1
        
        # Find significant repetitions
        for word, freq in word_freq.items():
            if freq > 1:
                patterns.append({
                    "type": "word_repetition",
                    "pattern": word,
                    "frequency": freq,
                    "significance": freq / len(doc)
                })
        
        return patterns

    def _analyze_emotional_patterns(
        self, doc: spacy.tokens.Doc
    ) -> List[Dict[str, Any]]:
        patterns = []
        
        # Analyze emotional transitions
        prev_emotion = None
        emotion_transitions = []
        
        for sent in doc.sents:
            emotion = self._get_dominant_emotion(sent.text)
            if prev_emotion and emotion != prev_emotion:
                emotion_transitions.append((prev_emotion, emotion))
            prev_emotion = emotion
        
        if emotion_transitions:
            patterns.append({
                "type": "emotional_transition",
                "pattern": emotion_transitions,
                "frequency": len(emotion_transitions),
                "significance": len(emotion_transitions) / len(list(doc.sents))
            })
        
        return patterns

    def _get_dominant_emotion(self, text: str) -> Optional[str]:
        max_score = 0
        dominant_emotion = None
        
        text_lower = text.lower()
        for emotion, terms in self.emotion_lexicon.items():
            score = sum(1 for term in terms if term in text_lower)
            if score > max_score:
                max_score = score
                dominant_emotion = emotion
        
        return dominant_emotion

    def _analyze_communication_style(
        self, doc: spacy.tokens.Doc
    ) -> Dict[str, Any]:
        # Analyze various aspects of communication
        assertiveness = self._measure_assertiveness(doc)
        directness = self._measure_directness(doc)
        formality = self._measure_formality(doc)
        
        # Determine primary style
        styles = {
            "assertive": assertiveness,
            "passive": 1 - assertiveness,
            "direct": directness,
            "indirect": 1 - directness,
            "formal": formality,
            "informal": 1 - formality
        }
        
        primary_style = max(styles.items(), key=lambda x: x[1])
        
        return {
            "primary_style": primary_style[0],
            "style_scores": styles,
            "confidence": primary_style[1]
        }

    def _measure_assertiveness(self, doc: spacy.tokens.Doc) -> float:
        assertive_indicators = [
            "I think", "I believe", "I need", "I want",
            "I will", "I can", "I do"
        ]
        passive_indicators = [
            "maybe", "perhaps", "sort of", "kind of",
            "I guess", "I suppose", "if possible"
        ]
        
        text_lower = doc.text.lower()
        assertive_count = sum(1 for ind in assertive_indicators if ind.lower() in text_lower)
        passive_count = sum(1 for ind in passive_indicators if ind.lower() in text_lower)
        
        total = assertive_count + passive_count
        if total == 0:
            return 0.5
        
        return assertive_count / total

    def _measure_directness(self, doc: spacy.tokens.Doc) -> float:
        # Count direct statements vs. hedging language
        direct_count = 0
        indirect_count = 0
        
        for sent in doc.sents:
            if any(hedge in sent.text.lower() for hedge in ["maybe", "perhaps", "might", "could"]):
                indirect_count += 1
            else:
                direct_count += 1
        
        total = direct_count + indirect_count
        if total == 0:
            return 0.5
        
        return direct_count / total

    def _measure_formality(self, doc: spacy.tokens.Doc) -> float:
        formal_indicators = len([
            token for token in doc
            if token.pos_ in ["AUX", "SCONJ"] or token.is_punct
        ])
        
        informal_indicators = len([
            token for token in doc
            if token.pos_ == "INTJ" or token.text.lower() in ["gonna", "wanna", "kinda"]
        ])
        
        total = formal_indicators + informal_indicators
        if total == 0:
            return 0.5
        
        return formal_indicators / total

    def _calculate_complexity(self, doc: spacy.tokens.Doc) -> float:
        # Calculate various complexity metrics
        avg_sentence_length = np.mean([len(sent) for sent in doc.sents])
        avg_word_length = np.mean([len(token.text) for token in doc if not token.is_punct])
        unique_words = len(set([token.text.lower() for token in doc if not token.is_punct]))
        
        # Combine metrics
        complexity = (
            (avg_sentence_length / 20) +  # Normalize by typical sentence length
            (avg_word_length / 5) +      # Normalize by typical word length
            (unique_words / 100)         # Normalize by typical vocabulary size
        ) / 3
        
        return min(max(complexity, 0), 1)  # Ensure score is between 0 and 1

    def _calculate_confidence(
        self,
        sentiment: Dict[str, Any],
        emotions: Dict[str, Any],
        themes: List[Dict[str, Any]]
    ) -> float:
        # Combine confidence scores from different analyses
        confidence_scores = [
            sentiment.get("confidence", 0),
            emotions.get("intensity", 0),
            np.mean([t.get("confidence", 0) for t in themes]) if themes else 0
        ]
        
        return np.mean(confidence_scores)

    def _calculate_batch_confidence(
        self,
        topics: List[Dict[str, Any]],
        sentiment_trends: Dict[str, Any]
    ) -> float:
        # Calculate confidence based on topic coherence and sentiment consistency
        topic_coherence = np.mean([
            t.get("coherence_score", 0) for t in topics
        ])
        sentiment_consistency = 1 - sentiment_trends.get("sentiment_variance", 0)
        
        return (topic_coherence + sentiment_consistency) / 2

    def _calculate_topic_coherence(
        self, topic: np.ndarray, feature_names: np.ndarray
    ) -> float:
        # Calculate simple topic coherence score
        top_term_indices = topic.argsort()[:-10:-1]
        top_terms = feature_names[top_term_indices]
        
        # Calculate average pairwise similarity
        coherence = 0
        count = 0
        
        for i in range(len(top_terms)):
            for j in range(i + 1, len(top_terms)):
                similarity = self._calculate_term_similarity(
                    top_terms[i], top_terms[j]
                )
                coherence += similarity
                count += 1
        
        return coherence / count if count > 0 else 0

    def _calculate_term_similarity(self, term1: str, term2: str) -> float:
        # Simple character-based similarity
        shorter = min(len(term1), len(term2))
        distance = sum(a != b for a, b in zip(term1[:shorter], term2[:shorter]))
        similarity = 1 - (distance / shorter)
        return similarity

    def _calculate_sentiment_trend(
        self, sentiments: List[Dict[str, Any]]
    ) -> str:
        if len(sentiments) < 2:
            return "stable"
        
        scores = [s["score"] for s in sentiments]
        slope = np.polyfit(range(len(scores)), scores, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"

    def _get_common_patterns(
        self, patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Group patterns by their exact match
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            key = str(pattern["pattern"])  # Convert pattern to string for hashing
            pattern_groups[key].append(pattern)
        
        # Get most common patterns
        common_patterns = []
        for key, group in pattern_groups.items():
            common_patterns.append({
                "pattern": group[0]["pattern"],
                "frequency": len(group),
                "significance": np.mean([p["significance"] for p in group])
            })
        
        # Sort by frequency and return top 5
        return sorted(
            common_patterns,
            key=lambda x: (x["frequency"], x["significance"]),
            reverse=True
        )[:5]
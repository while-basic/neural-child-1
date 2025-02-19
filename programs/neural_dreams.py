"""
neural_dreams.py
Created: 2024-03-21
Description: A system for simulating and visualizing the dreams, emotions, and memory consolidation
of a neural child during sleep states.

Author: Dr. Celaya
Project: Neural Child + Meta-Learning
"""

import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import requests
import json
import time
from dataclasses import dataclass
from enum import Enum
import random
from child_model import DynamicNeuralChild
import gradio as gr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DreamState(Enum):
    """Different states of the dreaming process"""
    LIGHT_SLEEP = "Light Sleep"
    DEEP_SLEEP = "Deep Sleep"
    REM = "REM Sleep"
    LUCID = "Lucid Dreaming"

@dataclass
class Memory:
    """Structure for storing individual memories"""
    timestamp: datetime
    content: str
    emotional_state: torch.Tensor
    importance: float
    tags: List[str]
    associations: List[str]
    dream_frequency: int = 0

@dataclass
class Dream:
    """Structure for storing dream sequences"""
    timestamp: datetime
    state: DreamState
    narrative: str
    emotions: torch.Tensor
    memories_referenced: List[Memory]
    symbols: List[str]
    intensity: float

@dataclass
class DreamInsight:
    """Structure for storing discovered insights from dreams"""
    timestamp: datetime
    pattern_type: str  # "emotional", "symbolic", "temporal", "causal"
    description: str
    confidence: float
    supporting_evidence: List[str]
    potential_implications: List[str]
    novelty_score: float

class OllamaAPI:
    """Interface for Ollama API interactions"""
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate_dream(self, 
                      prompt: str, 
                      model: str = "artifish/llama3.2-uncensored",
                      temperature: float = 0.9) -> str:
        """Generate dream content using Ollama"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": 150,
                    "stream": False
                },
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            if "response" in result:
                return result["response"].strip()
            else:
                logger.error(f"Unexpected response format: {result}")
                return ""
                
        except Exception as e:
            logger.error(f"Dream generation error: {str(e)}")
            return ""
            
    def __del__(self):
        """Clean up resources"""
        self.session.close()

class DreamAnalyzer:
    """Advanced system for analyzing dreams and discovering patterns"""
    def __init__(self, ollama_api: OllamaAPI):
        self.ollama = ollama_api
        self.insights: List[DreamInsight] = []
        self.pattern_memory = {}
        self.discovery_threshold = 0.7
        
    def analyze_dream_sequence(self, 
                             recent_dreams: List[Dream], 
                             memories: List[Memory]) -> Optional[DreamInsight]:
        """Analyze dream sequence for patterns and generate insights"""
        if len(recent_dreams) < 2:
            return None
            
        # Prepare analysis prompt
        analysis_prompt = self._create_analysis_prompt(recent_dreams, memories)
        
        # Generate insight using Ollama
        insight_text = self.ollama.generate_dream(
            prompt=analysis_prompt,
            temperature=0.8
        )
        
        if not insight_text:
            return None
            
        # Extract structured insight
        try:
            # Parse the insight text
            lines = insight_text.split("\n")
            pattern_type = ""
            description = ""
            implications = []
            evidence = []
            
            for line in lines:
                if line.startswith("PATTERN:"):
                    pattern_type = line.replace("PATTERN:", "").strip()
                elif line.startswith("INSIGHT:"):
                    description = line.replace("INSIGHT:", "").strip()
                elif line.startswith("EVIDENCE:"):
                    evidence.append(line.replace("EVIDENCE:", "").strip())
                elif line.startswith("IMPLICATION:"):
                    implications.append(line.replace("IMPLICATION:", "").strip())
            
            # Calculate confidence and novelty
            confidence = self._calculate_confidence(evidence)
            novelty = self._calculate_novelty(description)
            
            if confidence >= self.discovery_threshold:
                insight = DreamInsight(
                    timestamp=datetime.now(),
                    pattern_type=pattern_type,
                    description=description,
                    confidence=confidence,
                    supporting_evidence=evidence,
                    potential_implications=implications,
                    novelty_score=novelty
                )
                self.insights.append(insight)
                return insight
                
        except Exception as e:
            logger.error(f"Error processing insight: {str(e)}")
            return None
            
    def _create_analysis_prompt(self, dreams: List[Dream], memories: List[Memory]) -> str:
        """Create prompt for dream sequence analysis"""
        prompt = """Analyze this dream sequence for meaningful patterns and generate a scientific insight.
        Look for:
        1. Emotional patterns and their relationship to memories
        2. Recurring symbols and their evolving meanings
        3. Temporal patterns in dream state transitions
        4. Causal relationships between experiences and dreams
        5. Novel emergent behaviors or unexpected connections
        
        Recent Dreams:
        """
        
        # Add dream information
        for dream in dreams[-3:]:  # Last 3 dreams
            prompt += f"\nDream State: {dream.state.value}"
            prompt += f"\nNarrative: {dream.narrative}"
            prompt += f"\nEmotions: {[f'{e:.2f}' for e in dream.emotions.cpu().tolist()]}"
            prompt += f"\nSymbols: {', '.join(dream.symbols)}\n"
            
        # Add memory context
        prompt += "\nRelevant Memories:"
        for memory in memories[-5:]:  # Last 5 memories
            prompt += f"\n- {memory.content} (importance: {memory.importance:.2f})"
            
        prompt += """
        
        Generate a structured insight with the following format:
        PATTERN: [emotional/symbolic/temporal/causal]
        INSIGHT: [clear, specific description of the discovered pattern]
        EVIDENCE: [specific evidence from dreams/memories supporting this insight]
        EVIDENCE: [additional evidence points if available]
        IMPLICATION: [potential implications for child development]
        IMPLICATION: [additional implications if available]
        """
        
        return prompt
        
    def _calculate_confidence(self, evidence: List[str]) -> float:
        """Calculate confidence score for an insight"""
        # Base confidence on number and quality of evidence
        base_score = min(len(evidence) * 0.2, 0.6)
        
        # Add score for evidence quality
        quality_score = 0.0
        for e in evidence:
            if len(e) > 50:  # Detailed evidence
                quality_score += 0.1
            if any(word in e.lower() for word in ["because", "therefore", "suggests", "indicates"]):
                quality_score += 0.1
                
        return min(base_score + quality_score, 1.0)
        
    def _calculate_novelty(self, description: str) -> float:
        """Calculate novelty score for an insight"""
        # Convert description to feature vector (simple bag of words)
        words = set(description.lower().split())
        
        # Compare with previous insights
        if not self.pattern_memory:
            self.pattern_memory = {word: 1 for word in words}
            return 1.0
            
        # Calculate novelty based on rare words
        total_score = 0.0
        for word in words:
            freq = self.pattern_memory.get(word, 0)
            total_score += 1.0 / (freq + 1)
            self.pattern_memory[word] = freq + 1
            
        return min(total_score / len(words), 1.0)

class DreamProcessor:
    def __init__(self, child: DynamicNeuralChild):
        """Initialize the dream processor"""
        self.child = child
        self.ollama = OllamaAPI()
        self.analyzer = DreamAnalyzer(self.ollama)
        self.memories: List[Memory] = []
        self.dreams: List[Dream] = []
        self.current_dream_state = DreamState.LIGHT_SLEEP
        self.sleep_cycle_duration = 90  # minutes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.latest_insight: Optional[DreamInsight] = None
        
    def add_memory(self, content: str, emotional_state: torch.Tensor, tags: List[str]):
        """Add a new memory to the system"""
        importance = self._calculate_importance(emotional_state, content)
        memory = Memory(
            timestamp=datetime.now(),
            content=content,
            emotional_state=emotional_state,
            importance=importance,
            tags=tags,
            associations=self._find_associations(content, tags)
        )
        self.memories.append(memory)
        logger.info(f"Added new memory: {content[:50]}... (importance: {importance:.2f})")
        
    def generate_dream(self) -> Dream:
        """Generate a new dream and analyze for patterns"""
        dream = self._generate_base_dream()
        self.dreams.append(dream)
        
        # Analyze dream sequence for insights
        if len(self.dreams) >= 2:
            insight = self.analyzer.analyze_dream_sequence(self.dreams[-3:], self.memories)
            if insight and insight.novelty_score > 0.6:
                self.latest_insight = insight
                logger.info(f"New dream insight discovered: {insight.description}")
                
        return dream
        
    def _calculate_importance(self, emotional_state: torch.Tensor, content: str) -> float:
        """Calculate memory importance based on emotional intensity and content"""
        emotional_intensity = torch.norm(emotional_state).item()
        content_weight = len(content.split()) / 100  # Length factor
        return min(1.0, emotional_intensity * 0.7 + content_weight * 0.3)
        
    def _find_associations(self, content: str, tags: List[str]) -> List[str]:
        """Find associated concepts for a memory"""
        # Combine existing memory associations
        all_associations = []
        for memory in self.memories:
            all_associations.extend(memory.associations)
            
        # Add new associations based on content and tags
        new_associations = tags.copy()
        words = content.lower().split()
        important_words = [w for w in words if len(w) > 4]  # Simple importance filter
        new_associations.extend(important_words[:5])  # Limit to top 5 words
        
        return list(set(new_associations))  # Remove duplicates
        
    def _select_memories_for_dream(self) -> List[Memory]:
        """Select memories to incorporate into dream based on current state"""
        if not self.memories:
            return []
            
        # Weight memories by recency, importance, and emotional relevance
        weighted_memories = []
        current_time = datetime.now()
        
        for memory in self.memories:
            time_weight = 1.0 / (1.0 + (current_time - memory.timestamp).total_seconds() / 86400)
            emotional_relevance = torch.nn.functional.cosine_similarity(
                memory.emotional_state.unsqueeze(0),
                self.child.emotional_state.unsqueeze(0)
            ).item()
            
            total_weight = (
                time_weight * 0.3 +
                memory.importance * 0.4 +
                emotional_relevance * 0.3
            )
            
            weighted_memories.append((memory, total_weight))
            
        # Sort by weight and select top memories
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        num_memories = 3 if self.current_dream_state == DreamState.REM else 1
        
        return [mem for mem, _ in weighted_memories[:num_memories]]
        
    def _create_dream_prompt(self, selected_memories: List[Memory]) -> str:
        """Create a prompt for dream generation"""
        child_age = self.child.age
        emotional_state = self.child.emotional_state.cpu().tolist()
        
        base_prompt = f"""You are the dreaming mind of a {child_age}-year-old child.
        Your current emotional state is:
        - Joy: {emotional_state[0]:.2f}
        - Trust: {emotional_state[1]:.2f}
        - Fear: {emotional_state[2]:.2f}
        - Surprise: {emotional_state[3]:.2f}
        
        You are in {self.current_dream_state.value}.
        
        Recent memories:
        """
        
        # Add memory context
        for memory in selected_memories:
            base_prompt += f"\n- {memory.content}"
            
        if self.current_dream_state == DreamState.REM:
            base_prompt += "\n\nCreate a vivid, surreal dream narrative that weaves together these memories and emotions."
        elif self.current_dream_state == DreamState.LUCID:
            base_prompt += "\n\nCreate a dream where the child is aware they are dreaming and can influence the dream."
        else:
            base_prompt += "\n\nCreate a simple, brief dream fragment incorporating these elements."
            
        base_prompt += "\nKeep the dream age-appropriate and reflect the emotional state."
        
        return base_prompt
        
    def _calculate_dream_emotions(self, narrative: str, memories: List[Memory]) -> torch.Tensor:
        """Calculate emotional state of the dream"""
        # Start with current emotional state
        base_emotions = self.child.emotional_state.clone()
        
        # Influence from memories
        if memories:
            memory_emotions = torch.stack([m.emotional_state for m in memories])
            memory_influence = torch.mean(memory_emotions, dim=0)
            base_emotions = 0.7 * base_emotions + 0.3 * memory_influence
            
        # Add random fluctuation based on dream state
        if self.current_dream_state == DreamState.REM:
            fluctuation = torch.rand_like(base_emotions) * 0.3
            base_emotions += fluctuation
            
        # Normalize
        base_emotions = torch.clamp(base_emotions, 0, 1)
        
        return base_emotions
        
    def _extract_dream_symbols(self, narrative: str) -> List[str]:
        """Extract symbolic elements from dream narrative"""
        # Common dream symbols to look for
        common_symbols = [
            "flying", "falling", "running", "water", "door", "house", "monster",
            "friend", "family", "school", "toy", "animal", "light", "dark",
            "color", "space", "magic", "transform"
        ]
        
        found_symbols = []
        narrative_lower = narrative.lower()
        
        for symbol in common_symbols:
            if symbol in narrative_lower:
                found_symbols.append(symbol)
                
        return found_symbols
        
    def _update_memory_associations(self, dream: Dream):
        """Update memory associations based on dream content"""
        # Extract key terms from dream narrative
        dream_words = set(dream.narrative.lower().split())
        
        # Update referenced memories
        for memory in dream.memories_referenced:
            # Increment dream frequency
            memory.dream_frequency += 1
            
            # Add new associations from dream
            new_associations = [word for word in dream_words 
                              if len(word) > 4 and word not in memory.associations]
            memory.associations.extend(new_associations[:3])  # Limit new associations
            
    def advance_sleep_cycle(self):
        """Advance to the next sleep cycle state"""
        if self.current_dream_state == DreamState.LIGHT_SLEEP:
            self.current_dream_state = DreamState.DEEP_SLEEP
        elif self.current_dream_state == DreamState.DEEP_SLEEP:
            self.current_dream_state = DreamState.REM
        elif self.current_dream_state == DreamState.REM:
            if random.random() < 0.1:  # 10% chance of lucid dream
                self.current_dream_state = DreamState.LUCID
            else:
                self.current_dream_state = DreamState.LIGHT_SLEEP
        else:  # LUCID
            self.current_dream_state = DreamState.LIGHT_SLEEP
            
    def _generate_base_dream(self) -> Dream:
        """Generate a base dream based on current state and memories"""
        # Select relevant memories based on current emotional state
        selected_memories = self._select_memories_for_dream()
        
        # Create dream prompt based on state
        prompt = self._create_dream_prompt(selected_memories)
        
        # Generate dream narrative
        narrative = self.ollama.generate_dream(prompt)
        
        # Calculate dream emotions
        dream_emotions = self._calculate_dream_emotions(narrative, selected_memories)
        
        # Extract dream symbols
        symbols = self._extract_dream_symbols(narrative)
        
        # Create dream object
        dream = Dream(
            timestamp=datetime.now(),
            state=self.current_dream_state,
            narrative=narrative,
            emotions=dream_emotions,
            memories_referenced=selected_memories,
            symbols=symbols,
            intensity=torch.norm(dream_emotions).item()
        )
        
        # Update memory associations based on dream
        self._update_memory_associations(dream)
        
        return dream

class DreamInterface:
    def __init__(self, child: DynamicNeuralChild):
        """Initialize the dream visualization interface"""
        self.processor = DreamProcessor(child)
        self.is_sleeping = False
        
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface for dream visualization"""
        with gr.Blocks(title="Neural Child Dream Laboratory") as interface:
            gr.Markdown(
                """
                # 🌙 Neural Child Dream Laboratory
                Observe and analyze the dreams, memories, and emotional processing of the neural child.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Dream visualization
                    dream_display = gr.Textbox(
                        label="Current Dream",
                        value="Child is awake...",
                        lines=5,
                        max_lines=5
                    )
                    
                    # Sleep state
                    sleep_state = gr.Label(
                        label="Sleep State",
                        value={"Awake": 1.0}
                    )
                    
                    # Latest insight
                    insight_display = gr.Markdown(
                        label="Latest Discovery",
                        value="No insights discovered yet..."
                    )
                    
                with gr.Column(scale=2):
                    # Emotional state plot
                    emotion_plot = gr.Plot(
                        label="Dream Emotions"
                    )
                    
                    # Insight details
                    with gr.Accordion("Insight Details", open=False):
                        pattern_type = gr.Textbox(label="Pattern Type", interactive=False)
                        confidence = gr.Slider(label="Confidence", minimum=0, maximum=1, value=0)
                        novelty = gr.Slider(label="Novelty", minimum=0, maximum=1, value=0)
                        evidence = gr.Textbox(label="Evidence", lines=3, interactive=False)
                        implications = gr.Textbox(label="Implications", lines=2, interactive=False)
            
            with gr.Row():
                # Memory input
                memory_input = gr.Textbox(
                    label="Add Memory",
                    placeholder="Enter a memory to store..."
                )
                memory_tags = gr.Textbox(
                    label="Memory Tags",
                    placeholder="Enter tags separated by commas..."
                )
                add_memory_btn = gr.Button("Store Memory", variant="primary")
                
            with gr.Row():
                start_sleep_btn = gr.Button("Start Sleep 😴", variant="primary")
                wake_btn = gr.Button("Wake Up 👀", variant="secondary")
                
            with gr.Row():
                # Memory and dream history
                memory_display = gr.DataFrame(
                    headers=["Time", "Memory", "Importance", "Dream Frequency"],
                    label="Memory Bank"
                )
                
            def add_memory(text: str, tags: str):
                """Add a new memory to the system"""
                if not text.strip():
                    return {
                        memory_input: "Please enter a memory",
                        memory_tags: tags
                    }
                    
                # Create emotional state for memory
                emotional_state = torch.rand(4, device=self.processor.device)
                tag_list = [t.strip() for t in tags.split(",") if t.strip()]
                
                self.processor.add_memory(text, emotional_state, tag_list)
                
                # Update memory display
                memories_data = self._get_memories_data()
                
                return {
                    memory_input: "",
                    memory_tags: "",
                    memory_display: memories_data
                }
                
            def start_sleep():
                """Start the sleep cycle"""
                self.is_sleeping = True
                self.processor.current_dream_state = DreamState.LIGHT_SLEEP
                
                # Generate initial dream
                dream = self.processor.generate_dream()
                
                # Get insight if available
                insight_data = self._get_insight_data()
                
                return (
                    dream.narrative,
                    {dream.state.value: 1.0},
                    self._create_emotion_plot(dream.emotions),
                    *insight_data
                )
                
            def wake_up():
                """End the sleep cycle"""
                self.is_sleeping = False
                
                return (
                    "Child is awake...",
                    {"Awake": 1.0},
                    self._create_empty_emotion_plot(),
                    "No insights during waking state...",
                    "", 0.0, 0.0, "", ""
                )
                
            def auto_update():
                """Update the dream state and generate new dream"""
                if self.is_sleeping:
                    self.processor.advance_sleep_cycle()
                    dream = self.processor.generate_dream()
                    
                    # Get insight if available
                    insight_data = self._get_insight_data()
                    
                    return (
                        dream.narrative,
                        {dream.state.value: 1.0},
                        self._create_emotion_plot(dream.emotions),
                        *insight_data
                    )
                return (
                    dream_display.value,
                    sleep_state.value,
                    emotion_plot.value,
                    insight_display.value,
                    pattern_type.value,
                    confidence.value,
                    novelty.value,
                    evidence.value,
                    implications.value
                )
            
            # Connect event handlers
            add_memory_btn.click(
                add_memory,
                inputs=[memory_input, memory_tags],
                outputs=[memory_input, memory_tags, memory_display]
            )
            
            start_sleep_btn.click(
                start_sleep,
                outputs=[
                    dream_display, sleep_state, emotion_plot,
                    insight_display, pattern_type, confidence,
                    novelty, evidence, implications
                ]
            )
            
            wake_btn.click(
                wake_up,
                outputs=[
                    dream_display, sleep_state, emotion_plot,
                    insight_display, pattern_type, confidence,
                    novelty, evidence, implications
                ]
            )
            
            # Set up automatic dream updates using a hidden button
            auto_update_btn = gr.Button("Update", visible=False, elem_id="auto-update-btn")
            auto_update_btn.click(
                auto_update,
                outputs=[
                    dream_display, sleep_state, emotion_plot,
                    insight_display, pattern_type, confidence,
                    novelty, evidence, implications
                ]
            )
            
            # Add JavaScript for automatic updates
            gr.HTML(value="""
                <script>
                    function setupDreamUpdates() {
                        const updateBtn = document.getElementById('auto-update-btn');
                        if (!updateBtn) return;
                        
                        if (window.dreamInterval) {
                            clearInterval(window.dreamInterval);
                        }
                        
                        window.dreamInterval = setInterval(() => {
                            updateBtn.click();
                        }, 30000);
                    }
                    
                    // Initial setup
                    setupDreamUpdates();
                    
                    // Setup on dynamic updates
                    const observer = new MutationObserver((mutations) => {
                        for (const mutation of mutations) {
                            if (mutation.addedNodes.length) {
                                setupDreamUpdates();
                            }
                        }
                    });
                    
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                </script>
            """)
            
        return interface
        
    def _get_memories_data(self) -> List[List]:
        """Get formatted memory data for display"""
        return [
            [
                m.timestamp.strftime("%H:%M:%S"),
                m.content[:50] + "..." if len(m.content) > 50 else m.content,
                f"{m.importance:.2f}",
                str(m.dream_frequency)
            ]
            for m in self.processor.memories
        ]
        
    def _create_emotion_plot(self, emotions: torch.Tensor) -> go.Figure:
        """Create emotion plot for current dream state"""
        emotions = emotions.cpu().numpy()
        
        fig = go.Figure()
        
        emotions_labels = ['Joy', 'Trust', 'Fear', 'Surprise']
        
        fig.add_trace(go.Bar(
            x=emotions_labels,
            y=emotions,
            marker_color=['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        ))
        
        fig.update_layout(
            title='Dream Emotional State',
            template='plotly_dark',
            showlegend=False,
            height=300
        )
        
        return fig
        
    def _create_empty_emotion_plot(self) -> go.Figure:
        """Create empty emotion plot for awake state"""
        fig = go.Figure()
        
        fig.update_layout(
            title='Dream Emotional State',
            template='plotly_dark',
            showlegend=False,
            height=300
        )
        
        return fig
        
    def _get_insight_data(self) -> Tuple:
        """Get formatted insight data for display"""
        insight = self.processor.latest_insight
        if insight:
            markdown = f"""### 🔬 New Pattern Discovered!
            
            **{insight.description}**
            
            *Confidence: {insight.confidence:.0%} | Novelty: {insight.novelty_score:.0%}*
            """
            
            return (
                markdown,
                insight.pattern_type,
                insight.confidence,
                insight.novelty_score,
                "\n".join(insight.supporting_evidence),
                "\n".join(insight.potential_implications)
            )
        
        return (
            "Analyzing dream patterns...",
            "", 0.0, 0.0, "", ""
        )

def launch_dream_lab():
    """Launch the dream laboratory interface"""
    child = DynamicNeuralChild()
    interface = DreamInterface(child)
    demo = interface.create_interface()
    
    # Launch with basic configuration
    demo.queue().launch(
        server_name="localhost",
        server_port=7862,  # Different port from playground
        share=False,
        debug=True
    )

if __name__ == "__main__":
    launch_dream_lab() 
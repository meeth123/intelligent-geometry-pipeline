#!/usr/bin/env python3
"""
Pipeline Visualizer - Real-time agent status tracking and visual handoff display
"""

import time
import streamlit as st
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class AgentState(Enum):
    """Agent processing states"""
    WAITING = "⏳ Waiting"
    THINKING = "🧠 Thinking"
    PROCESSING = "⚡ Processing"
    COMPLETE = "✅ Complete"
    ERROR = "❌ Error"
    HANDOFF = "🤝 Handing off"

@dataclass
class AgentStatus:
    """Agent status information"""
    name: str
    icon: str
    model: str
    state: AgentState
    thinking_message: str
    progress: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    reasoning: str = ""

class PipelineVisualizer:
    """Visual pipeline status tracker for real-time agent monitoring"""
    
    def __init__(self):
        self.agents = {
            "prompt_interpreter": AgentStatus(
                name="Prompt Interpreter",
                icon="🧠",
                model="Gemini 2.5 Flash",
                state=AgentState.WAITING,
                thinking_message="Ready to analyze your prompt...",
                progress=0.0
            ),
            "image_preprocessor": AgentStatus(
                name="Image Preprocessor", 
                icon="🖼️",
                model="Gemini 2.5 Flash",
                state=AgentState.WAITING,
                thinking_message="Ready to enhance images...",
                progress=0.0
            ),
            "vision_interpreter": AgentStatus(
                name="Vision Interpreter",
                icon="👁️",
                model="Gemini 2.5 Flash", 
                state=AgentState.WAITING,
                thinking_message="Ready to analyze visual geometry...",
                progress=0.0
            ),
            "symbolic_geometry_planner": AgentStatus(
                name="Symbolic Geometry Planner",
                icon="🔢",
                model="Gemini 2.5 Pro",
                state=AgentState.WAITING,
                thinking_message="Ready to solve mathematical constraints...",
                progress=0.0
            ),
            "layout_designer": AgentStatus(
                name="Layout Designer",
                icon="🎨",
                model="Gemini 2.5 Flash",
                state=AgentState.WAITING,
                thinking_message="Ready to create beautiful layouts...",
                progress=0.0
            ),
            "renderer": AgentStatus(
                name="Renderer",
                icon="🖨️",
                model="Gemini 2.5 Flash",
                state=AgentState.WAITING,
                thinking_message="Ready to generate final outputs...",
                progress=0.0
            ),
            "math_consistency_verifier": AgentStatus(
                name="Math Consistency Verifier",
                icon="✅",
                model="Gemini 2.5 Flash",
                state=AgentState.WAITING,
                thinking_message="Ready to verify mathematical accuracy...",
                progress=0.0
            ),
            "orchestrator": AgentStatus(
                name="Orchestrator",
                icon="🎯",
                model="Gemini 2.5 Pro",
                state=AgentState.WAITING,
                thinking_message="Ready to coordinate the pipeline...",
                progress=0.0
            )
        }
        
        # Pipeline sequence
        self.pipeline_sequence = [
            "orchestrator",
            "prompt_interpreter", 
            "image_preprocessor",
            "vision_interpreter",
            "symbolic_geometry_planner",
            "layout_designer", 
            "renderer",
            "math_consistency_verifier"
        ]
    
    def update_agent_status(self, agent_name: str, state: AgentState, 
                          thinking_message: str = "", progress: float = None, reasoning: str = ""):
        """Update an agent's status"""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            agent.state = state
            if thinking_message:
                agent.thinking_message = thinking_message
            if progress is not None:
                agent.progress = progress
            if reasoning:
                agent.reasoning = reasoning
            
            # Update timestamps
            if state == AgentState.THINKING and agent.start_time is None:
                agent.start_time = datetime.now()
            elif state == AgentState.COMPLETE and agent.end_time is None:
                agent.end_time = datetime.now()
    
    def get_pipeline_progress(self) -> float:
        """Get overall pipeline progress"""
        completed = sum(1 for agent in self.agents.values() if agent.state == AgentState.COMPLETE)
        return completed / len(self.agents)
    
    def get_current_active_agent(self) -> Optional[str]:
        """Get the currently active agent"""
        for agent_name in self.pipeline_sequence:
            agent = self.agents[agent_name]
            if agent.state in [AgentState.THINKING, AgentState.PROCESSING]:
                return agent_name
        return None
    
    def display_pipeline_visual(self):
        """Display the visual pipeline in Streamlit"""
        st.markdown("## 🚀 Real-Time Pipeline Visualization")
        
        # Overall progress
        overall_progress = self.get_pipeline_progress()
        st.progress(overall_progress, text=f"Pipeline Progress: {overall_progress:.1%}")
        
        # Current active agent
        active_agent = self.get_current_active_agent()
        if active_agent:
            agent = self.agents[active_agent]
            st.info(f"🔥 **Active**: {agent.icon} {agent.name} - {agent.thinking_message}")
        
        # Agent grid display
        cols = st.columns(4)
        
        for i, agent_name in enumerate(self.pipeline_sequence):
            agent = self.agents[agent_name]
            col_idx = i % 4
            
            with cols[col_idx]:
                # Agent card
                status_color = self._get_status_color(agent.state)
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {status_color};
                    border-radius: 10px;
                    padding: 15px;
                    margin: 5px;
                    background-color: {status_color}20;
                    text-align: center;
                ">
                    <h3>{agent.icon} {agent.name}</h3>
                    <p><strong>{agent.state.value}</strong></p>
                    <p style="font-size: 0.8em; color: #666;">{agent.model}</p>
                    <p style="font-style: italic; font-size: 0.9em;">{agent.thinking_message}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar for individual agent
                if agent.progress > 0:
                    st.progress(agent.progress, text=f"{agent.progress:.0%}")
                
                # Timing info
                if agent.start_time and agent.end_time:
                    duration = (agent.end_time - agent.start_time).total_seconds()
                    st.caption(f"⏱️ {duration:.1f}s")
    
    def _get_status_color(self, state: AgentState) -> str:
        """Get color for agent state"""
        color_map = {
            AgentState.WAITING: "#808080",
            AgentState.THINKING: "#FFA500", 
            AgentState.PROCESSING: "#1E90FF",
            AgentState.COMPLETE: "#32CD32",
            AgentState.ERROR: "#DC143C",
            AgentState.HANDOFF: "#9370DB"
        }
        return color_map.get(state, "#808080")
    
    def display_agent_handoff_sequence(self):
        """Display agent handoff sequence"""
        st.markdown("### 🤝 Agent Handoff Sequence")
        
        for i, agent_name in enumerate(self.pipeline_sequence):
            agent = self.agents[agent_name]
            
            # Handoff arrow
            if i > 0:
                st.markdown("⬇️", unsafe_allow_html=True)
            
            # Agent status
            if agent.state == AgentState.COMPLETE:
                st.success(f"{agent.icon} {agent.name} ✅")
            elif agent.state in [AgentState.THINKING, AgentState.PROCESSING]:
                st.info(f"{agent.icon} {agent.name} 🔄 {agent.thinking_message}")
            elif agent.state == AgentState.ERROR:
                st.error(f"{agent.icon} {agent.name} ❌")
            else:
                st.write(f"{agent.icon} {agent.name} ⏳")
    
    def display_server_logs(self):
        """Display server logs with agent thoughts"""
        st.markdown("### 📝 Real-Time Server Logs")
        
        log_container = st.container()
        
        with log_container:
            for agent_name in self.pipeline_sequence:
                agent = self.agents[agent_name]
                
                if agent.state != AgentState.WAITING:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    if agent.state == AgentState.THINKING:
                        st.code(f"[{timestamp}] 🧠 {agent.name}: {agent.thinking_message}")
                    elif agent.state == AgentState.PROCESSING:
                        st.code(f"[{timestamp}] ⚡ {agent.name}: Processing with {agent.model}...")
                    elif agent.state == AgentState.COMPLETE:
                        st.code(f"[{timestamp}] ✅ {agent.name}: Complete! Handing off to next agent...")
                    elif agent.state == AgentState.ERROR:
                        st.code(f"[{timestamp}] ❌ {agent.name}: Error encountered")
                    
                    # Show reasoning if available
                    if agent.reasoning and len(agent.reasoning) > 0:
                        with st.expander(f"🧠 {agent.name} Detailed Reasoning"):
                            st.write(agent.reasoning[:500] + "..." if len(agent.reasoning) > 500 else agent.reasoning)

# Global visualizer instance
_visualizer = PipelineVisualizer()

def get_visualizer() -> PipelineVisualizer:
    """Get the global pipeline visualizer"""
    return _visualizer 
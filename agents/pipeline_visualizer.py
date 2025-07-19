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
            old_state = agent.state
            agent.state = state
            if thinking_message:
                agent.thinking_message = thinking_message
            if progress is not None:
                agent.progress = progress
            if reasoning:
                agent.reasoning = reasoning
            
            # Update timestamps with better tracking
            current_time = datetime.now()
            
            if state == AgentState.THINKING and old_state == AgentState.WAITING:
                agent.start_time = current_time
                agent.end_time = None  # Reset end time
            elif state == AgentState.PROCESSING and agent.start_time is None:
                agent.start_time = current_time
                agent.end_time = None
            elif state in [AgentState.COMPLETE, AgentState.ERROR] and agent.end_time is None:
                agent.end_time = current_time
    
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
    
    def get_agent_elapsed_time(self, agent_name: str) -> float:
        """Get elapsed time for an agent in seconds"""
        if agent_name not in self.agents:
            return 0.0
        
        agent = self.agents[agent_name]
        if agent.start_time is None:
            return 0.0
        
        end_time = agent.end_time or datetime.now()
        return (end_time - agent.start_time).total_seconds()
    
    def get_total_pipeline_time(self) -> float:
        """Get total pipeline elapsed time in seconds"""
        start_times = [agent.start_time for agent in self.agents.values() if agent.start_time]
        end_times = [agent.end_time for agent in self.agents.values() if agent.end_time]
        
        if not start_times:
            return 0.0
        
        earliest_start = min(start_times)
        latest_end = max(end_times) if end_times else datetime.now()
        
        return (latest_end - earliest_start).total_seconds()
    
    def get_timing_summary(self) -> dict:
        """Get comprehensive timing summary"""
        summary = {
            'total_time': self.get_total_pipeline_time(),
            'agents': {},
            'completed_agents': 0,
            'active_agent': self.get_current_active_agent()
        }
        
        for agent_name, agent in self.agents.items():
            elapsed_time = self.get_agent_elapsed_time(agent_name)
            summary['agents'][agent_name] = {
                'name': agent.name,
                'icon': agent.icon,
                'state': agent.state,
                'elapsed_time': elapsed_time,
                'is_complete': agent.state == AgentState.COMPLETE,
                'model': agent.model
            }
            
            if agent.state == AgentState.COMPLETE:
                summary['completed_agents'] += 1
        
        return summary
    
    def reset_pipeline(self):
        """Reset all agents to waiting state and clear timing"""
        for agent in self.agents.values():
            agent.state = AgentState.WAITING
            agent.thinking_message = agent.thinking_message.split("Ready to")[0] + "Ready to " + agent.thinking_message.split("Ready to")[1] if "Ready to" in agent.thinking_message else "Ready..."
            agent.progress = 0.0
            agent.start_time = None
            agent.end_time = None
            agent.reasoning = ""
    
    def display_pipeline_visual(self):
        """Display the visual pipeline in Streamlit"""
        st.markdown("## 🚀 Real-Time Pipeline Visualization")
        
        # Overall progress
        overall_progress = self.get_pipeline_progress()
        st.progress(overall_progress, text=f"Pipeline Progress: {overall_progress:.1%}")
        
        # Current active agent with real-time timing
        active_agent = self.get_current_active_agent()
        if active_agent:
            agent = self.agents[active_agent]
            elapsed_time = self.get_agent_elapsed_time(active_agent)
            st.info(f"🔥 **Active**: {agent.icon} {agent.name} - {agent.thinking_message} ⏱️ {elapsed_time:.1f}s")
        
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
                
                # Enhanced timing info
                elapsed_time = self.get_agent_elapsed_time(agent_name)
                if agent.state == AgentState.COMPLETE and elapsed_time > 0:
                    st.success(f"⏱️ Completed in {elapsed_time:.1f}s")
                elif agent.state in [AgentState.THINKING, AgentState.PROCESSING] and elapsed_time > 0:
                    st.info(f"⏱️ Running: {elapsed_time:.1f}s")
                elif agent.state == AgentState.ERROR and elapsed_time > 0:
                    st.error(f"⏱️ Failed after {elapsed_time:.1f}s")
                elif agent.state == AgentState.WAITING:
                    st.caption("⏳ Ready to start")
    
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
                    elapsed_time = self.get_agent_elapsed_time(agent_name)
                    
                    if agent.state == AgentState.THINKING:
                        st.code(f"[{timestamp}] 🧠 {agent.name}: {agent.thinking_message} | ⏱️ {elapsed_time:.1f}s")
                    elif agent.state == AgentState.PROCESSING:
                        st.code(f"[{timestamp}] ⚡ {agent.name}: Processing with {agent.model}... | ⏱️ {elapsed_time:.1f}s")
                    elif agent.state == AgentState.COMPLETE:
                        st.code(f"[{timestamp}] ✅ {agent.name}: Complete! Handing off to next agent... | ⏱️ {elapsed_time:.1f}s")
                    elif agent.state == AgentState.ERROR:
                        st.code(f"[{timestamp}] ❌ {agent.name}: Error encountered | ⏱️ {elapsed_time:.1f}s")
                    
                    # Show reasoning if available
                    if agent.reasoning and len(agent.reasoning) > 0:
                        with st.expander(f"🧠 {agent.name} Detailed Reasoning"):
                            st.write(agent.reasoning[:500] + "..." if len(agent.reasoning) > 500 else agent.reasoning)
    
    def display_timing_summary(self):
        """Display comprehensive timing summary"""
        st.markdown("### ⏱️ Performance Analytics")
        
        timing_summary = self.get_timing_summary()
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_time = timing_summary['total_time']
            st.metric(
                label="🕐 Total Pipeline Time",
                value=f"{total_time:.1f}s" if total_time > 0 else "Not started",
                delta=None
            )
        
        with col2:
            completed = timing_summary['completed_agents']
            total_agents = len(self.agents)
            st.metric(
                label="✅ Completed Agents", 
                value=f"{completed}/{total_agents}",
                delta=f"{(completed/total_agents*100):.0f}%" if total_agents > 0 else "0%"
            )
        
        with col3:
            active_agent = timing_summary['active_agent']
            if active_agent:
                active_time = timing_summary['agents'][active_agent]['elapsed_time']
                agent_name = timing_summary['agents'][active_agent]['name']
                st.metric(
                    label="🔥 Active Agent Time",
                    value=f"{active_time:.1f}s",
                    delta=f"{agent_name}"
                )
            else:
                st.metric(
                    label="🔥 Active Agent Time",
                    value="None active",
                    delta=None
                )
        
        # Individual agent performance table
        st.markdown("#### 📊 Individual Agent Performance")
        
        agent_data = []
        for agent_name in self.pipeline_sequence:
            agent_info = timing_summary['agents'][agent_name]
            
            # Status emoji
            if agent_info['state'] == AgentState.COMPLETE:
                status_emoji = "✅"
            elif agent_info['state'] in [AgentState.THINKING, AgentState.PROCESSING]:
                status_emoji = "🔄"
            elif agent_info['state'] == AgentState.ERROR:
                status_emoji = "❌"
            else:
                status_emoji = "⏳"
            
            agent_data.append({
                "Agent": f"{agent_info['icon']} {agent_info['name']}",
                "Status": f"{status_emoji} {agent_info['state'].value}",
                "Model": agent_info['model'],
                "Time": f"{agent_info['elapsed_time']:.1f}s" if agent_info['elapsed_time'] > 0 else "-",
                "Performance": self._get_performance_rating(agent_info['elapsed_time'], agent_info['model'])
            })
        
        if agent_data:
            st.table(agent_data)
        
        # Performance insights
        if timing_summary['completed_agents'] > 0:
            self._display_performance_insights(timing_summary)
    
    def _get_performance_rating(self, elapsed_time: float, model: str) -> str:
        """Get performance rating based on elapsed time and model"""
        if elapsed_time <= 0:
            return "⏳ Pending"
        elif elapsed_time < 3:
            return "🚀 Fast"
        elif elapsed_time < 8:
            return "⚡ Good"
        elif elapsed_time < 15:
            return "🐌 Slow"
        else:
            return "🕐 Very Slow"
    
    def _display_performance_insights(self, timing_summary: dict):
        """Display performance insights and recommendations"""
        st.markdown("#### 💡 Performance Insights")
        
        insights = []
        
        # Total time analysis
        total_time = timing_summary['total_time']
        if total_time > 30:
            insights.append("🐌 Pipeline taking longer than expected. Consider optimizing prompts or using faster models.")
        elif total_time < 10:
            insights.append("🚀 Excellent pipeline performance! Your agents are working efficiently.")
        
        # Model performance comparison
        pro_agents = []
        flash_agents = []
        
        for agent_name, agent_info in timing_summary['agents'].items():
            if agent_info['is_complete']:
                if "Pro" in agent_info['model']:
                    pro_agents.append(agent_info['elapsed_time'])
                elif "Flash" in agent_info['model']:
                    flash_agents.append(agent_info['elapsed_time'])
        
        if pro_agents and flash_agents:
            avg_pro = sum(pro_agents) / len(pro_agents)
            avg_flash = sum(flash_agents) / len(flash_agents)
            
            if avg_pro > avg_flash * 2:
                insights.append(f"⚡ Flash agents ({avg_flash:.1f}s avg) are significantly faster than Pro agents ({avg_pro:.1f}s avg)")
            elif avg_pro < avg_flash:
                insights.append(f"🤔 Pro agents ({avg_pro:.1f}s avg) are faster than Flash agents ({avg_flash:.1f}s avg) - unusual!")
        
        # Slowest agent
        completed_agents = [(name, info) for name, info in timing_summary['agents'].items() if info['is_complete']]
        if completed_agents:
            slowest_agent = max(completed_agents, key=lambda x: x[1]['elapsed_time'])
            if slowest_agent[1]['elapsed_time'] > 10:
                insights.append(f"🕐 {slowest_agent[1]['icon']} {slowest_agent[1]['name']} is the bottleneck ({slowest_agent[1]['elapsed_time']:.1f}s)")
        
        # Display insights
        for insight in insights:
            st.info(insight)
        
        if not insights:
            st.success("🎯 All agents performing optimally!")

# Global visualizer instance
_visualizer = PipelineVisualizer()

def get_visualizer() -> PipelineVisualizer:
    """Get the global pipeline visualizer"""
    return _visualizer 
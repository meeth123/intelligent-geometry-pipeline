"""
Data structures for the geometry pipeline agents.
Updated to match the Shared Object Glossary specification.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import re
import uuid
from datetime import datetime


class Status(Enum):
    """Status values for various objects in the pipeline."""
    DRAFT = "draft"
    VISION = "vision" 
    SOLVED = "solved"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    IMPOSSIBLE = "impossible"
    PASS = "pass"
    FAIL = "fail"
    FAILED = "failed"


@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float
    label: Optional[str] = None


@dataclass
class GeometryObject:
    """A geometric object with type, properties, and constraints."""
    type: str  # "point", "line", "circle", "triangle", "rectangle", etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class GeometryConstraint:
    """A constraint between geometric objects."""
    type: str  # "parallel", "perpendicular", "equal", "distance", etc.
    objects: List[str]  # IDs of objects involved
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle:
    """Bundle containing user prompt and any additional context."""
    prompt_id: str
    text: str
    images: List[str] = field(default_factory=list)  # List of image URIs
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.prompt_id:
            self.prompt_id = str(uuid.uuid4())


@dataclass 
class GeometrySpec:
    """Specification of geometric objects and their relationships."""
    objects: List[GeometryObject] = field(default_factory=list)
    constraints: List[GeometryConstraint] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    status: Status = Status.DRAFT
    confidence: float = 1.0
    
    # O3 Thinking Process
    agent_reasoning: Dict[str, str] = field(default_factory=dict)  # Store thinking from each agent
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)  # Step-by-step processing log
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = {}


@dataclass
class CoordinateSolution:
    """Solution with specific coordinates for all geometric objects."""
    # Legacy support
    points: Dict[str, Point] = field(default_factory=dict)  # object_id -> Point
    proof: str = ""  # Mathematical proof or reasoning
    status: Status = Status.SOLVED
    notes: Optional[str] = None
    
    # Enhanced coordinate solution fields
    object_coordinates: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # object_id -> coordinate data
    constraint_solutions: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # constraint_id -> solution data
    coordinate_system: Dict[str, Any] = field(default_factory=dict)  # coordinate system metadata
    accuracy_metrics: Dict[str, Any] = field(default_factory=dict)  # solution quality metrics
    mathematical_derivation: str = ""  # Step-by-step mathematical derivation
    
    # Agent reasoning and processing
    agent_reasoning: Dict[str, str] = field(default_factory=dict)  # Store thinking from each agent
    processing_steps: List[Dict[str, Any]] = field(default_factory=list)  # Step-by-step processing log


@dataclass
class LayoutPlan:
    """Layout plan with SVG and styling information."""
    svg: str  # SVG string with geometry-only content
    labels: List[Dict[str, Any]] = field(default_factory=list)
    style_tokens: Dict[str, Any] = field(default_factory=dict)
    
    # Agent reasoning and processing info
    agent_reasoning: Dict[str, str] = field(default_factory=dict)  # Agent reasoning


@dataclass
class RenderSet:
    """Set of rendered files."""
    render_svg: str  # SVG content
    render_png: Optional[str] = None  # PNG file path or base64
    render_svg_uri: Optional[str] = None  # URI to SVG file
    render_png_uri: Optional[str] = None  # URI to PNG file
    
    # Enhanced rendering metadata
    rendering_decisions: Dict[str, Any] = field(default_factory=dict)  # Rendering optimization decisions
    metadata: Dict[str, Any] = field(default_factory=dict)  # File metadata and quality info
    agent_reasoning: Dict[str, str] = field(default_factory=dict)  # Agent reasoning


@dataclass
class QAReport:
    """Quality assurance report."""
    status: Status
    tolerance_mm: float
    issues: List[str] = field(default_factory=list)
    
    # Enhanced verification data
    verification_data: Dict[str, Any] = field(default_factory=dict)  # Detailed verification results
    agent_reasoning: Dict[str, str] = field(default_factory=dict)  # Agent reasoning
    error_mask_uri: Optional[str] = None


@dataclass
class FinalAssets:
    """Final output assets."""
    final_svg: str
    final_png: Optional[str] = None
    final_svg_uri: Optional[str] = None
    final_png_uri: Optional[str] = None
    
    # Pipeline metadata
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)  # Complete pipeline results and reasoning


# Error types for agent failures
@dataclass
class AgentError:
    """Standard error response from agents."""
    error: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class NumericExtractor:
    """Utility class for extracting numeric values from text."""
    
    # Patterns for different types of measurements
    LENGTH_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(cm|mm|m|in|ft|px|units?)', re.IGNORECASE)
    ANGLE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(degrees?|°|deg)', re.IGNORECASE)
    NUMBER_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\b')
    
    @classmethod
    def extract_lengths(cls, text: str) -> Dict[str, float]:
        """Extract length measurements from text."""
        matches = cls.LENGTH_PATTERN.findall(text)
        lengths = {}
        for i, (value, unit) in enumerate(matches):
            key = f"length_{i+1}" if len(matches) > 1 else "length"
            lengths[f"{key}_{unit}"] = float(value)
        return lengths
    
    @classmethod
    def extract_angles(cls, text: str) -> Dict[str, float]:
        """Extract angle measurements from text."""
        matches = cls.ANGLE_PATTERN.findall(text)
        angles = {}
        for i, (value, unit) in enumerate(matches):
            key = f"angle_{i+1}" if len(matches) > 1 else "angle"
            angles[key] = float(value)
        return angles
    
    @classmethod
    def extract_numbers(cls, text: str) -> List[float]:
        """Extract all numbers from text."""
        matches = cls.NUMBER_PATTERN.findall(text)
        return [float(match) for match in matches]


# Utility functions for creating objects
def create_geometry_object(obj_type: str, **properties) -> GeometryObject:
    """Create a geometry object with the given type and properties."""
    return GeometryObject(type=obj_type, properties=properties)


def create_constraint(constraint_type: str, object_ids: List[str], **parameters) -> GeometryConstraint:
    """Create a constraint between objects."""
    return GeometryConstraint(type=constraint_type, objects=object_ids, parameters=parameters) 
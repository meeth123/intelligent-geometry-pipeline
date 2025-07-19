#!/usr/bin/env python3
"""
Math Consistency Verifier Agent - Verifies mathematical accuracy and constraint satisfaction
Uses Gemini 2.5 Pro for intelligent mathematical verification and quality assurance
"""

import os
import logging
import json
import re
from typing import Dict, Any, Optional, List
import google.generativeai as genai

from .data_structures import CoordinateSolution, QAReport, AgentError, Status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathConsistencyVerifier:
    """AI-powered math verifier using Gemini 2.5 Pro for intelligent accuracy verification."""
    
    def __init__(self):
        """Initialize the Math Consistency Verifier with Gemini 2.5 Pro."""
        self.model = None
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Configure Gemini 2.5 Pro for mathematical verification."""
        try:
            # Try to get API key from environment, with fallback
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                # Fallback API key setting
                api_key = "AIzaSyAHCbEYuASKZkr2adn2-CanH0aF7vusnus"
                os.environ['GOOGLE_API_KEY'] = api_key
                logger.warning("Setting Google API key directly in environment")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini math verifier configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            self.model = None

    def handle(self, coordinate_solution: CoordinateSolution, tolerance_mm: float = 0.1) -> QAReport:
        """
        Verify mathematical consistency and constraint satisfaction.
        
        Args:
            coordinate_solution: Solution to verify
            tolerance_mm: Acceptable tolerance in millimeters
            
        Returns:
            QAReport with verification results and issues
        """
        try:
            logger.info("Math Consistency Verifier agent starting...")
            
            if not self.model:
                            return AgentError(
                error="MODEL_UNAVAILABLE",
                message="Gemini model not available",
                details={"agent": "math_consistency_verifier", "error": "Model initialization failed"}
            )
            
            # Verify mathematical consistency using Gemini 2.5 Pro
            qa_report = self._verify_with_ai(coordinate_solution, tolerance_mm)
            
            if isinstance(qa_report, AgentError):
                return qa_report
            
            logger.info(f"Math Consistency Verifier completed - Status: {qa_report.status}")
            return qa_report
            
        except Exception as e:
            logger.error(f"Error in math consistency verifier: {e}")
            return AgentError(
                error="VERIFICATION_FAILED",
                message=f"Mathematical verification failed: {str(e)}",
                details={"agent": "math_consistency_verifier", "error": str(e)}
            )

    def _verify_with_ai(self, coordinate_solution: CoordinateSolution, tolerance_mm: float) -> QAReport:
        """Use Gemini 2.5 Pro to verify mathematical consistency."""
        
        # Prepare verification data for AI analysis
        verification_prompt = f"""
You are an expert mathematical verifier specializing in geometric constraint solving. Verify the mathematical accuracy and consistency of this coordinate solution.

COORDINATE SOLUTION TO VERIFY:
Object Coordinates: {json.dumps(coordinate_solution.object_coordinates, indent=2)}
Constraint Solutions: {json.dumps(coordinate_solution.constraint_solutions, indent=2)}
Coordinate System: {json.dumps(coordinate_solution.coordinate_system, indent=2)}
Mathematical Derivation: {coordinate_solution.mathematical_derivation[:500]}...
Accuracy Metrics: {json.dumps(coordinate_solution.accuracy_metrics, indent=2)}

VERIFICATION REQUIREMENTS:
1. Check mathematical accuracy of all coordinate calculations
2. Verify that all constraints are properly satisfied within tolerance
3. Validate geometric relationships and dependencies
4. Check for mathematical consistency and logical coherence
5. Identify any numerical errors or approximations
6. Verify coordinate system consistency
7. Check scaling and unit consistency

TOLERANCE: {tolerance_mm}mm

RESPONSE FORMAT:
```json
{{
    "reasoning": "Your detailed mathematical verification process...",
    "verification_status": "VERIFIED" | "ISSUES_FOUND" | "FAILED",
    "constraint_verification": {{
        "total_constraints": number,
        "satisfied_constraints": number,
        "failed_constraints": [list of constraint IDs that failed],
        "tolerance_violations": [list of violations with details]
    }},
    "mathematical_accuracy": {{
        "calculation_errors": [list of mathematical errors found],
        "precision_issues": [list of precision concerns],
        "consistency_check": "PASS" | "FAIL",
        "numerical_stability": "STABLE" | "UNSTABLE"
    }},
    "coordinate_validation": {{
        "coordinate_system_valid": true/false,
        "scaling_consistent": true/false,
        "unit_consistency": true/false,
        "geometric_relationships": "VALID" | "INVALID"
    }},
    "issues_found": [
        {{
            "issue_type": "constraint_violation | calculation_error | inconsistency",
            "severity": "critical | warning | minor",
            "description": "detailed description of the issue",
            "affected_objects": ["list of object IDs"],
            "suggested_fix": "recommendation for fixing the issue"
        }}
    ],
    "overall_quality_score": number_between_0_and_1,
    "recommendations": [
        "list of recommendations for improvement"
    ]
}}
```

Perform a thorough mathematical verification and provide detailed analysis.
"""
        
        try:
            logger.info(f"Verifying {len(coordinate_solution.object_coordinates)} objects with {len(coordinate_solution.constraint_solutions)} constraints")
            
            response = self.model.generate_content(verification_prompt)
            response_text = response.text
            
            logger.info(f"Gemini verification response: {len(response_text)} characters")
            
            # Extract reasoning for frontend display
            reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', response_text, re.DOTALL)
            reasoning = reasoning_match.group(1) if reasoning_match else "Mathematical verification completed with AI analysis."
            
            # Store reasoning for frontend
            if not hasattr(coordinate_solution, 'agent_reasoning'):
                coordinate_solution.agent_reasoning = {}
            coordinate_solution.agent_reasoning['math_consistency_verifier'] = reasoning
            
            logger.info(f"Gemini reasoning extracted: {reasoning[:100]}...")
            
            # Parse JSON response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                verification_data = json.loads(json_match.group(1))
            else:
                # Fallback parsing
                verification_data = json.loads(response_text)
            
            # Extract verification results
            verification_status = verification_data.get('verification_status', 'ISSUES_FOUND')
            issues_found = verification_data.get('issues_found', [])
            overall_quality_score = verification_data.get('overall_quality_score', 0.5)
            
            # Determine status
            if verification_status == "VERIFIED" and overall_quality_score >= 0.8:
                status = Status.VERIFIED
            elif verification_status == "FAILED" or overall_quality_score < 0.3:
                status = Status.FAILED
            else:
                status = Status.DRAFT  # Needs improvement
            
            # Extract issue descriptions
            issue_descriptions = []
            for issue in issues_found:
                severity = issue.get('severity', 'unknown')
                description = issue.get('description', 'Unknown issue')
                issue_descriptions.append(f"[{severity.upper()}] {description}")
            
            # Create QA report
            qa_report = QAReport(
                status=status,
                tolerance_mm=tolerance_mm,
                issues=issue_descriptions
            )
            
            # Add detailed verification data
            qa_report.verification_data = verification_data
            qa_report.agent_reasoning = getattr(coordinate_solution, 'agent_reasoning', {})
            
            return qa_report
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return self._create_fallback_qa_report(coordinate_solution, tolerance_mm)
            
        except Exception as e:
            logger.error(f"Gemini verification failed: {e}")
            return self._create_fallback_qa_report(coordinate_solution, tolerance_mm)

    def _create_fallback_qa_report(self, coordinate_solution: CoordinateSolution, tolerance_mm: float) -> QAReport:
        """Create a basic fallback QA report when AI fails."""
        logger.warning("Creating fallback QA report")
        
        # Basic verification
        issues = []
        status = Status.DRAFT
        
        # Check if we have coordinate data
        if not coordinate_solution.object_coordinates:
            issues.append("[CRITICAL] No object coordinates found")
            status = Status.FAILED
        
        # Check if we have constraint solutions
        if not coordinate_solution.constraint_solutions:
            issues.append("[WARNING] No constraint solutions found")
        
        # Basic consistency check
        satisfied_constraints = 0
        total_constraints = len(coordinate_solution.constraint_solutions)
        
        for constraint_id, solution in coordinate_solution.constraint_solutions.items():
            if isinstance(solution, dict) and solution.get('satisfied', False):
                satisfied_constraints += 1
        
        if total_constraints > 0:
            satisfaction_rate = satisfied_constraints / total_constraints
            if satisfaction_rate < 0.5:
                issues.append(f"[CRITICAL] Only {satisfaction_rate:.1%} of constraints satisfied")
                status = Status.FAILED
            elif satisfaction_rate < 0.8:
                issues.append(f"[WARNING] {satisfaction_rate:.1%} of constraints satisfied")
            else:
                status = Status.VERIFIED if satisfaction_rate >= 0.95 else Status.DRAFT
        
        qa_report = QAReport(
            status=status,
            tolerance_mm=tolerance_mm,
            issues=issues
        )
        
        qa_report.verification_data = {
            "verification_status": "FALLBACK",
            "constraint_satisfaction_rate": satisfied_constraints / max(total_constraints, 1),
            "fallback_reason": "AI verification failed, using basic checks"
        }
        
        return qa_report


# Global instance
math_consistency_verifier = MathConsistencyVerifier()

def handle(coordinate_solution: CoordinateSolution, tolerance_mm: float = 0.1) -> QAReport:
    """Handle math consistency verification request."""
    return math_consistency_verifier.handle(coordinate_solution, tolerance_mm) 
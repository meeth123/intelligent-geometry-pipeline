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
from datetime import datetime

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
        
        # Create debug directory for JSON logs
        self.debug_dir = "debug_json_responses"
        os.makedirs(self.debug_dir, exist_ok=True)
    
    def _save_debug_json(self, response_text: str, prompt_id: str, status: str):
        """Save raw JSON response for debugging purposes"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"math_verifier_{prompt_id[:8]}_{timestamp}_{status}.json"
            filepath = os.path.join(self.debug_dir, filename)
            
            debug_data = {
                "agent": "math_consistency_verifier",
                "prompt_id": prompt_id,
                "timestamp": timestamp,
                "status": status,
                "raw_response": response_text,
                "response_length": len(response_text)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved debug JSON to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save debug JSON: {e}")
    
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
            self.model = genai.GenerativeModel('gemini-2.5-flash')
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
            
            # Save raw response for debugging
            coord_id = getattr(coordinate_solution, 'solution_id', 'verification')
            self._save_debug_json(response_text, coord_id, "raw_response")
            
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
                json_str = json_match.group(1)
                verification_data = json.loads(json_str)
                # Save successfully parsed JSON
                self._save_debug_json(json_str, coord_id, "parsed_success")
            else:
                # Fallback parsing
                verification_data = json.loads(response_text)
                # Save fallback parsed JSON
                self._save_debug_json(response_text, coord_id, "fallback_parsed")
            
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
            
            # Save failed JSON for debugging
            self._save_debug_json(response_text, coord_id, "parse_failed")
            
            # Try to clean the JSON and parse again
            try:
                cleaned_response = self._clean_json_response(response_text)
                verification_data = json.loads(cleaned_response)
                logger.info("Successfully parsed cleaned JSON response")
                
                # Save cleaned successful JSON
                self._save_debug_json(cleaned_response, coord_id, "cleaned_success")
                
                # Continue with cleaned verification data
                qa_report = QAReport(
                    status=Status.VERIFIED if verification_data.get('all_constraints_satisfied', False) else Status.FAILED,
                    tolerance_mm=tolerance_mm,
                    issues=verification_data.get('issues_found', [])
                )
                
                qa_report.verification_data = verification_data
                qa_report.agent_reasoning = getattr(coordinate_solution, 'agent_reasoning', {})
                return qa_report
                
            except Exception as cleanup_error:
                logger.warning(f"JSON cleanup also failed: {cleanup_error}")
                # Save final failed attempt
                self._save_debug_json(cleaned_response if 'cleaned_response' in locals() else response_text, 
                                    coord_id, "cleanup_failed")
            
            return self._create_fallback_qa_report(coordinate_solution, tolerance_mm)
            
        except Exception as e:
            logger.error(f"Gemini verification failed: {e}")
            return self._create_fallback_qa_report(coordinate_solution, tolerance_mm)

    def _clean_json_response(self, response_text: str) -> str:
        """Clean JSON response by removing control characters and fixing common issues."""
        import re
        import json
        
        # First, handle markdown code blocks (```json ... ```)
        code_block_match = re.search(r'```[a-zA-Z]*\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if code_block_match:
            response_text = code_block_match.group(1)
        
        # Remove control characters (except valid JSON whitespace)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', response_text)
        
        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # Fix corrupted HTML/SVG closing tags (Issue #3)
        def fix_corrupted_closing_tags(text):
            corrupted_tag_pattern = r'\\([a-zA-Z][a-zA-Z0-9]*)'
            
            def fix_tag(match):
                tag_name = match.group(1)
                remaining_text = text[match.end():]
                if remaining_text.startswith('>'):
                    return f'</{tag_name}'
                else:
                    return match.group(0)
            
            return re.sub(corrupted_tag_pattern, fix_tag, text)
        
        cleaned = fix_corrupted_closing_tags(cleaned)
        
        # If it's still not valid JSON, try a more aggressive approach
        try:
            json.loads(cleaned)
            return cleaned.strip()
        except:
            # More aggressive cleaning for problematic cases
            
            # Extract just the JSON object structure
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                extracted = json_match.group()
                
                # Clean up the extracted JSON more aggressively
                extracted = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', extracted)
                extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)
                
                # Try to fix common issues with multiline strings
                # Replace literal newlines in string values with \n
                in_string = False
                escape_next = False
                result = ""
                
                for char in extracted:
                    if escape_next:
                        result += char
                        escape_next = False
                    elif char == '\\':
                        result += char
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                        result += char
                    elif char == '\n' and in_string:
                        result += '\\n'  # Escape newlines inside strings
                    elif char == '\t' and in_string:
                        result += '\\t'  # Escape tabs inside strings  
                    elif char == '\r' and in_string:
                        result += '\\r'  # Escape carriage returns inside strings
                    else:
                        result += char
                
                return result.strip()
        
        return cleaned.strip()
    
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
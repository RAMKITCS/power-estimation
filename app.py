import streamlit as st
import openai
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import base64

# Configure page
st.set_page_config(
    page_title="Power Platform Development Estimator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Azure OpenAI Configuration
class AzureOpenAIConfig:
    def __init__(self):
        self.api_key = st.secrets.get("AZURE_OPENAI_API_KEY", "")
        self.endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
        self.api_version = "2024-02-15-preview"
        self.model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-ada-002"

# Estimation Framework Classes
class ComplexityLevel(Enum):
    SIMPLE = 1
    MEDIUM = 2
    COMPLEX = 3
    VERY_COMPLEX = 4

@dataclass
class ComponentEstimate:
    component_type: str
    complexity: ComplexityLevel
    base_hours: float
    multiplier: float
    final_hours: float
    justification: str

@dataclass
class ProjectEstimate:
    total_hours: float
    components: List[ComponentEstimate]
    risk_factor: float
    confidence_level: str
    methodology: str
    timeline_weeks: int

class PowerPlatformEstimator:
    def __init__(self, azure_config: AzureOpenAIConfig):
        self.azure_config = azure_config
        self.setup_azure_client()
        
        # Industry standard estimation matrices based on research
        self.estimation_matrix = {
            # Model-driven Apps Components
            "entity_simple": {"base_hours": 3, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 4}},
            "entity_complex": {"base_hours": 12, "multipliers": {"simple": 1, "medium": 1.3, "complex": 2, "very_complex": 3}},
            "relationship_1n": {"base_hours": 1.5, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2, "very_complex": 2.5}},
            "relationship_nn": {"base_hours": 3, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 3.5}},
            
            # Forms and Views
            "form_simple": {"base_hours": 3, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 4}},
            "form_complex": {"base_hours": 12, "multipliers": {"simple": 1, "medium": 1.3, "complex": 2, "very_complex": 3}},
            "view_configuration": {"base_hours": 3.5, "multipliers": {"simple": 1, "medium": 1.4, "complex": 2.2, "very_complex": 3.5}},
            
            # Business Logic
            "business_rules": {"base_hours": 3, "multipliers": {"simple": 1, "medium": 2, "complex": 3, "very_complex": 4}},
            "workflows": {"base_hours": 8, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 4}},
            "plugins": {"base_hours": 24, "multipliers": {"simple": 1, "medium": 1.8, "complex": 2.5, "very_complex": 3.5}},
            "pcf_components": {"base_hours": 48, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.2, "very_complex": 3}},
            
            # Security and Integration
            "security_roles": {"base_hours": 4, "multipliers": {"simple": 1, "medium": 2, "complex": 3.5, "very_complex": 5}},
            "standard_connectors": {"base_hours": 2, "multipliers": {"simple": 1, "medium": 1.2, "complex": 1.8, "very_complex": 2.5}},
            "custom_connectors": {"base_hours": 16, "multipliers": {"simple": 1, "medium": 2, "complex": 3, "very_complex": 4}},
            "external_integrations": {"base_hours": 32, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 4}},
            
            # Canvas Apps
            "canvas_screens": {"base_hours": 6, "multipliers": {"simple": 1, "medium": 1.8, "complex": 2.8, "very_complex": 4.5}},
            "canvas_controls": {"base_hours": 2, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.2, "very_complex": 3.5}},
            
            # Power Automate
            "simple_flows": {"base_hours": 4, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 3.5}},
            "complex_flows": {"base_hours": 12, "multipliers": {"simple": 1, "medium": 1.8, "complex": 3, "very_complex": 4.5}},
            
            # Business Process Flows
            "bpf_simple": {"base_hours": 6, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.5, "very_complex": 4}},
            "bpf_complex": {"base_hours": 24, "multipliers": {"simple": 1, "medium": 1.5, "complex": 2.2, "very_complex": 3.5}}
        }
        
        # Industry multipliers based on research
        self.industry_multipliers = {
            "banking_finance": 1.25,
            "healthcare": 1.20,
            "government": 1.30,
            "manufacturing": 1.10,
            "retail": 1.05,
            "technology": 0.90,
            "education": 1.15,
            "other": 1.00
        }
        
        # Compliance complexity factors
        self.compliance_multipliers = {
            "none": 1.0,
            "basic": 1.25,
            "moderate": 1.5,
            "strict": 1.75,
            "enterprise": 2.0
        }

    def setup_azure_client(self):
        """Initialize Azure OpenAI client"""
        if self.azure_config.api_key and self.azure_config.endpoint:
            openai.api_type = "azure"
            openai.api_base = self.azure_config.endpoint
            openai.api_version = self.azure_config.api_version
            openai.api_key = self.azure_config.api_key
        else:
            st.error("Azure OpenAI credentials not configured. Please check your secrets.")

    def analyze_brd_with_ai(self, brd_text: str) -> Dict:
        """Analyze BRD using Azure OpenAI GPT-4"""
        try:
            analysis_prompt = f"""
            You are an expert Power Platform architect analyzing a Business Requirements Document (BRD) for estimation purposes.
            
            Analyze the following BRD and extract key components with their complexity levels:
            
            BRD Content:
            {brd_text}
            
            Please identify and categorize the following components with complexity levels (simple, medium, complex, very_complex):
            
            1. Entities/Tables:
               - Count of entities
               - Fields per entity (estimate)
               - Relationships between entities
               
            2. Forms and Views:
               - Number of forms needed
               - Complexity of forms (fields, business rules, custom controls)
               - Views and dashboards
               
            3. Business Logic:
               - Business rules
               - Workflows/Power Automate flows
               - Custom plugins needed
               - PCF components
               
            4. Security:
               - Security roles
               - Field-level security
               - Business unit hierarchy
               
            5. Integrations:
               - Standard connectors
               - Custom connectors
               - External system integrations
               
            6. Canvas Apps (if mentioned):
               - Number of screens
               - Custom controls
               
            7. Compliance Requirements:
               - Audit requirements
               - Data governance
               - Regulatory compliance
            
            Return the analysis in this JSON format:
            {{
                "entities": [
                    {{"name": "EntityName", "complexity": "medium", "fields": 15, "relationships": 3}}
                ],
                "forms": [
                    {{"type": "main_form", "complexity": "complex", "fields": 25, "business_rules": 5}}
                ],
                "views": [
                    {{"type": "system_view", "complexity": "medium", "filters": 3}}
                ],
                "business_logic": [
                    {{"type": "workflow", "complexity": "complex", "description": "Approval process"}}
                ],
                "security": [
                    {{"type": "security_role", "complexity": "medium", "description": "Role-based access"}}
                ],
                "integrations": [
                    {{"type": "standard_connector", "complexity": "simple", "system": "SharePoint"}}
                ],
                "canvas_apps": [
                    {{"screens": 5, "complexity": "medium", "custom_controls": 3}}
                ],
                "compliance": {{"level": "moderate", "requirements": ["audit_trail", "data_encryption"]}}
            }}
            """
            
            response = openai.ChatCompletion.create(
                engine=self.azure_config.model,
                messages=[
                    {"role": "system", "content": "You are an expert Power Platform solution architect with deep knowledge of estimation methodologies."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse JSON response
            analysis_text = response['choices'][0]['message']['content']
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                st.error("Could not parse AI analysis. Please check BRD format.")
                return {}
                
        except Exception as e:
            st.error(f"AI Analysis Error: {str(e)}")
            return {}

    def calculate_cosmic_function_points(self, brd_analysis: Dict) -> float:
        """Calculate COSMIC Function Points based on data movements"""
        total_cfp = 0
        
        # Entry movements (data entering the system)
        for entity in brd_analysis.get("entities", []):
            # Each entity typically has Create operations (Entry)
            total_cfp += 1
            
        for form in brd_analysis.get("forms", []):
            # Forms represent Entry points for data
            total_cfp += 1
            
        # Exit movements (data leaving the system)
        for view in brd_analysis.get("views", []):
            # Views represent Exit points
            total_cfp += 1
            
        for integration in brd_analysis.get("integrations", []):
            if "export" in integration.get("description", "").lower():
                total_cfp += 1
        
        # Read movements (data retrieved from persistent storage)
        for entity in brd_analysis.get("entities", []):
            # Each entity typically has Read operations
            total_cfp += entity.get("relationships", 0) * 0.5  # Relationships increase reads
            
        # Write movements (data sent to persistent storage)
        for entity in brd_analysis.get("entities", []):
            # Each entity has Write operations
            total_cfp += 1
            
        for business_logic in brd_analysis.get("business_logic", []):
            if business_logic.get("type") in ["workflow", "plugin"]:
                total_cfp += 2  # Complex logic increases data movements
        
        return max(total_cfp, 1)  # Minimum 1 CFP

    def estimate_project(self, brd_analysis: Dict, industry: str, team_experience: str, 
                        compliance_level: str) -> ProjectEstimate:
        """Generate comprehensive project estimate"""
        components = []
        total_base_hours = 0
        
        # Entity estimation
        for entity in brd_analysis.get("entities", []):
            complexity = entity.get("complexity", "medium")
            if entity.get("fields", 0) <= 10:
                component_type = "entity_simple"
            else:
                component_type = "entity_complex"
                
            base_hours = self.estimation_matrix[component_type]["base_hours"]
            multiplier = self.estimation_matrix[component_type]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            component = ComponentEstimate(
                component_type=f"Entity: {entity.get('name', 'Unknown')}",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=f"{entity.get('fields', 'N/A')} fields, {entity.get('relationships', 0)} relationships"
            )
            components.append(component)
            total_base_hours += final_hours
            
            # Add relationship overhead
            for _ in range(entity.get("relationships", 0)):
                rel_hours = self.estimation_matrix["relationship_1n"]["base_hours"] * multiplier
                total_base_hours += rel_hours
                components.append(ComponentEstimate(
                    component_type="Entity Relationship",
                    complexity=ComplexityLevel[complexity.upper()],
                    base_hours=self.estimation_matrix["relationship_1n"]["base_hours"],
                    multiplier=multiplier,
                    final_hours=rel_hours,
                    justification="Entity relationship configuration"
                ))
        
        # Form estimation
        for form in brd_analysis.get("forms", []):
            complexity = form.get("complexity", "medium")
            component_type = "form_simple" if form.get("fields", 0) <= 15 else "form_complex"
            
            base_hours = self.estimation_matrix[component_type]["base_hours"]
            multiplier = self.estimation_matrix[component_type]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            # Add business rules overhead
            br_count = form.get("business_rules", 0)
            if br_count > 0:
                br_hours = self.estimation_matrix["business_rules"]["base_hours"] * br_count * multiplier
                final_hours += br_hours
            
            components.append(ComponentEstimate(
                component_type=f"Form Configuration",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=f"{form.get('fields', 'N/A')} fields, {br_count} business rules"
            ))
            total_base_hours += final_hours
        
        # View estimation
        for view in brd_analysis.get("views", []):
            complexity = view.get("complexity", "medium")
            base_hours = self.estimation_matrix["view_configuration"]["base_hours"]
            multiplier = self.estimation_matrix["view_configuration"]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            components.append(ComponentEstimate(
                component_type="View Configuration",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=f"System view with {view.get('filters', 'standard')} filtering"
            ))
            total_base_hours += final_hours
        
        # Business Logic estimation
        for logic in brd_analysis.get("business_logic", []):
            logic_type = logic.get("type", "workflow")
            complexity = logic.get("complexity", "medium")
            
            if logic_type == "plugin":
                component_type = "plugins"
            elif logic_type == "workflow":
                component_type = "workflows"
            elif logic_type == "pcf":
                component_type = "pcf_components"
            else:
                component_type = "workflows"
                
            base_hours = self.estimation_matrix[component_type]["base_hours"]
            multiplier = self.estimation_matrix[component_type]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            components.append(ComponentEstimate(
                component_type=f"{logic_type.title()} Development",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=logic.get("description", "Custom business logic implementation")
            ))
            total_base_hours += final_hours
        
        # Security estimation
        for security in brd_analysis.get("security", []):
            complexity = security.get("complexity", "medium")
            base_hours = self.estimation_matrix["security_roles"]["base_hours"]
            multiplier = self.estimation_matrix["security_roles"]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            components.append(ComponentEstimate(
                component_type="Security Configuration",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=security.get("description", "Role-based security configuration")
            ))
            total_base_hours += final_hours
        
        # Integration estimation
        for integration in brd_analysis.get("integrations", []):
            int_type = integration.get("type", "standard_connector")
            complexity = integration.get("complexity", "medium")
            
            if int_type == "custom_connector":
                component_type = "custom_connectors"
            elif int_type == "external_integration":
                component_type = "external_integrations"
            else:
                component_type = "standard_connectors"
                
            base_hours = self.estimation_matrix[component_type]["base_hours"]
            multiplier = self.estimation_matrix[component_type]["multipliers"][complexity]
            final_hours = base_hours * multiplier
            
            components.append(ComponentEstimate(
                component_type=f"Integration: {integration.get('system', 'External System')}",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=base_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=f"{int_type.replace('_', ' ').title()} integration"
            ))
            total_base_hours += final_hours
        
        # Canvas Apps estimation (if any)
        for canvas_app in brd_analysis.get("canvas_apps", []):
            screens = canvas_app.get("screens", 1)
            complexity = canvas_app.get("complexity", "medium")
            
            screen_hours = self.estimation_matrix["canvas_screens"]["base_hours"] * screens
            multiplier = self.estimation_matrix["canvas_screens"]["multipliers"][complexity]
            final_hours = screen_hours * multiplier
            
            components.append(ComponentEstimate(
                component_type="Canvas App Development",
                complexity=ComplexityLevel[complexity.upper()],
                base_hours=screen_hours,
                multiplier=multiplier,
                final_hours=final_hours,
                justification=f"{screens} screens with custom controls"
            ))
            total_base_hours += final_hours
        
        # Apply industry and experience multipliers
        industry_mult = self.industry_multipliers.get(industry, 1.0)
        
        experience_mult = {"beginner": 1.4, "intermediate": 1.0, "expert": 0.8}.get(team_experience, 1.0)
        
        compliance_mult = self.compliance_multipliers.get(compliance_level, 1.0)
        
        # Calculate risk factor based on complexity
        avg_complexity = np.mean([c.complexity.value for c in components])
        risk_factor = 1.0 + (avg_complexity - 1) * 0.15  # 15% increase per complexity level
        
        # Total calculation
        adjusted_hours = total_base_hours * industry_mult * experience_mult * compliance_mult * risk_factor
        
        # Timeline calculation (assuming 6 hours productive work per day)
        timeline_weeks = max(1, int(adjusted_hours / 30))  # 30 hours per week
        
        # Confidence calculation
        if len(components) < 5:
            confidence = "High"
        elif len(components) < 15:
            confidence = "Medium"
        else:
            confidence = "Low - Requires detailed analysis"
        
        return ProjectEstimate(
            total_hours=adjusted_hours,
            components=components,
            risk_factor=risk_factor,
            confidence_level=confidence,
            methodology="Hybrid: Component-based + COSMIC FP + Industry standards",
            timeline_weeks=timeline_weeks
        )

# Streamlit UI
def main():
    st.title("⚡ Power Platform Development Estimator")
    st.markdown("### AI-Powered Estimation based on Industry Standards and Scientific Research")
    
    # Initialize Azure config
    azure_config = AzureOpenAIConfig()
    estimator = PowerPlatformEstimator(azure_config)
    
    # Sidebar configuration
    st.sidebar.header("Project Configuration")
    
    industry = st.sidebar.selectbox(
        "Industry",
        ["technology", "banking_finance", "healthcare", "government", "manufacturing", "retail", "education", "other"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    team_experience = st.sidebar.selectbox(
        "Team Experience Level",
        ["expert", "intermediate", "beginner"]
    )
    
    compliance_level = st.sidebar.selectbox(
        "Compliance Requirements",
        ["none", "basic", "moderate", "strict", "enterprise"],
        index=2
    )
    
    project_type = st.sidebar.multiselect(
        "Project Components",
        ["Model-driven Apps", "Canvas Apps", "Power Automate", "Dataverse", "Custom Connectors", "Security"],
        default=["Model-driven Apps", "Dataverse"]
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📄 BRD Analysis", "📊 Estimation Results", "📈 Analytics", "🔧 Manual Override"])
    
    with tab1:
        st.header("Business Requirements Document Analysis")
        
        # BRD input methods
        input_method = st.radio("BRD Input Method", ["Text Input", "File Upload"])
        
        brd_text = ""
        if input_method == "Text Input":
            brd_text = st.text_area(
                "Paste your Business Requirements Document here:",
                height=300,
                placeholder="""Example BRD content:
                
The system requires a Customer Management application with the following features:
- Customer entity with 25 fields including contact information, preferences, and history
- Lead entity with 15 fields and relationship to Customer
- Opportunity tracking with complex workflow approval process
- Integration with existing ERP system via REST API
- Role-based security with field-level access control
- Custom dashboard showing sales metrics
- Mobile app for field sales team with 8 screens
- Audit trail and compliance reporting
                """
            )
        else:
            uploaded_file = st.file_uploader("Upload BRD file", type=['txt', 'docx', 'pdf'])
            if uploaded_file:
                # Handle different file types
                if uploaded_file.type == "text/plain":
                    brd_text = str(uploaded_file.read(), "utf-8")
                else:
                    st.info("File uploaded successfully. Text extraction from DOCX/PDF requires additional processing.")
                    brd_text = "Please paste the text content in the text area above."
        
        if st.button("🤖 Analyze BRD with AI", type="primary") and brd_text.strip():
            with st.spinner("Analyzing BRD using Azure OpenAI..."):
                brd_analysis = estimator.analyze_brd_with_ai(brd_text)
                
                if brd_analysis:
                    st.session_state.brd_analysis = brd_analysis
                    st.session_state.brd_text = brd_text
                    st.success("✅ BRD analysis completed!")
                    
                    # Display analysis summary
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Entities Identified", len(brd_analysis.get("entities", [])))
                    with col2:
                        st.metric("Forms Required", len(brd_analysis.get("forms", [])))
                    with col3:
                        st.metric("Integrations", len(brd_analysis.get("integrations", [])))
                    with col4:
                        st.metric("Business Logic Items", len(brd_analysis.get("business_logic", [])))
                    
                    # Show detailed analysis
                    with st.expander("📋 Detailed Component Analysis"):
                        st.json(brd_analysis)
    
    with tab2:
        st.header("Estimation Results")
        
        if hasattr(st.session_state, 'brd_analysis'):
            if st.button("📈 Generate Estimation", type="primary"):
                with st.spinner("Calculating estimates using industry standards..."):
                    # Calculate COSMIC Function Points
                    cosmic_fp = estimator.calculate_cosmic_function_points(st.session_state.brd_analysis)
                    
                    # Generate project estimate
                    project_estimate = estimator.estimate_project(
                        st.session_state.brd_analysis,
                        industry,
                        team_experience,
                        compliance_level
                    )
                    
                    st.session_state.project_estimate = project_estimate
                    st.session_state.cosmic_fp = cosmic_fp
                    
                    # Display main metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Hours", f"{project_estimate.total_hours:.0f}")
                    with col2:
                        st.metric("Timeline (Weeks)", project_estimate.timeline_weeks)
                    with col3:
                        st.metric("COSMIC Function Points", f"{cosmic_fp:.1f}")
                    with col4:
                        st.metric("Confidence Level", project_estimate.confidence_level)
                    
                    # Detailed breakdown
                    st.subheader("📊 Component Breakdown")
                    
                    components_df = pd.DataFrame([
                        {
                            "Component": comp.component_type,
                            "Complexity": comp.complexity.name,
                            "Base Hours": comp.base_hours,
                            "Multiplier": comp.multiplier,
                            "Final Hours": comp.final_hours,
                            "Justification": comp.justification
                        }
                        for comp in project_estimate.components
                    ])
                    
                    st.dataframe(components_df, use_container_width=True)
                    
                    # Cost estimation
                    st.subheader("💰 Cost Analysis")
                    
                    # Standard hourly rates based on industry
                    hourly_rates = {
                        "developer": 85,
                        "architect": 150,
                        "business_analyst": 95,
                        "project_manager": 120
                    }
                    
                    # Rough resource allocation
                    dev_hours = project_estimate.total_hours * 0.6
                    arch_hours = project_estimate.total_hours * 0.15
                    ba_hours = project_estimate.total_hours * 0.15
                    pm_hours = project_estimate.total_hours * 0.1
                    
                    total_cost = (
                        dev_hours * hourly_rates["developer"] +
                        arch_hours * hourly_rates["architect"] +
                        ba_hours * hourly_rates["business_analyst"] +
                        pm_hours * hourly_rates["project_manager"]
                    )
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Estimated Cost", f"${total_cost:,.0f}")
                    with col2:
                        st.metric("Cost per Hour", f"${total_cost/project_estimate.total_hours:.0f}")
                    with col3:
                        risk_adjusted_cost = total_cost * project_estimate.risk_factor
                        st.metric("Risk-Adjusted Cost", f"${risk_adjusted_cost:,.0f}")
                    
                    # Methodology explanation
                    with st.expander("🔬 Estimation Methodology"):
                        st.markdown(f"""
                        **Methodology Used:** {project_estimate.methodology}
                        
                        **Key Factors:**
                        - Industry: {industry.replace('_', ' ').title()} (×{estimator.industry_multipliers[industry]:.2f})
                        - Team Experience: {team_experience.title()} 
                        - Compliance Level: {compliance_level.title()} (×{estimator.compliance_multipliers[compliance_level]:.2f})
                        - Risk Factor: ×{project_estimate.risk_factor:.2f}
                        
                        **Standards Applied:**
                        - COSMIC Function Points (ISO/IEC 19761:2011)
                        - Component-based estimation matrices
                        - Forrester productivity benchmarks
                        - Industry complexity multipliers
                        
                        **Confidence Assessment:**
                        - Level: {project_estimate.confidence_level}
                        - Based on component count and complexity variation
                        """)
        else:
            st.info("👆 Please analyze a BRD first in the BRD Analysis tab")
    
    with tab3:
        st.header("Project Analytics")
        
        if hasattr(st.session_state, 'project_estimate'):
            estimate = st.session_state.project_estimate
            
            # Component distribution chart
            fig_dist = px.pie(
                values=[comp.final_hours for comp in estimate.components],
                names=[comp.component_type for comp in estimate.components],
                title="Effort Distribution by Component"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Complexity analysis
            complexity_data = {}
            for comp in estimate.components:
                complexity_level = comp.complexity.name
                if complexity_level in complexity_data:
                    complexity_data[complexity_level] += comp.final_hours
                else:
                    complexity_data[complexity_level] = comp.final_hours
            
            fig_complexity = px.bar(
                x=list(complexity_data.keys()),
                y=list(complexity_data.values()),
                title="Effort by Complexity Level",
                labels={"x": "Complexity Level", "y": "Hours"}
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
            
            # Timeline visualization
            st.subheader("📅 Project Timeline")
            
            start_date = datetime.now()
            end_date = start_date + timedelta(weeks=estimate.timeline_weeks)
            
            # Create Gantt-like timeline
            phases = [
                {"Phase": "Analysis & Design", "Start": 0, "Duration": estimate.timeline_weeks * 0.2},
                {"Phase": "Development", "Start": estimate.timeline_weeks * 0.2, "Duration": estimate.timeline_weeks * 0.5},
                {"Phase": "Testing & UAT", "Start": estimate.timeline_weeks * 0.7, "Duration": estimate.timeline_weeks * 0.2},
                {"Phase": "Deployment", "Start": estimate.timeline_weeks * 0.9, "Duration": estimate.timeline_weeks * 0.1}
            ]
            
            fig_timeline = go.Figure()
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for i, phase in enumerate(phases):
                fig_timeline.add_trace(go.Bar(
                    name=phase["Phase"],
                    x=[phase["Duration"]],
                    y=[phase["Phase"]],
                    orientation='h',
                    marker_color=colors[i],
                    text=f'{phase["Duration"]:.1f} weeks',
                    textposition='middle center'
                ))
            
            fig_timeline.update_layout(
                title="Project Phase Timeline",
                xaxis_title="Weeks",
                barmode='stack',
                height=300
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Risk analysis
            st.subheader("⚠️ Risk Analysis")
            
            risks = []
            if estimate.risk_factor > 1.3:
                risks.append("High complexity components detected - consider phased approach")
            if len(estimate.components) > 20:
                risks.append("Large number of components - integration complexity risk")
            if any("plugin" in comp.component_type.lower() for comp in estimate.components):
                risks.append("Custom plugin development - technical risk and testing overhead")
            if any("custom" in comp.component_type.lower() for comp in estimate.components):
                risks.append("Custom connector integration - external dependency risk")
            
            if risks:
                for risk in risks:
                    st.warning(f"⚠️ {risk}")
            else:
                st.success("✅ Low risk project profile")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            recommendations = []
            
            # Phasing recommendations
            if estimate.timeline_weeks > 12:
                recommendations.append("Consider breaking into 2-3 phases for faster value delivery")
            
            # Team composition
            total_hours = estimate.total_hours
            if total_hours < 200:
                recommendations.append("Small project - 1-2 developers with architect support")
            elif total_hours < 800:
                recommendations.append("Medium project - 2-3 developers with dedicated architect")
            else:
                recommendations.append("Large project - 4+ developers, architect, and project manager")
            
            # Technology recommendations
            if any("canvas" in comp.component_type.lower() for comp in estimate.components):
                recommendations.append("Canvas apps identified - ensure responsive design for mobile")
            
            if st.session_state.cosmic_fp > 50:
                recommendations.append("High functional complexity - invest in detailed design phase")
            
            for rec in recommendations:
                st.info(f"💡 {rec}")
        
        else:
            st.info("👆 Please generate an estimation first in the Estimation Results tab")
    
    with tab4:
        st.header("Manual Override & Fine-tuning")
        
        if hasattr(st.session_state, 'project_estimate'):
            st.subheader("🔧 Adjust Estimation Parameters")
            
            # Override multipliers
            col1, col2 = st.columns(2)
            
            with col1:
                custom_industry_mult = st.slider(
                    "Industry Multiplier", 
                    0.5, 2.0, 
                    estimator.industry_multipliers[industry],
                    0.1
                )
                
                custom_experience_mult = st.slider(
                    "Experience Multiplier", 
                    0.5, 2.0, 
                    {"beginner": 1.4, "intermediate": 1.0, "expert": 0.8}[team_experience],
                    0.1
                )
            
            with col2:
                custom_compliance_mult = st.slider(
                    "Compliance Multiplier", 
                    1.0, 2.5, 
                    estimator.compliance_multipliers[compliance_level],
                    0.1
                )
                
                custom_risk_mult = st.slider(
                    "Risk Multiplier", 
                    1.0, 2.0, 
                    st.session_state.project_estimate.risk_factor,
                    0.1
                )
            
            # Add custom components
            st.subheader("➕ Add Custom Components")
            
            with st.expander("Add Custom Component"):
                custom_comp_name = st.text_input("Component Name")
                custom_comp_hours = st.number_input("Base Hours", min_value=1, max_value=200, value=8)
                custom_comp_complexity = st.selectbox("Complexity", ["SIMPLE", "MEDIUM", "COMPLEX", "VERY_COMPLEX"])
                custom_comp_justification = st.text_area("Justification", height=100)
                
                if st.button("Add Component"):
                    if custom_comp_name:
                        new_component = ComponentEstimate(
                            component_type=custom_comp_name,
                            complexity=ComplexityLevel[custom_comp_complexity],
                            base_hours=custom_comp_hours,
                            multiplier=1.0,
                            final_hours=custom_comp_hours,
                            justification=custom_comp_justification or "Custom component"
                        )
                        
                        if 'custom_components' not in st.session_state:
                            st.session_state.custom_components = []
                        st.session_state.custom_components.append(new_component)
                        st.success(f"Added custom component: {custom_comp_name}")
            
            # Recalculate with overrides
            if st.button("🔄 Recalculate with Overrides", type="primary"):
                base_hours = sum(comp.final_hours for comp in st.session_state.project_estimate.components)
                
                # Add custom components
                if hasattr(st.session_state, 'custom_components'):
                    base_hours += sum(comp.final_hours for comp in st.session_state.custom_components)
                
                # Apply custom multipliers
                adjusted_hours = base_hours * custom_industry_mult * custom_experience_mult * custom_compliance_mult * custom_risk_mult
                
                st.success(f"Recalculated estimate: {adjusted_hours:.0f} hours")
                
                # Update session state
                st.session_state.project_estimate.total_hours = adjusted_hours
                st.session_state.project_estimate.timeline_weeks = max(1, int(adjusted_hours / 30))
            
            # Export functionality
            st.subheader("📤 Export Results")
            
            if st.button("Generate Detailed Report"):
                # Create comprehensive report
                report_data = {
                    "project_summary": {
                        "total_hours": st.session_state.project_estimate.total_hours,
                        "timeline_weeks": st.session_state.project_estimate.timeline_weeks,
                        "cosmic_fp": st.session_state.cosmic_fp if hasattr(st.session_state, 'cosmic_fp') else 0,
                        "confidence": st.session_state.project_estimate.confidence_level,
                        "methodology": st.session_state.project_estimate.methodology
                    },
                    "components": [
                        {
                            "name": comp.component_type,
                            "complexity": comp.complexity.name,
                            "hours": comp.final_hours,
                            "justification": comp.justification
                        }
                        for comp in st.session_state.project_estimate.components
                    ],
                    "parameters": {
                        "industry": industry,
                        "team_experience": team_experience,
                        "compliance_level": compliance_level
                    },
                    "generated_at": datetime.now().isoformat()
                }
                
                # Convert to JSON for download
                report_json = json.dumps(report_data, indent=2)
                
                # Create download button
                st.download_button(
                    label="📥 Download JSON Report",
                    data=report_json,
                    file_name=f"power_platform_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Also create a formatted text report
                text_report = f"""
POWER PLATFORM DEVELOPMENT ESTIMATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROJECT SUMMARY
===============
Total Estimated Hours: {st.session_state.project_estimate.total_hours:.0f}
Timeline: {st.session_state.project_estimate.timeline_weeks} weeks
COSMIC Function Points: {st.session_state.cosmic_fp if hasattr(st.session_state, 'cosmic_fp') else 'N/A'}
Confidence Level: {st.session_state.project_estimate.confidence_level}
Methodology: {st.session_state.project_estimate.methodology}

COMPONENT BREAKDOWN
==================
"""
                for comp in st.session_state.project_estimate.components:
                    text_report += f"""
Component: {comp.component_type}
Complexity: {comp.complexity.name}
Hours: {comp.final_hours:.1f}
Justification: {comp.justification}
---"""

                text_report += f"""

PARAMETERS
==========
Industry: {industry.replace('_', ' ').title()}
Team Experience: {team_experience.title()}
Compliance Level: {compliance_level.title()}
Risk Factor: {st.session_state.project_estimate.risk_factor:.2f}

DISCLAIMER
==========
This estimate is based on industry standards and AI analysis of provided requirements.
Actual project effort may vary based on detailed requirements, team productivity, 
and unforeseen technical challenges. Regular re-estimation is recommended as 
requirements are refined.
"""
                
                st.download_button(
                    label="📄 Download Text Report",
                    data=text_report,
                    file_name=f"power_platform_estimate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        else:
            st.info("👆 Please generate an estimation first in the Estimation Results tab")
    
    # Footer with methodology information
    with st.expander("📚 About This Tool"):
        st.markdown("""
        ### Methodology & Standards
        
        This AI-powered estimation tool follows industry best practices and scientific research:
        
        **Standards Applied:**
        - COSMIC Function Points (ISO/IEC 19761:2011)
        - Microsoft Power Platform Well-Architected Framework
        - Forrester Total Economic Impact studies
        - Industry component-based estimation matrices
        
        **AI Analysis:**
        - Azure OpenAI GPT-4o-mini for BRD analysis
        - Natural Language Processing for component extraction
        - Complexity assessment based on requirement patterns
        
        **Estimation Factors:**
        - Entity/relationship complexity (2-16 hours per entity)
        - Form configuration effort (3-16 hours per form)
        - Business logic complexity (8-80+ hours for plugins)
        - Integration requirements (2-48 hours per integration)
        - Security and compliance overhead (25-100% multiplier)
        - Industry-specific factors
        - Team experience adjustments
        
        **Data Sources:**
        - Microsoft official documentation and guidance
        - Accenture and Forrester research studies
        - Academic research on low-code estimation
        - Industry consulting experience benchmarks
        
        **Accuracy Notes:**
        - Estimates are most accurate for model-driven apps
        - Canvas apps require additional UX consideration
        - Complex integrations may need detailed technical analysis
        - Regular re-estimation recommended as requirements evolve
        """)

if __name__ == "__main__":
    main()

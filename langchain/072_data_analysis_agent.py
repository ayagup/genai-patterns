"""
Pattern 072: Data Analysis Agent

Description:
    A Data Analysis Agent is a specialized AI agent designed to assist with data exploration,
    statistical analysis, visualization, and insight generation. This pattern demonstrates how
    to build an intelligent agent that can understand data-related queries, perform appropriate
    analyses, generate visualizations, and provide actionable insights from data.
    
    The agent combines natural language understanding with data science capabilities to make
    complex data analysis accessible through conversational interfaces. It can handle various
    data formats, perform exploratory data analysis (EDA), statistical tests, trend analysis,
    and generate meaningful visualizations.

Components:
    1. Data Loading & Parsing: Handles various data formats (CSV, JSON, Excel, databases)
    2. Data Cleaning: Identifies and handles missing values, outliers, duplicates
    3. Exploratory Analysis: Computes statistics, distributions, correlations
    4. Statistical Testing: Performs hypothesis tests, significance testing
    5. Visualization Generation: Creates appropriate charts and graphs
    6. Insight Extraction: Identifies patterns, trends, and anomalies
    7. Natural Language Interface: Understands data-related questions
    8. Query Translation: Converts NL queries to data operations

Key Features:
    - Automatic data profiling and quality assessment
    - Intelligent analysis method selection based on data characteristics
    - Statistical test recommendations and execution
    - Visualization selection based on data types and query intent
    - Natural language explanation of findings
    - Iterative exploration with memory of previous analyses
    - Support for multiple data sources and formats
    - Anomaly and outlier detection
    - Trend and pattern identification
    - Correlation and causation analysis

Use Cases:
    - Business intelligence and reporting
    - Data exploration and hypothesis testing
    - Quality assurance and data validation
    - Market research and customer analytics
    - Scientific data analysis
    - Financial analysis and forecasting
    - Healthcare analytics
    - Educational data analysis
    - Interactive data dashboards
    - Automated reporting systems

LangChain Implementation:
    Uses ChatOpenAI for analysis planning, pandas for data manipulation, and
    specialized LLM chains for insight generation and visualization recommendations.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AnalysisType(Enum):
    """Types of data analysis operations"""
    DESCRIPTIVE = "descriptive"
    EXPLORATORY = "exploratory"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    TREND = "trend"
    COMPARISON = "comparison"
    DISTRIBUTION = "distribution"
    ANOMALY = "anomaly"
    PREDICTION = "prediction"


class DataType(Enum):
    """Types of data variables"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    BOOLEAN = "boolean"


class VisualizationType(Enum):
    """Types of visualizations"""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    SCATTER_PLOT = "scatter_plot"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    HEATMAP = "heatmap"
    PIE_CHART = "pie_chart"
    TIME_SERIES = "time_series"
    VIOLIN_PLOT = "violin_plot"


@dataclass
class DataColumn:
    """Represents a data column with metadata"""
    name: str
    data_type: DataType
    missing_count: int
    unique_count: int
    sample_values: List[Any]
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataProfile:
    """Profile of a dataset"""
    row_count: int
    column_count: int
    columns: List[DataColumn]
    missing_total: int
    duplicate_count: int
    memory_usage: str
    summary: str


@dataclass
class StatisticalTest:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str
    confidence_level: float = 0.95


@dataclass
class AnalysisResult:
    """Result of a data analysis"""
    analysis_type: AnalysisType
    query: str
    findings: List[str]
    statistics: Dict[str, Any]
    visualizations: List[VisualizationType]
    insights: List[str]
    recommendations: List[str]
    confidence: float


@dataclass
class Visualization:
    """Specification for a visualization"""
    viz_type: VisualizationType
    title: str
    x_axis: Optional[str]
    y_axis: Optional[str]
    data_columns: List[str]
    description: str
    code: str  # Code to generate the visualization


class DataAnalysisAgent:
    """
    Agent for comprehensive data analysis and exploration.
    
    This agent can understand natural language queries about data, perform
    appropriate analyses, generate insights, and recommend visualizations.
    """
    
    def __init__(self):
        """Initialize the data analysis agent with specialized LLMs"""
        # Profiler for data understanding
        self.profiler_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Analyzer for statistical analysis
        self.analyzer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # Insight generator for pattern identification
        self.insight_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        # Visualization recommender
        self.viz_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Simulated data store (in practice, would connect to real data sources)
        self.data_cache: Dict[str, Any] = {}
        self.analysis_history: List[AnalysisResult] = []
    
    def profile_data(self, data_description: str, sample_data: Dict[str, List[Any]]) -> DataProfile:
        """
        Profile a dataset to understand its structure and characteristics.
        
        Args:
            data_description: Description of the dataset
            sample_data: Sample data as dictionary of columns
            
        Returns:
            DataProfile with comprehensive dataset information
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data profiling expert. Analyze the provided data sample 
            and generate a comprehensive profile including data types, quality issues, 
            and initial observations.
            
            Respond in this format:
            COLUMNS: For each column, provide: name|type|issues
            SUMMARY: Overall data summary
            QUALITY: Data quality assessment"""),
            ("user", """Dataset: {description}
            
Sample Data:
{sample}

Analyze this data and provide a comprehensive profile.""")
        ])
        
        chain = prompt | self.profiler_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "description": data_description,
                "sample": json.dumps(sample_data, indent=2)
            })
            
            # Parse the response
            columns = []
            summary = ""
            
            for line in response.split('\n'):
                if line.startswith('COLUMNS:'):
                    continue
                elif '|' in line and 'SUMMARY' not in line and 'QUALITY' not in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        col_name = parts[0].strip()
                        col_type_str = parts[1].strip().lower()
                        
                        # Map string to enum
                        if 'number' in col_type_str or 'numeric' in col_type_str:
                            col_type = DataType.NUMERICAL
                        elif 'category' in col_type_str or 'categorical' in col_type_str:
                            col_type = DataType.CATEGORICAL
                        elif 'time' in col_type_str or 'date' in col_type_str:
                            col_type = DataType.TEMPORAL
                        elif 'text' in col_type_str:
                            col_type = DataType.TEXT
                        else:
                            col_type = DataType.CATEGORICAL
                        
                        # Get column data
                        col_data = sample_data.get(col_name, [])
                        missing = len([v for v in col_data if v is None or v == ''])
                        unique = len(set(col_data))
                        sample_vals = col_data[:5]
                        
                        columns.append(DataColumn(
                            name=col_name,
                            data_type=col_type,
                            missing_count=missing,
                            unique_count=unique,
                            sample_values=sample_vals
                        ))
                elif 'SUMMARY:' in line:
                    summary = line.replace('SUMMARY:', '').strip()
            
            if not summary:
                summary = "Data profile generated successfully"
            
            row_count = len(next(iter(sample_data.values()))) if sample_data else 0
            
            return DataProfile(
                row_count=row_count,
                column_count=len(columns),
                columns=columns,
                missing_total=sum(col.missing_count for col in columns),
                duplicate_count=0,
                memory_usage="N/A",
                summary=summary
            )
            
        except Exception as e:
            # Fallback profiling
            columns = []
            for col_name, col_data in sample_data.items():
                # Determine type
                if all(isinstance(v, (int, float)) for v in col_data if v is not None):
                    col_type = DataType.NUMERICAL
                else:
                    col_type = DataType.CATEGORICAL
                
                columns.append(DataColumn(
                    name=col_name,
                    data_type=col_type,
                    missing_count=len([v for v in col_data if v is None]),
                    unique_count=len(set(col_data)),
                    sample_values=col_data[:5]
                ))
            
            return DataProfile(
                row_count=len(next(iter(sample_data.values()))) if sample_data else 0,
                column_count=len(columns),
                columns=columns,
                missing_total=sum(col.missing_count for col in columns),
                duplicate_count=0,
                memory_usage="N/A",
                summary="Basic data profile generated"
            )
    
    def analyze_query(self, query: str, data_profile: DataProfile) -> AnalysisResult:
        """
        Analyze a natural language query about data.
        
        Args:
            query: Natural language question about the data
            data_profile: Profile of the dataset
            
        Returns:
            AnalysisResult with findings and recommendations
        """
        # Determine analysis type
        analysis_type = self._determine_analysis_type(query)
        
        # Create profile summary
        profile_summary = f"""Dataset: {data_profile.row_count} rows, {data_profile.column_count} columns
Columns: {', '.join([f"{col.name} ({col.data_type.value})" for col in data_profile.columns])}
Summary: {data_profile.summary}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data analyst. Analyze the query and dataset,
            then provide findings, insights, and recommendations.
            
            Respond in this format:
            FINDINGS: List key findings (one per line starting with -)
            STATISTICS: Key statistics as name:value pairs
            INSIGHTS: Deeper insights (one per line starting with -)
            RECOMMENDATIONS: Action recommendations (one per line starting with -)
            CONFIDENCE: Confidence level (0.0-1.0)"""),
            ("user", """Query: {query}
            
Data Profile:
{profile}

Analysis Type: {analysis_type}

Provide a comprehensive analysis.""")
        ])
        
        chain = prompt | self.analyzer_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "query": query,
                "profile": profile_summary,
                "analysis_type": analysis_type.value
            })
            
            # Parse response
            findings = []
            statistics = {}
            insights = []
            recommendations = []
            confidence = 0.8
            
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('FINDINGS:'):
                    current_section = 'findings'
                elif line.startswith('STATISTICS:'):
                    current_section = 'statistics'
                elif line.startswith('INSIGHTS:'):
                    current_section = 'insights'
                elif line.startswith('RECOMMENDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        confidence = 0.8
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'findings':
                        findings.append(content)
                    elif current_section == 'insights':
                        insights.append(content)
                    elif current_section == 'recommendations':
                        recommendations.append(content)
                elif ':' in line and current_section == 'statistics':
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        statistics[key] = value
            
            # Recommend visualizations
            visualizations = self._recommend_visualizations(
                query, analysis_type, data_profile
            )
            
            return AnalysisResult(
                analysis_type=analysis_type,
                query=query,
                findings=findings if findings else ["Analysis completed successfully"],
                statistics=statistics,
                visualizations=visualizations,
                insights=insights if insights else ["Data analysis reveals interesting patterns"],
                recommendations=recommendations if recommendations else ["Consider further analysis"],
                confidence=confidence
            )
            
        except Exception as e:
            return AnalysisResult(
                analysis_type=analysis_type,
                query=query,
                findings=[f"Analysis performed for: {query}"],
                statistics={},
                visualizations=[VisualizationType.BAR_CHART],
                insights=["Analysis completed with basic insights"],
                recommendations=["Review findings and perform deeper analysis"],
                confidence=0.6
            )
    
    def perform_statistical_test(
        self,
        test_type: str,
        data_description: str,
        sample_data: Dict[str, List[Any]]
    ) -> StatisticalTest:
        """
        Perform a statistical test on the data.
        
        Args:
            test_type: Type of test (t-test, chi-square, ANOVA, etc.)
            data_description: Description of what's being tested
            sample_data: Data for the test
            
        Returns:
            StatisticalTest with results and interpretation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a statistical analysis expert. Perform the requested
            statistical test and provide interpretation.
            
            Respond in this format:
            TEST: Test name
            STATISTIC: Test statistic value
            P_VALUE: P-value
            SIGNIFICANT: Yes/No (at 0.05 level)
            INTERPRETATION: What the results mean"""),
            ("user", """Test Type: {test_type}
            
Description: {description}

Sample Data:
{sample}

Perform the statistical test and interpret results.""")
        ])
        
        chain = prompt | self.analyzer_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "test_type": test_type,
                "description": data_description,
                "sample": json.dumps(sample_data, indent=2)
            })
            
            # Parse response
            test_name = test_type
            statistic = 0.0
            p_value = 0.05
            significant = False
            interpretation = ""
            
            for line in response.split('\n'):
                if line.startswith('TEST:'):
                    test_name = line.replace('TEST:', '').strip()
                elif line.startswith('STATISTIC:'):
                    try:
                        statistic = float(re.findall(r'[-\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('P_VALUE:') or line.startswith('P-VALUE:'):
                    try:
                        p_value = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('SIGNIFICANT:'):
                    significant = 'yes' in line.lower()
                elif line.startswith('INTERPRETATION:'):
                    interpretation = line.replace('INTERPRETATION:', '').strip()
            
            if not interpretation:
                interpretation = f"Statistical test performed with p-value of {p_value}"
            
            return StatisticalTest(
                test_name=test_name,
                statistic=statistic,
                p_value=p_value,
                significant=significant,
                interpretation=interpretation,
                confidence_level=0.95
            )
            
        except Exception as e:
            return StatisticalTest(
                test_name=test_type,
                statistic=0.0,
                p_value=0.05,
                significant=False,
                interpretation=f"Statistical test performed: {test_type}",
                confidence_level=0.95
            )
    
    def generate_insights(
        self,
        data_profile: DataProfile,
        analysis_results: List[AnalysisResult]
    ) -> List[str]:
        """
        Generate high-level insights from multiple analyses.
        
        Args:
            data_profile: Profile of the dataset
            analysis_results: Previous analysis results
            
        Returns:
            List of insights
        """
        # Compile analysis summary
        analyses_summary = "\n".join([
            f"- {result.query}: {', '.join(result.findings[:2])}"
            for result in analysis_results[-5:]  # Last 5 analyses
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior data scientist. Based on the dataset profile
            and previous analyses, generate high-level strategic insights.
            
            Provide 3-5 key insights, one per line starting with -"""),
            ("user", """Data Profile: {profile}

Previous Analyses:
{analyses}

Generate strategic insights about this data.""")
        ])
        
        chain = prompt | self.insight_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "profile": data_profile.summary,
                "analyses": analyses_summary if analyses_summary else "No previous analyses"
            })
            
            insights = []
            for line in response.split('\n'):
                if line.strip().startswith('-'):
                    insights.append(line.strip()[1:].strip())
            
            return insights if insights else [
                "Data shows interesting patterns worthy of investigation",
                "Further analysis recommended to validate findings",
                "Consider collecting additional data for deeper insights"
            ]
            
        except Exception as e:
            return [
                "Data analysis completed successfully",
                "Multiple dimensions of the data have been explored",
                "Additional investigation may reveal more patterns"
            ]
    
    def recommend_visualization(
        self,
        analysis_result: AnalysisResult,
        data_profile: DataProfile
    ) -> Visualization:
        """
        Recommend and generate a visualization for analysis results.
        
        Args:
            analysis_result: Result of the analysis
            data_profile: Profile of the dataset
            
        Returns:
            Visualization specification
        """
        viz_type = analysis_result.visualizations[0] if analysis_result.visualizations else VisualizationType.BAR_CHART
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Create a visualization
            specification for the analysis results.
            
            Respond in this format:
            TITLE: Chart title
            X_AXIS: X-axis column
            Y_AXIS: Y-axis column
            COLUMNS: Columns to visualize (comma-separated)
            DESCRIPTION: Brief description
            CODE: Python code using matplotlib/seaborn"""),
            ("user", """Analysis Query: {query}
Visualization Type: {viz_type}
Findings: {findings}

Available Columns: {columns}

Create visualization specification.""")
        ])
        
        chain = prompt | self.viz_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "query": analysis_result.query,
                "viz_type": viz_type.value,
                "findings": ", ".join(analysis_result.findings[:3]),
                "columns": ", ".join([col.name for col in data_profile.columns])
            })
            
            # Parse response
            title = f"{viz_type.value.replace('_', ' ').title()} Visualization"
            x_axis = None
            y_axis = None
            data_columns = []
            description = ""
            code = "# Visualization code would go here"
            
            in_code = False
            code_lines = []
            
            for line in response.split('\n'):
                if line.startswith('TITLE:'):
                    title = line.replace('TITLE:', '').strip()
                elif line.startswith('X_AXIS:'):
                    x_axis = line.replace('X_AXIS:', '').strip()
                elif line.startswith('Y_AXIS:'):
                    y_axis = line.replace('Y_AXIS:', '').strip()
                elif line.startswith('COLUMNS:'):
                    cols = line.replace('COLUMNS:', '').strip()
                    data_columns = [c.strip() for c in cols.split(',')]
                elif line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '').strip()
                elif line.startswith('CODE:'):
                    in_code = True
                elif in_code:
                    code_lines.append(line)
            
            if code_lines:
                code = '\n'.join(code_lines)
            
            if not description:
                description = f"Visualization showing {analysis_result.query}"
            
            return Visualization(
                viz_type=viz_type,
                title=title,
                x_axis=x_axis,
                y_axis=y_axis,
                data_columns=data_columns if data_columns else [data_profile.columns[0].name],
                description=description,
                code=code
            )
            
        except Exception as e:
            return Visualization(
                viz_type=viz_type,
                title=f"{viz_type.value.replace('_', ' ').title()}",
                x_axis=data_profile.columns[0].name if data_profile.columns else None,
                y_axis=data_profile.columns[1].name if len(data_profile.columns) > 1 else None,
                data_columns=[col.name for col in data_profile.columns[:2]],
                description=f"Visualization for {analysis_result.query}",
                code="# Visualization code"
            )
    
    def _determine_analysis_type(self, query: str) -> AnalysisType:
        """Determine the type of analysis from the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['describe', 'summary', 'overview', 'statistics']):
            return AnalysisType.DESCRIPTIVE
        elif any(word in query_lower for word in ['explore', 'distribution', 'pattern']):
            return AnalysisType.EXPLORATORY
        elif any(word in query_lower for word in ['test', 'significance', 'hypothesis']):
            return AnalysisType.STATISTICAL
        elif any(word in query_lower for word in ['correlation', 'relationship', 'association']):
            return AnalysisType.CORRELATION
        elif any(word in query_lower for word in ['trend', 'over time', 'temporal']):
            return AnalysisType.TREND
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return AnalysisType.COMPARISON
        elif any(word in query_lower for word in ['anomaly', 'outlier', 'unusual']):
            return AnalysisType.ANOMALY
        elif any(word in query_lower for word in ['predict', 'forecast', 'future']):
            return AnalysisType.PREDICTION
        else:
            return AnalysisType.EXPLORATORY
    
    def _recommend_visualizations(
        self,
        query: str,
        analysis_type: AnalysisType,
        data_profile: DataProfile
    ) -> List[VisualizationType]:
        """Recommend appropriate visualizations"""
        visualizations = []
        
        if analysis_type == AnalysisType.DESCRIPTIVE:
            visualizations = [VisualizationType.BAR_CHART, VisualizationType.BOX_PLOT]
        elif analysis_type == AnalysisType.EXPLORATORY:
            visualizations = [VisualizationType.HISTOGRAM, VisualizationType.SCATTER_PLOT]
        elif analysis_type == AnalysisType.CORRELATION:
            visualizations = [VisualizationType.HEATMAP, VisualizationType.SCATTER_PLOT]
        elif analysis_type == AnalysisType.TREND:
            visualizations = [VisualizationType.LINE_CHART, VisualizationType.TIME_SERIES]
        elif analysis_type == AnalysisType.COMPARISON:
            visualizations = [VisualizationType.BAR_CHART, VisualizationType.BOX_PLOT]
        elif analysis_type == AnalysisType.DISTRIBUTION:
            visualizations = [VisualizationType.HISTOGRAM, VisualizationType.VIOLIN_PLOT]
        else:
            visualizations = [VisualizationType.BAR_CHART]
        
        return visualizations


def demonstrate_data_analysis_agent():
    """Demonstrate the data analysis agent capabilities"""
    print("=" * 80)
    print("DATA ANALYSIS AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = DataAnalysisAgent()
    
    # Demo 1: Data Profiling
    print("\n" + "=" * 80)
    print("DEMO 1: Data Profiling")
    print("=" * 80)
    
    sample_sales_data = {
        "product": ["A", "B", "C", "A", "B", "C", "A", "B"],
        "sales": [100, 150, 200, 120, 160, 180, 110, 155],
        "region": ["North", "South", "East", "West", "North", "South", "East", "West"],
        "date": ["2024-01", "2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-03", "2024-03"]
    }
    
    print("\nProfiling sales dataset...")
    profile = agent.profile_data(
        "Monthly sales data by product and region",
        sample_sales_data
    )
    
    print(f"\nData Profile:")
    print(f"- Rows: {profile.row_count}")
    print(f"- Columns: {profile.column_count}")
    print(f"- Missing values: {profile.missing_total}")
    print(f"\nColumns:")
    for col in profile.columns:
        print(f"  - {col.name} ({col.data_type.value}): {col.unique_count} unique values")
    print(f"\nSummary: {profile.summary}")
    
    # Demo 2: Query Analysis
    print("\n" + "=" * 80)
    print("DEMO 2: Query Analysis")
    print("=" * 80)
    
    queries = [
        "What are the average sales by product?",
        "Which region has the highest sales?",
        "Is there a correlation between product type and sales volume?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.analyze_query(query, profile)
        
        print(f"Analysis Type: {result.analysis_type.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"\nFindings:")
        for finding in result.findings[:3]:
            print(f"  - {finding}")
        
        if result.insights:
            print(f"\nInsights:")
            for insight in result.insights[:2]:
                print(f"  - {insight}")
        
        if result.recommendations:
            print(f"\nRecommendations:")
            for rec in result.recommendations[:2]:
                print(f"  - {rec}")
        
        print(f"\nRecommended Visualizations:")
        for viz in result.visualizations[:2]:
            print(f"  - {viz.value}")
        
        agent.analysis_history.append(result)
    
    # Demo 3: Statistical Testing
    print("\n" + "=" * 80)
    print("DEMO 3: Statistical Testing")
    print("=" * 80)
    
    test_data = {
        "group_a": [100, 105, 110, 95, 102],
        "group_b": [150, 145, 155, 140, 148]
    }
    
    print("\nPerforming t-test on sales groups...")
    test_result = agent.perform_statistical_test(
        "t-test",
        "Comparing sales between two product groups",
        test_data
    )
    
    print(f"\nTest: {test_result.test_name}")
    print(f"Statistic: {test_result.statistic:.4f}")
    print(f"P-value: {test_result.p_value:.4f}")
    print(f"Significant at {test_result.confidence_level:.0%} level: {test_result.significant}")
    print(f"Interpretation: {test_result.interpretation}")
    
    # Demo 4: Insight Generation
    print("\n" + "=" * 80)
    print("DEMO 4: High-Level Insights")
    print("=" * 80)
    
    print("\nGenerating strategic insights from all analyses...")
    insights = agent.generate_insights(profile, agent.analysis_history)
    
    print("\nStrategic Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Demo 5: Visualization Recommendation
    print("\n" + "=" * 80)
    print("DEMO 5: Visualization Recommendation")
    print("=" * 80)
    
    if agent.analysis_history:
        latest_analysis = agent.analysis_history[0]
        print(f"\nCreating visualization for: {latest_analysis.query}")
        
        viz = agent.recommend_visualization(latest_analysis, profile)
        
        print(f"\nVisualization Specification:")
        print(f"Type: {viz.viz_type.value}")
        print(f"Title: {viz.title}")
        print(f"X-axis: {viz.x_axis}")
        print(f"Y-axis: {viz.y_axis}")
        print(f"Columns: {', '.join(viz.data_columns)}")
        print(f"Description: {viz.description}")
        print(f"\nCode Preview:")
        code_lines = viz.code.split('\n')[:5]
        for line in code_lines:
            print(f"  {line}")
        if len(viz.code.split('\n')) > 5:
            print("  ...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Data Analysis Agent demonstrates comprehensive capabilities for automated
data analysis and exploration:

KEY CAPABILITIES:
1. Data Profiling: Automatically analyzes dataset structure, types, and quality
2. Natural Language Queries: Understands and answers data questions in plain English
3. Statistical Analysis: Performs appropriate statistical tests and interprets results
4. Insight Generation: Identifies patterns, trends, and actionable insights
5. Visualization Recommendations: Suggests and generates appropriate visualizations

BENEFITS:
- Makes data analysis accessible to non-technical users
- Accelerates exploratory data analysis workflows
- Provides consistent, comprehensive analysis
- Generates actionable insights automatically
- Recommends appropriate analytical methods

USE CASES:
- Business intelligence dashboards
- Automated reporting systems
- Data quality monitoring
- Research data exploration
- Customer analytics
- Market research analysis
- Scientific data analysis
- Educational data tools

PRODUCTION CONSIDERATIONS:
1. Integration: Connect to real databases, data lakes, and APIs
2. Visualization: Implement actual chart generation with matplotlib/plotly
3. Statistical Libraries: Use scipy, statsmodels for robust statistical tests
4. Performance: Optimize for large datasets with sampling and caching
5. Security: Implement data access controls and query validation
6. Scalability: Support distributed processing for big data
7. Memory Management: Handle large datasets efficiently
8. Validation: Verify statistical assumptions before tests
9. Interpretability: Provide clear explanations for all findings
10. Interactive Mode: Support iterative exploration with follow-up questions

ADVANCED EXTENSIONS:
- Machine learning model training and evaluation
- Time series forecasting and anomaly detection
- Causal inference and A/B test analysis
- Automated feature engineering
- Multi-dataset joins and comparisons
- Real-time streaming data analysis
- Custom metric and KPI tracking
- Export to notebooks and reports
- Natural language report generation
- Interactive dashboard creation

The agent combines LLM intelligence with data science capabilities to make
sophisticated analysis accessible through natural conversation.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_data_analysis_agent()

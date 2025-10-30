"""
Data Analysis Agent Pattern

Specialized agent for data exploration, analysis, and visualization.
Combines statistical analysis, data manipulation, and insight generation
for business intelligence and research tasks.

Use Cases:
- Exploratory data analysis
- Statistical analysis
- Data visualization
- Business intelligence
- Research data processing
- Automated reporting

Benefits:
- Accelerated data exploration
- Consistent analysis methodology
- Automated insight discovery
- Reproducible analysis
- Natural language data querying
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math
import random


class DataType(Enum):
    """Data types for analysis"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"


class AnalysisType(Enum):
    """Types of analysis"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


@dataclass
class Dataset:
    """Dataset representation"""
    name: str
    data: List[Dict[str, Any]]
    schema: Dict[str, DataType] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def get_column(self, column_name: str) -> List[Any]:
        """Extract column values"""
        return [row[column_name] for row in self.data if column_name in row]
    
    def get_columns(self) -> List[str]:
        """Get all column names"""
        if not self.data:
            return []
        return list(self.data[0].keys())


@dataclass
class StatisticalSummary:
    """Statistical summary of data"""
    column: str
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std_dev: Optional[float] = None
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    unique_count: Optional[int] = None
    missing_count: int = 0


@dataclass
class Insight:
    """Discovered insight from data"""
    title: str
    description: str
    insight_type: str
    confidence: float  # 0-1
    supporting_data: Dict[str, Any] = field(default_factory=dict)


class DataExplorer:
    """
    Explores datasets to understand structure and content
    """
    
    def __init__(self):
        self.insights: List[Insight] = []
    
    def explore_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Perform initial data exploration"""
        print(f"\n[Exploring Dataset] {dataset.name}")
        print(f"  Rows: {len(dataset)}")
        print(f"  Columns: {len(dataset.get_columns())}")
        
        exploration = {
            "name": dataset.name,
            "row_count": len(dataset),
            "column_count": len(dataset.get_columns()),
            "columns": dataset.get_columns(),
            "sample": dataset.data[:5] if dataset.data else []
        }
        
        return exploration
    
    def infer_schema(self, dataset: Dataset) -> Dict[str, DataType]:
        """Infer data types of columns"""
        schema = {}
        
        for column in dataset.get_columns():
            values = dataset.get_column(column)
            data_type = self._infer_type(values)
            schema[column] = data_type
        
        dataset.schema = schema
        
        print(f"\n[Schema Inferred]")
        for col, dtype in schema.items():
            print(f"  {col}: {dtype.value}")
        
        return schema
    
    def _infer_type(self, values: List[Any]) -> DataType:
        """Infer data type from values"""
        non_null = [v for v in values if v is not None]
        
        if not non_null:
            return DataType.TEXT
        
        # Check if numerical
        if all(isinstance(v, (int, float)) for v in non_null):
            return DataType.NUMERICAL
        
        # Check if temporal (simplified)
        if all(isinstance(v, str) and ('-' in v or '/' in v) for v in non_null[:10]):
            return DataType.TEMPORAL
        
        # Check unique count for categorical
        unique_ratio = len(set(non_null)) / len(non_null)
        if unique_ratio < 0.5:
            return DataType.CATEGORICAL
        
        return DataType.TEXT


class StatisticalAnalyzer:
    """
    Performs statistical analysis on data
    """
    
    def __init__(self):
        self.summaries: Dict[str, StatisticalSummary] = {}
    
    def analyze_column(
        self,
        dataset: Dataset,
        column: str
    ) -> StatisticalSummary:
        """Analyze single column"""
        values = dataset.get_column(column)
        
        # Count non-null values
        non_null = [v for v in values if v is not None]
        missing = len(values) - len(non_null)
        
        summary = StatisticalSummary(
            column=column,
            count=len(non_null),
            missing_count=missing
        )
        
        # Numerical analysis
        if dataset.schema.get(column) == DataType.NUMERICAL:
            numeric_values = [float(v) for v in non_null]
            
            if numeric_values:
                summary.mean = statistics.mean(numeric_values)
                summary.median = statistics.median(numeric_values)
                summary.min_val = min(numeric_values)
                summary.max_val = max(numeric_values)
                
                if len(numeric_values) > 1:
                    summary.std_dev = statistics.stdev(numeric_values)
        
        # Categorical/text analysis
        else:
            summary.unique_count = len(set(non_null))
            if non_null:
                summary.min_val = min(non_null)
                summary.max_val = max(non_null)
        
        self.summaries[column] = summary
        return summary
    
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, StatisticalSummary]:
        """Analyze all columns in dataset"""
        print(f"\n[Statistical Analysis]")
        
        for column in dataset.get_columns():
            summary = self.analyze_column(dataset, column)
            
            print(f"\n{column}:")
            print(f"  Count: {summary.count}")
            if summary.mean is not None:
                print(f"  Mean: {summary.mean:.2f}")
            if summary.median is not None:
                print(f"  Median: {summary.median:.2f}")
            if summary.std_dev is not None:
                print(f"  Std Dev: {summary.std_dev:.2f}")
            if summary.unique_count is not None:
                print(f"  Unique: {summary.unique_count}")
        
        return self.summaries
    
    def find_correlations(
        self,
        dataset: Dataset,
        column1: str,
        column2: str
    ) -> float:
        """Calculate correlation between two numerical columns"""
        values1 = [float(v) for v in dataset.get_column(column1) if v is not None]
        values2 = [float(v) for v in dataset.get_column(column2) if v is not None]
        
        if len(values1) != len(values2) or len(values1) < 2:
            return 0.0
        
        # Simple correlation coefficient
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        denominator = math.sqrt(
            sum((x - mean1) ** 2 for x in values1) *
            sum((y - mean2) ** 2 for y in values2)
        )
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


class InsightGenerator:
    """
    Generates insights from data analysis
    """
    
    def __init__(self):
        self.insights: List[Insight] = []
    
    def generate_insights(
        self,
        dataset: Dataset,
        summaries: Dict[str, StatisticalSummary]
    ) -> List[Insight]:
        """Generate insights from statistical summaries"""
        print(f"\n[Generating Insights]")
        
        insights = []
        
        # Check for missing data
        for column, summary in summaries.items():
            if summary.missing_count > 0:
                missing_pct = (summary.missing_count / (summary.count + summary.missing_count)) * 100
                
                if missing_pct > 10:
                    insight = Insight(
                        title=f"High Missing Data in {column}",
                        description=f"{column} has {missing_pct:.1f}% missing values",
                        insight_type="data_quality",
                        confidence=0.9,
                        supporting_data={"missing_count": summary.missing_count, "percentage": missing_pct}
                    )
                    insights.append(insight)
        
        # Check for outliers
        for column, summary in summaries.items():
            if summary.std_dev and summary.mean and summary.max_val is not None:
                # Check if max is far from mean
                z_score = abs(float(summary.max_val) - summary.mean) / summary.std_dev
                
                if z_score > 3:
                    insight = Insight(
                        title=f"Potential Outliers in {column}",
                        description=f"Maximum value {summary.max_val} is {z_score:.1f} standard deviations from mean",
                        insight_type="outlier",
                        confidence=0.8,
                        supporting_data={"z_score": z_score, "max_value": summary.max_val}
                    )
                    insights.append(insight)
        
        # Check for low cardinality
        for column, summary in summaries.items():
            if summary.unique_count and summary.count > 0:
                cardinality_ratio = summary.unique_count / summary.count
                
                if cardinality_ratio < 0.1 and summary.count > 10:
                    insight = Insight(
                        title=f"Low Cardinality in {column}",
                        description=f"{column} has only {summary.unique_count} unique values",
                        insight_type="cardinality",
                        confidence=0.7,
                        supporting_data={"unique_count": summary.unique_count}
                    )
                    insights.append(insight)
        
        self.insights = insights
        
        print(f"Generated {len(insights)} insights:")
        for insight in insights:
            print(f"  - {insight.title}")
        
        return insights


class DataAnalysisAgent:
    """
    Data Analysis Agent
    
    Combines data exploration, statistical analysis, and insight
    generation for automated data analysis tasks.
    """
    
    def __init__(self, name: str = "Data Analyst"):
        self.name = name
        self.explorer = DataExplorer()
        self.analyzer = StatisticalAnalyzer()
        self.insight_generator = InsightGenerator()
        self.datasets: Dict[str, Dataset] = {}
        
        print(f"[Data Analysis Agent] Initialized: {name}")
    
    def load_dataset(
        self,
        name: str,
        data: List[Dict[str, Any]]
    ) -> Dataset:
        """Load dataset for analysis"""
        print(f"\n[Loading Dataset] {name}")
        
        dataset = Dataset(name=name, data=data)
        self.datasets[name] = dataset
        
        # Automatic exploration
        self.explorer.explore_dataset(dataset)
        self.explorer.infer_schema(dataset)
        
        return dataset
    
    def analyze(self, dataset_name: str) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = self.datasets[dataset_name]
        
        # Perform analysis
        summaries = self.analyzer.analyze_dataset(dataset)
        
        # Generate insights
        insights = self.insight_generator.generate_insights(dataset, summaries)
        
        return {
            "dataset": dataset_name,
            "summaries": summaries,
            "insights": insights
        }
    
    def query(self, dataset_name: str, query: str) -> Any:
        """Natural language query on dataset"""
        print(f"\n[Query] {query}")
        
        if dataset_name not in self.datasets:
            return "Dataset not found"
        
        dataset = self.datasets[dataset_name]
        query_lower = query.lower()
        
        # Simple query parsing
        if "average" in query_lower or "mean" in query_lower:
            for column in dataset.get_columns():
                if column.lower() in query_lower:
                    if column in self.analyzer.summaries:
                        return self.analyzer.summaries[column].mean
        
        elif "count" in query_lower or "how many" in query_lower:
            return len(dataset)
        
        elif "maximum" in query_lower or "max" in query_lower:
            for column in dataset.get_columns():
                if column.lower() in query_lower:
                    if column in self.analyzer.summaries:
                        return self.analyzer.summaries[column].max_val
        
        elif "minimum" in query_lower or "min" in query_lower:
            for column in dataset.get_columns():
                if column.lower() in query_lower:
                    if column in self.analyzer.summaries:
                        return self.analyzer.summaries[column].min_val
        
        return "Query not understood"
    
    def compare_columns(
        self,
        dataset_name: str,
        column1: str,
        column2: str
    ) -> Dict[str, Any]:
        """Compare two columns"""
        print(f"\n[Comparing] {column1} vs {column2}")
        
        if dataset_name not in self.datasets:
            return {}
        
        dataset = self.datasets[dataset_name]
        
        # Calculate correlation if both numerical
        correlation = self.analyzer.find_correlations(dataset, column1, column2)
        
        result = {
            "column1": column1,
            "column2": column2,
            "correlation": correlation
        }
        
        print(f"  Correlation: {correlation:.3f}")
        
        return result
    
    def generate_report(self, dataset_name: str) -> str:
        """Generate analysis report"""
        if dataset_name not in self.datasets:
            return "Dataset not found"
        
        dataset = self.datasets[dataset_name]
        
        report = [
            f"DATA ANALYSIS REPORT: {dataset.name}",
            "=" * 50,
            f"\nDataset Overview:",
            f"  Rows: {len(dataset)}",
            f"  Columns: {len(dataset.get_columns())}",
            f"\nColumn Summary:"
        ]
        
        for column, summary in self.analyzer.summaries.items():
            report.append(f"\n{column}:")
            report.append(f"  Count: {summary.count}")
            if summary.mean:
                report.append(f"  Mean: {summary.mean:.2f}")
            if summary.unique_count:
                report.append(f"  Unique: {summary.unique_count}")
        
        report.append(f"\nKey Insights:")
        for insight in self.insight_generator.insights:
            report.append(f"  - {insight.title}")
            report.append(f"    {insight.description}")
        
        return "\n".join(report)


def demonstrate_data_analysis_agent():
    """
    Demonstrate Data Analysis Agent pattern
    """
    print("=" * 70)
    print("DATA ANALYSIS AGENT DEMONSTRATION")
    print("=" * 70)
    
    agent = DataAnalysisAgent("Business Intelligence Agent")
    
    # Example 1: Load and explore data
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Data Loading and Exploration")
    print("=" * 70)
    
    # Create sample dataset
    sales_data = [
        {"product": "A", "sales": 100, "price": 10.0, "region": "North"},
        {"product": "B", "sales": 150, "price": 15.0, "region": "South"},
        {"product": "A", "sales": 120, "price": 10.0, "region": "East"},
        {"product": "C", "sales": 200, "price": 20.0, "region": "West"},
        {"product": "B", "sales": 180, "price": 15.0, "region": "North"},
        {"product": "A", "sales": 90, "price": 10.0, "region": "South"},
        {"product": "C", "sales": 220, "price": 20.0, "region": "East"},
        {"product": "B", "sales": 160, "price": 15.0, "region": "West"},
    ]
    
    dataset = agent.load_dataset("sales_2024", sales_data)
    
    # Example 2: Statistical analysis
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Statistical Analysis")
    print("=" * 70)
    
    analysis_result = agent.analyze("sales_2024")
    
    # Example 3: Natural language queries
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Natural Language Queries")
    print("=" * 70)
    
    queries = [
        "What is the average sales?",
        "How many records are there?",
        "What is the maximum price?",
    ]
    
    for query in queries:
        result = agent.query("sales_2024", query)
        print(f"  Answer: {result}")
    
    # Example 4: Column comparison
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Column Comparison")
    print("=" * 70)
    
    agent.compare_columns("sales_2024", "sales", "price")
    
    # Example 5: Generate report
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Analysis Report")
    print("=" * 70)
    
    report = agent.generate_report("sales_2024")
    print(f"\n{report}")


if __name__ == "__main__":
    demonstrate_data_analysis_agent()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Data analysis agents automate exploratory analysis
2. Statistical summaries provide quick insights
3. Automated insight generation saves time
4. Natural language queries democratize data access
5. Reproducible analysis workflows

Best Practices:
- Start with data quality checks
- Infer schema automatically
- Generate multiple insight types
- Provide confidence scores
- Support natural language queries
- Create comprehensive reports
- Visualize key findings
- Validate statistical assumptions
    """)

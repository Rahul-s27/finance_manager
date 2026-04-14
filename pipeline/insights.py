"""
Spending Insights Module for Finance AutoML Manager.

Provides analysis and suggestions on spending patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpendingInsight:
    """Represents a single spending insight."""
    category: str
    metric_type: str  # 'overspending', 'underspending', 'top_merchant', 'frequent_small', etc.
    message: str
    severity: str  # 'high', 'medium', 'low'
    recommendation: str
    savings_potential: Optional[float] = None


class SpendingAnalyzer:
    """Analyze transaction data and provide spending insights."""
    
    # Category benchmarks (typical % of monthly income)
    BENCHMARKS = {
        'food': {'ideal_pct': 0.15, 'warning_pct': 0.25},
        'transport': {'ideal_pct': 0.10, 'warning_pct': 0.20},
        'shopping': {'ideal_pct': 0.10, 'warning_pct': 0.20},
        'groceries': {'ideal_pct': 0.15, 'warning_pct': 0.20},
        'bills': {'ideal_pct': 0.25, 'warning_pct': 0.35},
        'subscription': {'ideal_pct': 0.05, 'warning_pct': 0.10},
        'fuel': {'ideal_pct': 0.05, 'warning_pct': 0.10},
        'other_expenses': {'ideal_pct': 0.10, 'warning_pct': 0.15},
    }
    
    def __init__(self, df: pd.DataFrame, total_monthly_income: Optional[float] = None):
        """
        Initialize the SpendingAnalyzer.
        
        Args:
            df: DataFrame with transactions (must have 'category', 'amount', 'merchant')
            total_monthly_income: Optional monthly income for percentage calculations
        """
        self.df = df.copy()
        self.insights: List[SpendingInsight] = []
        self.total_spending = df[df['amount'] > 0]['amount'].sum() if 'amount' in df.columns else 0
        self.total_income = total_monthly_income
        
    def analyze_spending_by_category(self) -> pd.DataFrame:
        """
        Analyze spending breakdown by category.
        
        Returns:
            DataFrame with category statistics
        """
        if 'category' not in self.df.columns or 'amount' not in self.df.columns:
            return pd.DataFrame()
        
        # Filter expenses only (positive amounts)
        expenses = self.df[self.df['amount'] > 0].copy()
        
        stats = expenses.groupby('category').agg({
            'amount': ['sum', 'mean', 'count', 'std']
        }).reset_index()
        
        stats.columns = ['category', 'total_spent', 'avg_transaction', 'num_transactions', 'std_dev']
        stats['pct_of_total'] = stats['total_spent'] / self.total_spending * 100
        stats = stats.sort_values('total_spent', ascending=False)
        
        return stats
    
    def identify_overspending(self) -> List[SpendingInsight]:
        """
        Identify categories where spending exceeds benchmarks.
        
        Returns:
            List of overspending insights
        """
        insights = []
        category_stats = self.analyze_spending_by_category()
        
        if category_stats.empty:
            return insights
        
        # Use total spending as proxy for income if income not provided
        reference_amount = self.total_income if self.total_income else self.total_spending
        
        for _, row in category_stats.iterrows():
            category = row['category']
            spent = row['total_spent']
            pct_of_ref = spent / reference_amount if reference_amount > 0 else 0
            
            if category in self.BENCHMARKS:
                benchmark = self.BENCHMARKS[category]
                
                if pct_of_ref > benchmark['warning_pct']:
                    excess_pct = (pct_of_ref - benchmark['ideal_pct']) * 100
                    savings_potential = spent - (reference_amount * benchmark['ideal_pct'])
                    
                    insight = SpendingInsight(
                        category=category,
                        metric_type='overspending',
                        message=f"⚠️ Spending {pct_of_ref*100:.1f}% on {category} (recommended: {benchmark['ideal_pct']*100:.0f}%)",
                        severity='high',
                        recommendation=f"Reduce {category} spending by {excess_pct:.1f}%. Try cooking at home, using public transport, or finding cheaper alternatives.",
                        savings_potential=max(0, savings_potential)
                    )
                    insights.append(insight)
                    
                elif pct_of_ref > benchmark['ideal_pct']:
                    insight = SpendingInsight(
                        category=category,
                        metric_type='moderate_spending',
                        message=f"ℹ️ {category} is at {pct_of_ref*100:.1f}% (slightly above ideal)",
                        severity='medium',
                        recommendation=f"Monitor {category} spending to stay within budget."
                    )
                    insights.append(insight)
        
        return insights
    
    def identify_frequent_small_transactions(self) -> List[SpendingInsight]:
        """
        Identify categories with many small transactions (could indicate wasteful spending).
        
        Returns:
            List of insights about frequent small transactions
        """
        insights = []
        
        if 'category' not in self.df.columns or 'amount' not in self.df.columns:
            return insights
        
        expenses = self.df[self.df['amount'] > 0]
        
        for category in expenses['category'].unique():
            cat_data = expenses[expenses['category'] == category]
            num_txns = len(cat_data)
            avg_amount = cat_data['amount'].mean()
            
            # Flag if many small transactions (e.g., daily coffee, small purchases)
            if num_txns > 10 and avg_amount < 100:
                monthly_cost = cat_data['amount'].sum()
                if monthly_cost > 500:  # Significant monthly cost
                    insight = SpendingInsight(
                        category=category,
                        metric_type='frequent_small',
                        message=f"📝 {num_txns} small transactions in {category} (avg ₹{avg_amount:.0f})",
                        severity='medium',
                        recommendation=f"Consider reducing frequency of small {category} purchases. Monthly total: ₹{monthly_cost:.0f}",
                        savings_potential=monthly_cost * 0.3  # Potential 30% reduction
                    )
                    insights.append(insight)
        
        return insights
    
    def identify_top_merchants(self) -> List[SpendingInsight]:
        """
        Identify top merchants by spending.
        
        Returns:
            List of insights about top merchants
        """
        insights = []
        
        if 'merchant' not in self.df.columns or 'amount' not in self.df.columns:
            return insights
        
        expenses = self.df[self.df['amount'] > 0]
        
        top_merchants = expenses.groupby('merchant')['amount'].agg(['sum', 'count']).reset_index()
        top_merchants = top_merchants[top_merchants['merchant'] != 'unknown']
        top_merchants = top_merchants.sort_values('sum', ascending=False).head(5)
        
        for _, row in top_merchants.iterrows():
            merchant = row['merchant']
            total_spent = row['sum']
            num_visits = row['count']
            
            insight = SpendingInsight(
                category=merchant,
                metric_type='top_merchant',
                message=f"🏪 Top spending at {merchant}: ₹{total_spent:.0f} ({num_visits} visits)",
                severity='low',
                recommendation=f"Review if {merchant} spending aligns with your budget priorities."
            )
            insights.append(insight)
        
        return insights
    
    def identify_subscription_fatigue(self) -> List[SpendingInsight]:
        """
        Identify if subscription spending is too high.
        
        Returns:
            List of subscription-related insights
        """
        insights = []
        
        if 'category' not in self.df.columns or 'amount' not in self.df.columns:
            return insights
        
        subscription_data = self.df[self.df['category'] == 'subscription']
        
        if len(subscription_data) > 0:
            total_subscriptions = subscription_data['amount'].sum()
            sub_count = len(subscription_data)
            
            if sub_count > 5:
                insight = SpendingInsight(
                    category='subscription',
                    metric_type='subscription_fatigue',
                    message=f"📺 {sub_count} subscription services costing ₹{total_subscriptions:.0f}/month",
                    severity='medium',
                    recommendation="Review unused subscriptions. Cancel those you don't use regularly.",
                    savings_potential=total_subscriptions * 0.4  # Can save ~40%
                )
                insights.append(insight)
        
        return insights
    
    def detect_anomalies(self) -> List[SpendingInsight]:
        """
        Detect unusual spending patterns (outliers).
        
        Returns:
            List of anomaly insights
        """
        insights = []
        
        if 'category' not in self.df.columns or 'amount' not in self.df.columns:
            return insights
        
        expenses = self.df[self.df['amount'] > 0]
        
        for category in expenses['category'].unique():
            cat_data = expenses[expenses['category'] == category]['amount']
            
            if len(cat_data) < 3:
                continue
            
            mean = cat_data.mean()
            std = cat_data.std()
            
            if std == 0:
                continue
            
            # Find outliers (> 2.5 std deviations)
            outliers = cat_data[cat_data > (mean + 2.5 * std)]
            
            for amount in outliers:
                insight = SpendingInsight(
                    category=category,
                    metric_type='anomaly',
                    message=f"🔍 Unusually high {category} expense: ₹{amount:.0f} (typical: ₹{mean:.0f})",
                    severity='low',
                    recommendation="Verify if this was a planned expense or if there's a billing error."
                )
                insights.append(insight)
        
        return insights
    
    def generate_insights(self) -> List[SpendingInsight]:
        """
        Generate all spending insights.
        
        Returns:
            List of all insights
        """
        self.insights = []
        
        self.insights.extend(self.identify_overspending())
        self.insights.extend(self.identify_frequent_small_transactions())
        self.insights.extend(self.identify_top_merchants())
        self.insights.extend(self.identify_subscription_fatigue())
        self.insights.extend(self.detect_anomalies())
        
        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        self.insights.sort(key=lambda x: severity_order.get(x.severity, 3))
        
        return self.insights
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for dashboard display.
        
        Returns:
            Dictionary of summary statistics
        """
        if 'amount' not in self.df.columns:
            return {}
        
        expenses = self.df[self.df['amount'] > 0]
        income = self.df[self.df['amount'] < 0]  # Negative amounts are income
        
        return {
            'total_spent': expenses['amount'].sum(),
            'num_transactions': len(expenses),
            'avg_transaction': expenses['amount'].mean() if len(expenses) > 0 else 0,
            'top_category': expenses.groupby('category')['amount'].sum().idxmax() if 'category' in expenses.columns else None,
            'total_income': abs(income['amount'].sum()),
            'net_savings': abs(income['amount'].sum()) - expenses['amount'].sum() if len(income) > 0 else -expenses['amount'].sum(),
            'highest_expense': expenses['amount'].max() if len(expenses) > 0 else 0,
            'most_frequent_category': expenses['category'].mode().iloc[0] if 'category' in expenses.columns and len(expenses) > 0 else None,
        }
    
    def get_category_recommendations(self) -> Dict[str, str]:
        """
        Get specific recommendations for each category.
        
        Returns:
            Dictionary of category recommendations
        """
        return {
            'food': 'Try meal prepping or cooking at home more often. Use discount apps.',
            'transport': 'Consider using public transport, carpooling, or biking for short distances.',
            'shopping': 'Wait 24 hours before making non-essential purchases. Look for sales.',
            'groceries': 'Buy in bulk, use store brands, and plan meals to reduce waste.',
            'bills': 'Negotiate with providers or switch to cheaper plans. Check for unused services.',
            'subscription': 'Cancel unused subscriptions. Share family plans where possible.',
            'fuel': 'Use fuel rewards programs, maintain tire pressure, and plan efficient routes.',
            'other_expenses': 'Track these expenses carefully - they often contain hidden leaks.',
        }


def analyze_spending(df: pd.DataFrame, monthly_income: Optional[float] = None) -> Tuple[List[SpendingInsight], Dict]:
    """
    Convenience function to analyze spending and generate insights.
    
    Args:
        df: DataFrame with transaction data
        monthly_income: Optional monthly income for percentage calculations
        
    Returns:
        Tuple of (insights, summary_stats)
    """
    analyzer = SpendingAnalyzer(df, monthly_income)
    insights = analyzer.generate_insights()
    stats = analyzer.get_summary_stats()
    
    return insights, stats


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[1]))
    
    # Try to load and analyze sample data
    try:
        data_path = Path(__file__).parents[1] / "data" / "financial_transactions_500.csv"
        df = pd.read_csv(data_path)
        
        print("📊 Analyzing spending patterns...")
        print()
        
        insights, stats = analyze_spending(df)
        
        print(f"Total Spent: ₹{stats.get('total_spent', 0):.2f}")
        print(f"Transactions: {stats.get('num_transactions', 0)}")
        print(f"Top Category: {stats.get('top_category', 'N/A')}")
        print()
        
        if insights:
            print("💡 Insights:")
            for insight in insights[:5]:
                print(f"  {insight.message}")
                print(f"    → {insight.recommendation}")
                print()
        else:
            print("✅ No spending issues detected!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

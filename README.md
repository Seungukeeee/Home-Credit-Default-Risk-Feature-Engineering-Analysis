01. Feature Optimization and Analysis

Objective

In real-world credit scoring systems, maintaining a massive number of features increases data management costs and operational complexity. This phase focuses on optimizing model efficiency by identifying a "Lightweight Feature Set" that maintains high predictive power with significantly fewer variables.

Key Methodology

Target Data: Kaggle Home Credit Default Risk Dataset.

Model: LightGBM with 5-fold Cross-Validation.

Optimization Strategy: Feature Importance-based Recursive Elimination to achieve a balance between model complexity and performance (AUC).

Key Results

Feature Reduction: Successfully reduced features from 1,389 to 483 (a 65.2% reduction).

Performance Retention: Maintained 95% of the original model's explanatory power, with the CV AUC only dropping slightly from 0.7923 to 0.7889.

Concentrated Influence: Discovered that the Top 10 features alone account for 31.25% of the total model importance.

Business Impact
By reducing the feature count by over 60% while maintaining nearly identical performance, we can significantly decrease the technical debt of the credit scoring engine and lower the cost of third-party data acquisition without compromising risk management quality.

# Predicting-compressive-strength-of-concrete-mixtures

A machine learning project that uses regression models to predict the compressive strength of high-performance concrete based on mixture composition and curing age.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Status](https://img.shields.io/badge/status-completed-green.svg)

## Project Overview

Concrete compressive strength is a critical quality metric in construction engineering. This project develops predictive models to estimate concrete strength based on ingredient composition, enabling: 
- **Quality control**: Predict strength before physical testing
- **Mix optimization**: Identify optimal ingredient ratios
- **Cost reduction**: Minimize expensive physical testing

## Objectives

1. **Predictive Modeling**: Build regression models that generalize well to unseen concrete mixtures
2. **Feature Importance**: Quantify the relative impact of cement, slag, fly ash, aggregates, and other components
3. **Model Simplification**: Evaluate whether feature selection or regularization improves performance and interpretability

## Dataset

- **Size**: 1,030 concrete samples
- **Features**: 8 material composition variables + curing age
- **Target**:  Compressive strength (MPa - megapascals)

### Features
| Feature | Description | Unit |
|---------|-------------|------|
| Cement | Portland cement content | kg/m^3 |
| Blast Furnace Slag | Industrial byproduct additive | kg/m^3 |
| Fly Ash | Coal combustion byproduct | kg/m^3 |
| Water | Water content | kg/m^3 |
| Superplasticizer | Chemical admixture for workability | kg/m^3 |
| Coarse Aggregate | Gravel/crushed stone | kg/m^3 |
| Fine Aggregate | Sand | kg/m^3 |
| Age | Curing time | days |

**Target Variable**: Concrete compressive strength (MPa)

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical distribution analysis using Shapiro-Wilk tests
- Skewness assessment for all features
- Identification of transformation requirements

### 2. Data Preprocessing

Applied targeted transformations to reduce skewness and improve model performance: 

| Feature | Transformation | Rationale |
|---------|----------------|-----------|
| Cement | Yeo-Johnson | Positive values with moderate skew |
| Blast Furnace Slag | Yeo-Johnson | Moderate skew, strictly positive |
| Fly Ash | Log (log1p) | Contains zeros, strongly right-skewed |
| Water | None | Already symmetric distribution |
| Superplasticizer | Yeo-Johnson | Strictly positive and skewed |
| Coarse Aggregate | None | Symmetric, bell-shaped distribution |
| Fine Aggregate | None | Minor skew, preserves interpretability |
| Age | Yeo-Johnson | Includes zeros, strong skew |

**Target (Compressive Strength)**: No transformation - mild skew is acceptable, preserves interpretability and real-world units (MPa).

### 3. Model Development

Evaluated multiple regression approaches:
- **Baseline**: Dummy Regressor (mean predictor)
- **Linear Regression**: Standard OLS
- **Ridge Regression**: L2 regularization
- **Lasso Regression**: L1 regularization with feature selection
- **ElasticNet**: Combined L1 + L2 regularization

### 4. Model Selection & Tuning
- 10-fold cross-validation for robust performance estimation
- Grid search for hyperparameter optimization (alpha tuning)
- Recursive Feature Elimination with Cross-Validation (RFECV)

## Technical Stack

```python
# Core Libraries
pandas              # Data manipulation
numpy               # Numerical operations
scikit-learn        # Machine learning models and preprocessing
matplotlib          # Visualization
seaborn             # Statistical visualization
scipy               # Statistical tests
```

## Project Structure

```
.
├── data/
│   └── Concrete_Data.xls        # Raw dataset
├── src/
│   └── utils. py                 # Helper functions
├── notebook. ipynb               # Main analysis notebook
└── README.md                    # Project documentation
```

## Key Functions

### Custom Utility Functions

**`fit_model_return_cv_score(model, X_train, y_train, cv=10)`**
- Performs k-fold cross-validation
- Returns mean CV score and fitted model

**`perform_grid_search(model, param_grid, X_train, y_train, cv=10)`**
- Executes grid search for hyperparameter tuning
- Visualizes CV performance across alpha values
- Returns best parameters and estimator

**`rfecv_feature_selection(estimator, X_train, y_train, cv=10)`**
- Recursive feature elimination with cross-validation
- Identifies optimal feature subset
- Returns selector and best R² score

## Results & Findings

*(To be populated with final model performance metrics)*

- **Best Model**: Ridge (Base) 
- **Cross-Validation R^2**: 0.793776
- **Test Set R^2**: 0.816507

## Key Insights

1. **Transformation Impact**: Targeted transformations significantly improved model linearity assumptions without sacrificing interpretability
2. **Feature Engineering**: Domain-specific preprocessing (e.g., log1p for chemical dosages) was crucial for model performance
3. **Regularization**: Ridge Regularization was used without Recursive Feature Selection provided the best performing model

## Future Improvements

- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Explore polynomial features and interaction terms
- [ ] Deploy model as web API using Flask/FastAPI
- [ ] Add confidence intervals for predictions
- [ ] Create interactive dashboard for mix optimization

## References

- Dataset: UCI Machine Learning Repository - Concrete Compressive Strength Data Set
- Yeh, I-Cheng, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)

## Author

**MengenL-ds**
- GitHub: [@MengenL-ds](https://github.com/MengenL-ds)
- Project Link: [https://github.com/MengenL-ds/Predicting-compressive-strength-of-concrete-mixtures](https://github.com/MengenL-ds/Predicting-compressive-strength-of-concrete-mixtures)

## License

This project is available for educational and portfolio purposes. 

---

*Built with Python, scikit-learn, and a data-driven approach to materials science* 🔬

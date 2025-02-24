# House Price Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**TLDR:** An end-to-end machine learning project that predicts house prices using the Ames Housing dataset, with predictions averaging within 11.9% of actual values. Features interactive visualizations and real-time price estimation through a user-friendly web interface.

## Project Overview

This machine learning application predicts house prices based on the Ames Housing dataset, combining robust data analysis with an intuitive user interface. Real estate professionals, homeowners, and data enthusiasts can explore market trends and generate accurate price estimates in seconds.

**Demo:** [View Live Application](https://sjagtani-house-price-prediction.streamlit.app/)

## Key Features

The application offers an interactive data dashboard for exploring housing trends through dynamic visualizations, a smart price predictor that provides accurate estimates with just a few clicks ($40,352 RMSE), feature impact analysis to understand exactly what drives property value in your area, and market comparison to see how properties stack up against similar listings.

## Model Performance

Our carefully engineered Lasso Regression model delivers exceptional results:

| Model | RMSE | Accuracy | Notes |
|-------|------|----------|-------|
| Linear Regression | $41,826 | 88.3% | Good baseline |
| Ridge (α=10) | $40,442 | 88.8% | Handles multicollinearity |
| **Lasso (α=0.001)** | **$40,352** | **88.9%** | **Best performance & feature selection** |

The final model accurately predicts house prices within 11.9% of actual values on average, with key features having the following impact: Overall Quality (+12% per quality point), Ground Living Area (+0.02% per square foot), Garage Cars (+7% per car space), and Year Built (+0.26% per year newer).

## Quick Start

```bash
# Clone repository
git clone https://github.com/sjagtani/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

### Example Usage

```python
from src.predictor import HousePricePredictor

# Load the model
predictor = HousePricePredictor()

# Get a price estimate
estimate = predictor.predict({
    'Overall_Quality': 8,
    'Gr_Liv_Area': 2000,
    'Garage_Cars': 2,
    'Year_Built': 2005,
    'Total_Bsmt_SF': 1000
})

print(f"Estimated house price: ${estimate:.2f}")
```

## Technical Implementation

### Data Analysis & Feature Engineering

The model is built on extensive data analysis of 2,930 properties with 82 features. After correlation analysis and feature engineering, we identified the most influential factors and applied transformations to improve prediction accuracy. This process included log transformation of target variable to address right-skewed distribution, median imputation for missing values, creation of composite features like TotalArea, and regularization to prevent overfitting.

### Project Structure

```
house-price-prediction/
├── data/
│   └── AmesHousing.csv       # Dataset (2,930 properties)
├── notebooks/
│   ├── model_development.ipynb      # Model training & evaluation
│   └── data_visualization.ipynb     # EDA & feature analysis
├── src/
│   ├── app.py                # Streamlit application
│   ├── predictor.py          # Prediction module
│   └── data_processor.py     # Data cleaning & preparation
├── models/
│   └── best_lasso_model.pkl  # Trained model (11.9% MAPE)
├── tests/                    # Unit & integration tests
├── requirements.txt          # Dependencies
├── LICENSE                   # MIT License
└── README.md
```

## Technologies Used

The project leverages Python 3.7+ with Pandas, NumPy, and Scikit-learn for core functionality. Visualizations are created using Plotly, Matplotlib, and Seaborn, while the web application is built with Streamlit and deployed via GitHub Actions and Streamlit Cloud. Testing is handled with Pytest.

## Future Improvements

Future enhancements include implementing gradient boosting for better accuracy, adding geospatial visualization of price trends, displaying prediction uncertainty through confidence intervals, tracking price changes over time with time-series analysis, and suggesting similar houses based on features.

## Contributing

Contributions are welcome! Check out the [issues page](https://github.com/sjagtani/house-price-prediction/issues) for ways to contribute. The process involves forking the repository, creating your feature branch (`git checkout -b feature/amazing-feature`), committing your changes (`git commit -m 'Add some amazing feature'`), pushing to the branch (`git push origin feature/amazing-feature`), and opening a Pull Request.

## License & Acknowledgments

This project is licensed under MIT and uses the Ames Housing dataset from Kaggle. It's based on principles from Andrew Ng's Machine Learning Specialization.

---

Created by Sumesh Jagtani | [Contact](mailto:sumeshjagtani@gmail.com)

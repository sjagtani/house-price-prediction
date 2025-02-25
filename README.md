# House Price Prediction

![Python](https://img.shields.io/badge/Python-Latest-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**TLDR:** An end-to-end machine learning project that predicts house prices using the Ames Housing dataset, with predictions averaging within 11.9% of actual values. Features interactive visualizations and real-time price estimation through a user-friendly web interface.

## Project Overview

This machine learning application predicts house prices based on the Ames Housing dataset, combining robust data analysis with an intuitive user interface. After finishing Andrew Ng's Machine Learning specialization, I was eager to apply what I learned and put the concepts into practice. Hence, I started working on this project to gain hands-on experience and deepen my understanding.

**Demo:** [View Live Application](https://sjagtani-house-price-prediction.streamlit.app/)

## Key Features

The application offers an interactive data dashboard for exploring housing trends through dynamic visualizations and a price predictor that provides estimates with just a few clicks ($40,352 RMSE). Users can also view model performance metrics and understand how different features impact house prices.

## Model Performance

| Model | RMSE | R² Score | Notes |
|-------|------|----------|-------|
| Linear Regression | $41,826 | 0.782 | Good baseline |
| Ridge (α=10) | $40,442 | - | Handles multicollinearity |
| **Lasso (α=0.001)** | **$40,352** | **0.744** | **Best performance & feature selection** |

The final model accurately predicts house prices within 11.9% of actual values on average (MAPE), with key features having the following impact: Overall Quality (+12% per quality point), Ground Living Area (+0.02% per square foot), Garage Cars (+7% per car space), and Year Built (+0.26% per year newer).

## Quick Start

```bash
# Clone repository
git clone https://github.com/sjagtani/house-price-prediction.git
cd house-price-prediction/house_price_dashboard

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

### Example Usage

```python
# Load and prepare data
import os
import pandas as pd
import numpy as np
import pickle

# Setup paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'best_lasso_model.pkl')
CSV_PATH = os.path.join(BASE_PATH, 'AmesHousing.csv')

# Load the model
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Get a price estimate
house_features = {
    'Overall Qual': 8,
    'Gr Liv Area': 2000,
    'Garage Cars': 2,
    'Year Built': 2005,
    'Total Bsmt SF': 1000,
    'Lot Area': 8000,
    'Full Bath': 2
}

# Prepare features
input_data = pd.DataFrame([house_features])

# Add engineered features
input_data['TotalArea'] = input_data['Gr Liv Area'] + input_data['Total Bsmt SF']
input_data['LogLotArea'] = np.log(input_data['Lot Area'] + 1)

# Make prediction
y_pred_log = model.predict(input_data)
estimate = np.exp(y_pred_log)[0] - 1
print(f"Estimated house price: ${estimate:,.2f}")
```

## Technical Implementation

### Data Analysis & Feature Engineering

The model is built on extensive data analysis of 2,930 properties with 82 features. My preprocessing pipeline includes log transformation of the target variable to address right-skewed distribution, median imputation for missing values, and strategic feature engineering.

I created composite features like TotalArea (combining Ground Living Area and Total Basement SF) and LogLotArea, while using Lasso regularization for effective feature selection, resulting in a robust model that captures key property value drivers.

### Project Structure

```
house-price-prediction/
├── data/
│   └── AmesHousing.csv
├── notebooks/
│   ├── model_development.ipynb
│   └── data_visualization.ipynb
├── src/
│   ├── app_copy.py
├── models/
│   └── best_lasso_model.pkl    # Trained model (11.9% MAPE)
├── requirements.txt            # Dependencies
├── README.md
└── house_price_dashboard/
    ├── AmesHousing.csv
    ├── app.py                  # Streamlit application
    ├── best_lasso_model.pkl    # Trained model (11.9% MAPE)
    └── requirements.txt        # Dependencies
```

## Technologies Used

The project leverages Python with Pandas, NumPy, and Scikit-learn for core functionality. Interactive visualizations are created using Plotly, and the web application is built with Streamlit and deployed via Streamlit Cloud.

## Future Improvements

Future improvements to the project will focus on enhancing both the technical performance and the overall user experience. I plan to experiment with ensemble methods such as Random Forest and XGBoost, incorporate advanced feature engineering, and introduce cross-validation for hyperparameter tuning to boost model performance. 

Ideally, I'll get around to displaying confidence intervals for predictions and showing feature importance visualizations. I would also love to improve the visual aesthetic and the overall UI, leading to a more fun and intuitive experience.

## License & Acknowledgments

This project is licensed under MIT and uses the Ames Housing dataset from Kaggle. It's based on principles from Andrew Ng's Machine Learning Specialization.

---

Created by Sumesh Jagtani | [Contact](mailto:sumeshjagtani@gmail.com)

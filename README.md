# House Price Prediction Dashboard

An interactive web application for predicting house prices using machine learning, built with Python, Scikit-learn, Plotly, and Streamlit.

## Project Overview

This project implements a machine learning model to predict house prices based on the Ames Housing dataset. The application includes:
- Interactive data visualizations
- Real-time price predictions
- Model performance metrics
- Feature importance analysis

## Features

- **Data Analysis Dashboard:**
  - House price distribution visualization
  - Feature correlation heatmap
  - Interactive scatter plots
  - Price vs Living Area analysis

- **Price Prediction:**
  - Real-time price predictions
  - User-friendly input interface
  - Key feature selection
  - Instant feedback

- **Model Performance:**
  - Predicted vs Actual price comparisons
  - Model accuracy metrics (R², RMSE, MAPE)
  - Performance visualization

## Technologies Used

- Python 3.x
- Pandas & NumPy for data processing
- Scikit-learn for machine learning
- Plotly for interactive visualizations
- Streamlit for web interface
- Pickle for model serialization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Model Details

- **Algorithm:** Lasso Regression
- **Features Used:**
  - Overall Quality
  - Living Area
  - Garage Capacity
  - Year Built
  - Basement Area
  - Lot Area
  - Full Bathrooms

- **Performance Metrics:**
  - R² Score: 0.744
  - RMSE: $41,340.23
  - MAPE: 11.9%

## Project Structure

```
house-price-prediction/
├── data/
│   └── AmesHousing.csv
├── notebooks/
│   ├── model_development.ipynb
│   └── visualizations.ipynb
├── src/
│   └── app.py
├── models/
│   └── best_lasso_model.pkl
├── requirements.txt
└── README.md
```

## Usage

1. Navigate to the Data Analysis page to explore housing data trends
2. Use the Price Prediction page to estimate house prices
3. Check the Model Performance page for accuracy metrics

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- Dataset: Ames Housing dataset
- Inspired by the Machine Learning Specialization by Andrew Ng

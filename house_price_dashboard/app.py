import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

# Determine absolute path
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_PATH, 'AmesHousing.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'best_lasso_model.pkl')

# Load model
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load & cache data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    st.title('House Price Prediction Dashboard')
    
    # Load data & model
    df = load_data()
    model = load_model()
    
    if model is None:
        st.error("Please ensure 'best_lasso_model.pkl' is in the same directory as this app.")
        return
    
    # Sidebar navigation
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Select Page', 
                           ['Data Analysis', 
                            'Price Prediction', 
                            'Model Performance'])
    
    if page == 'Data Analysis':
        st.header('Data Analysis')
        
        # Price distribution
        st.subheader('House Price Distributions')
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Sale Price Distribution', 
                                         'Log Sale Price Distribution'))
        
        fig.add_trace(
            go.Histogram(x=df['SalePrice'], name='Sale Price',
                        nbinsx=50),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=np.log(df['SalePrice']), name='Log Sale Price',
                        nbinsx=50),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig)
        
        # Correlation heatmap
        st.subheader('Feature Correlations')
        features = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 
                   'Garage Cars', 'Total Bsmt SF', 'Year Built', 'Lot Area',
                   'Full Bath']
        
        corr_matrix = df[features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=features,
            y=features,
            colorscale='Viridis',
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}'))
        
        fig.update_layout(height=600)
        st.plotly_chart(fig)
        
        # Scatter plot
        st.subheader('Price vs Living Area')
        fig = px.scatter(df, x='Gr Liv Area', y='SalePrice',
                        color='Overall Qual',
                        title='House Prices by Living Area and Quality')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig)
        
    elif page == 'Price Prediction':
        st.header('House Price Prediction')
        
        # Input
        col1, col2 = st.columns(2)
        
        with col1:
            overall_qual = st.slider('Overall Quality', 1, 10, 5)
            gr_liv_area = st.number_input('Living Area (sqft)', 500, 5000, 1500)
            garage_cars = st.slider('Garage Capacity (cars)', 0, 4, 2)
            full_bath = st.number_input('Full Bathrooms', 0, 6, 2)
            
        with col2:
            year_built = st.number_input('Year Built', 1900, 2023, 1990)
            total_bsmt = st.number_input('Basement Area (sqft)', 0, 3000, 1000)
            lot_area = st.number_input('Lot Area (sqft)', 1000, 20000, 8000)
            
        if st.button('Predict Price'):
            try:
                # Input data
                input_data = pd.DataFrame({
                    'Overall Qual': [overall_qual],
                    'Gr Liv Area': [gr_liv_area],
                    'Garage Cars': [garage_cars],
                    'Total Bsmt SF': [total_bsmt],
                    'Year Built': [year_built],
                    'Lot Area': [lot_area],
                    'Full Bath': [full_bath]
                })
                
                # Engineered features
                input_data['TotalArea'] = input_data['Gr Liv Area'] + input_data['Total Bsmt SF']
                input_data['LogLotArea'] = np.log(input_data['Lot Area'] + 1)
                
                # Make prediction
                y_pred_log = model.predict(input_data)
                prediction = np.exp(y_pred_log) - 1
                
                st.success(f'Estimated House Price: ${prediction[0]:,.2f}')
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
            
    else:  # Model performance
        st.header('Model Performance')
        
        # Prep features with engineering
        feature_cols = ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 
                       'Total Bsmt SF', 'Year Built', 'Lot Area', 'Full Bath']
        
        X = df[feature_cols].copy()
        X['TotalArea'] = X['Gr Liv Area'] + X['Total Bsmt SF']
        X['LogLotArea'] = np.log(X['Lot Area'] + 1)
        X = X.fillna(X.mean())
        y = df['SalePrice']
        
        # Get predictions
        y_pred_log = model.predict(X)
        y_pred = np.exp(y_pred_log) - 1
        
        # Model performance plot
        st.subheader('Predicted vs Actual Prices')
        fig = go.Figure()
        
        # Scatter plot of predictions
        fig.add_trace(go.Scatter(
            x=y,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        max_val = max(max(y), max(y_pred))
        min_val = min(min(y), min(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            xaxis_title='Actual Price',
            yaxis_title='Predicted Price',
            height=500
        )
        
        st.plotly_chart(fig)
        
        # Calculate metrics
        r2 = np.corrcoef(y, y_pred)[0, 1]**2
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('R² Score', f"{r2:.3f}")
        with col2:
            st.metric('RMSE', f"${rmse:,.2f}")
        with col3:
            st.metric('MAPE', f"{mape:.1f}%")

if __name__ == '__main__':
    main()

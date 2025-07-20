# app.py
# This is the main script for the Streamlit web application.

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris

# --- Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model and Data Loading ---
@st.cache_resource
def load_model():
    """Load the pre-trained logistic regression model."""
    try:
        model = joblib.load('iris_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run `train_model.py` first.")
        return None

@st.cache_data
def load_iris_data():
    """Load the original Iris dataset for visualization."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df

model = load_model()
iris_df = load_iris_data()

# --- Sidebar for User Input ---
st.sidebar.header("Input Features")
st.sidebar.markdown("Use the sliders to provide the flower's measurements.")

def get_user_input():
    """Get input from the user via sidebar sliders."""
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

user_input_df = get_user_input()

# --- Main Panel ---
st.title("🌸 Iris Flower Species Predictor")
st.markdown(
    "This application uses a machine learning model to predict the species of an Iris flower "
    "based on its sepal and petal measurements. The sidebar on the left allows you to input custom values."
)

# Display user input
st.subheader("Your Input Measurements")
st.table(user_input_df)

if model is not None:
    # --- Prediction Logic ---
    prediction = model.predict(user_input_df)
    prediction_proba = model.predict_proba(user_input_df)
    
    st.subheader("Prediction")
    species_prediction = prediction[0]
    
    # Display prediction with some styling
    if species_prediction == 'setosa':
        st.success(f"The model predicts the species is **Setosa** 🌿")
    elif species_prediction == 'versicolor':
        st.info(f"The model predicts the species is **Versicolor** 🌷")
    else:
        st.warning(f"The model predicts the species is **Virginica** 🌺")

    # --- Visualization Section ---
    st.header("Understanding the Prediction")
    
    col1, col2 = st.columns(2)

    with col1:
        # 1. Prediction Probabilities Chart
        st.subheader("Prediction Probabilities")
        
        prob_df = pd.DataFrame({
            'Species': model.classes_,
            'Probability': prediction_proba[0]
        }).sort_values(by='Probability', ascending=False)
        
        fig_prob = px.bar(
            prob_df,
            x='Probability',
            y='Species',
            orientation='h',
            text='Probability',
            color='Species',
            color_discrete_map={
                'setosa': '#2ca02c', 
                'versicolor': '#1f77b4', 
                'virginica': '#ff7f0e'
            }
        )
        fig_prob.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig_prob.update_layout(
            uniformtext_minsize=8, 
            uniformtext_mode='hide',
            yaxis_title="",
            xaxis_title="Probability",
            legend_title_text='Species'
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col2:
        # 2. Feature Comparison Scatter Plot
        st.subheader("How Your Input Compares")
        
        x_axis = st.selectbox("X-axis Feature", iris_df.columns[:-1], index=0)
        y_axis = st.selectbox("Y-axis Feature", iris_df.columns[:-1], index=1)

        fig_scatter = px.scatter(
            iris_df, 
            x=x_axis, 
            y=y_axis, 
            color='species',
            title=f'Scatter Plot of {y_axis} vs. {x_axis}',
            color_discrete_map={
                'setosa': '#2ca02c', 
                'versicolor': '#1f77b4', 
                'virginica': '#ff7f0e'
            }
        )
        
        # Add a marker for the user's input
        fig_scatter.add_trace(go.Scatter(
            x=user_input_df[x_axis],
            y=user_input_df[y_axis],
            mode='markers',
            marker=dict(color='red', size=15, symbol='star'),
            name='Your Input'
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.info("Please wait for the model to be loaded or check the error message above.")

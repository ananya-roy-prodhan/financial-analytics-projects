import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


st.title("📊 Credit Risk Prediction App")

st.markdown("***From age to income — See how borrower features shape default risk in real time!***")

#log_reg_model = joblib.load("log_reg_model.pkl")
model_dict = joblib.load("logistic_model_with_feature_names.pkl")
log_reg_model = model_dict['model']
features_model = model_dict['feature_names']

df = joblib.load("df_no_outlier.pkl") 
#df = model_dict['data_cleaned']
feature_names = list(df.columns)
scaler        = joblib.load("scaler.pkl")


if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
    st.error("No valid training data ('X_train_full') found in the file.")
    st.stop()
available_features = [col for col in feature_names if col in df.columns]
if not available_features:
    st.error("No matching columns found between model and loaded data.")
    st.stop()
    
    
tab1, tab2 = st.tabs(["Visualization", "Prediction"])


with tab1:
    st.header("Visualization")
    st.write("*This tab is to see distributions of features*")
    numeric_col = {'person_income', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income'}
    
    # Function for numerical variable plots
    def plot_numerical(df, column):
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax, color="#1C39BB")
        #ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# Function for categorical variable plots
    def plot_categorical(df, column):
        fig, ax = plt.subplots()
        sns.countplot(x=df[column], ax=ax, palette="bright")
        #ax.set_title(f"Bar Chart of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        #plt.xticks(rotation=45)
        st.pyplot(fig)

# Example Streamlit app
    st.header("Distributions of Hitorical Loan Data")

# Restrict selection to your training columns
    column = st.selectbox("Select feature", feature_names)

# Decide automatically: treat dummy variables as categorical
    if column in numeric_col or df[column].nunique() > 10:
        st.subheader(f"Numerical Distribution – {column}")
        plot_numerical(df, column)
    else:
        st.subheader(f"Categorical Distribution – {column}")
        plot_categorical(df, column)

   
with tab2:
    st.header("Prediction")
    st.write("*This is the tab where you can run your model predictions*")

    st.sidebar.header("Borrower Demographics")

# --- Input Widgets ---

    income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    employment_length = st.sidebar.slider("Employment Length (years)", 0, 40, 5)
    loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=1000, value=10000, step=500)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=12.0, step=0.1)
    home_ownership = st.sidebar.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])
    loan_intent = st.sidebar.selectbox("Loan Intent", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
    default_on_file = st.sidebar.selectbox("Default on File", ["Yes", "No"])


# --- Prediction Button ---
    
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: teal;
            color: white;
            font-weight: bold;
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.sidebar.button("Predict Default Risk"):
        with st.spinner("Making prediction..."):
            
            input_data = {col: 0 for col in features_model}
            
            input_data['person_income'] = income
            input_data['person_emp_length'] = employment_length
            input_data['loan_amnt'] = loan_amount
            input_data['loan_int_rate'] = interest_rate
            input_data['loan_percent_income'] = loan_amount / income if income > 0 else 0
            #input_data['CreditHistoryLength'] = credit_history_length
        
        # Home ownership (Mortgage is baseline → all zeros)

            if home_ownership.upper() == "OTHER":
                input_data['person_home_ownership_OTHER'] = 1
            elif home_ownership.upper() == "OWN":
                input_data['person_home_ownership_OWN'] = 1
            elif home_ownership.upper() == "RENT":
                input_data['person_home_ownership_RENT'] = 1

            if loan_intent.upper() != "DEBTCONSOLIDATION":
                input_data[f'loan_intent_{loan_intent.upper()}'] = 1

            if default_on_file == "Yes":
                input_data['cb_person_default_on_file_Y'] = 1
            

            features_df = pd.DataFrame([input_data], columns=features_model)
            
            features_scaled = scaler.transform(features_df)
            #features_scaled = scaler.fit_transform(features_df)
        
       
# Predict Probability
            y_pred_proba = log_reg_model.predict_proba(features_scaled)[0,1]
     
     
        # ── Risk level buckets ──
            if y_pred_proba < 0.30:
                risk_level = "Low Risk"
                bg_color   = "#e6ffe6"       # very light green
                text_color = "#006400"       # dark green
                border_color = "#b3ffb3"

            # elif y_pred_proba < 0.30:
                # risk_level = "Low Risk"
                # bg_color   = "#f0fff0"
                # text_color = "#006400"
                # border_color = "#99ff99"

            elif y_pred_proba < 0.50:
                risk_level = "Moderate Risk"
                bg_color   = "#fffbe6"       # very light yellow/amber
                text_color = "#664d03"
                border_color = "#ffe066"

            elif y_pred_proba < 0.75:
                risk_level = "High Risk"
                bg_color   = "#fff5e6"       # orange
                text_color = "#cc5500"
                border_color = "#ffb347"

            else:
                risk_level = "Very High Risk – Decline Recommended"
                bg_color   = "#ffebee"
                text_color = "#b71c1c"       # strong dark red
                border_color = "#ef5350"

        # Show probability
            st.subheader("Prediction Results")
            #st.metric("Probability of Default", f"{y_pred_proba*100:.2f}%")

        
        # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = y_pred_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Probability", 'font': {'color': 'darkslategray'}},
                number = {'suffix': "%", 'font': {'color': 'black'}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 50], 'color': "yellow"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': y_pred_proba*100}}))
            st.plotly_chart(fig, use_container_width=True)
            
            # Full-width colored result box (no emoji)
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    padding: 20px;
                    border-radius: 12px;
                    margin: 16px 0;
                    font-size: 25px;
                    font-weight: 700;
                    text-align: center;
                    border: 2px solid {border_color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                ">
                    {risk_level}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.write("Features row being passed to model:")
            st.write(features_df)
        
       



 






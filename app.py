# app.py (Updated with CSV Upload & Feature Engineering Model Selection)
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
import glob
import io # Needed for reading uploaded file buffer

# --- Configuration ---
API_URL = "http://127.0.0.1:5000/predict"
DATA_FILE = 'exoTrain.csv'
EXPECTED_RAW_FEATURES = 3197 # !! Adjust if needed !!
SAMPLE_TEST_DATA_FILE = 'test_flux_data.csv'
AVAILABLE_MODELS = ['xgb_model.pkl', 'svm_model.pkl', 'rf_model.pkl'] # List available model files

# --- Helper Functions ---
@st.cache_data
def load_sample_data(filename, nrows=100):
    try:
        df = pd.read_csv(filename, nrows=nrows)
        return df
    except FileNotFoundError:
        st.warning(f"Data file ({filename}) not found. Cannot display sample data.")
        return None
    except Exception as e:
        st.error(f"Error loading sample data from {filename}: {e}")
        return None

def plot_flux(flux_data, title="Flux Variation Over Time"):
    if not isinstance(flux_data, (list, np.ndarray)) or len(flux_data) == 0:
        st.warning("Invalid data provided for plotting.")
        return None
    time_steps = list(range(1, len(flux_data) + 1))
    fig = px.line(x=time_steps, y=flux_data, labels={'x': 'Time Step', 'y': 'Flux'}, title=title)
    fig.update_layout(template="plotly_white")
    return fig

# --- Streamlit App Layout ---
st.set_page_config(page_title="Exoplanet Detector FE", layout="wide")

st.title("🌟 Exoplanet Host Star Detector (Feature Engineering) 🌟")
st.markdown("Predict using engineered features derived from raw flux.")

# --- Sidebar ---
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox("Choose Mode", ["🚀 Predict New Star", "📊 Explore Raw Training Data", "ℹ️ About"])

# --- Prediction Mode ---
if app_mode == "🚀 Predict New Star":
    st.header("🔭 Predict Exoplanet Presence")
    st.markdown(f"""
        Provide the **raw flux values** for a star using one of the methods below.
        The system expects **{EXPECTED_RAW_FEATURES}** distinct flux measurements per star.
    """)

    # --- Model Selection in Sidebar ---
    selected_model = st.sidebar.selectbox("Select Model", AVAILABLE_MODELS)

    st.subheader("Input Method 1: Upload CSV File")
    st.markdown(f"""
        Upload a CSV file containing **only the raw flux values**.
        - Each row should represent one star.
        - The file should **not** have a header row.
        - Each row must contain exactly **{EXPECTED_RAW_FEATURES}** columns (flux values).
        *(If multiple rows are present, only the first row will be used for prediction)*
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    st.markdown("---") # Separator

    st.subheader("Input Method 2: Paste Data")
    st.markdown("Alternatively, paste the comma-separated raw flux values below:")
    # Disable text area if a file is uploaded to avoid confusion
    flux_input_text = st.text_area(
        "Paste Raw Flux Data (comma-separated):",
        height=100,
        placeholder=f"e.g., 15.3, -4.6, 20.1, ... ({EXPECTED_RAW_FEATURES} values)",
        disabled=(uploaded_file is not None) # Disable if file is uploaded
    )

    # Provide sample data download button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sample Data")
    try:
        # Use the content from download_test.py output
        with open(SAMPLE_TEST_DATA_FILE, "r") as f:
            st.sidebar.download_button(
                label="📥 Download Sample Raw CSV",
                data=f.read(),
                file_name=SAMPLE_TEST_DATA_FILE,
                mime="text/csv",
            )
            st.sidebar.markdown(f"Use this file with 'Upload CSV File' or copy a row to paste.")
    except FileNotFoundError:
        st.sidebar.warning(f"`{SAMPLE_TEST_DATA_FILE}` not found. Run `download_test.py`.")


    if st.button("Predict", key="predict_button"):
        raw_flux_values_to_predict = None
        source = None

        # --- Prioritize uploaded file ---
        if uploaded_file is not None:
            source = f"Uploaded File: `{uploaded_file.name}`"
            try:
                # Read the CSV - assume no header
                df_upload = pd.read_csv(uploaded_file, header=None)

                if df_upload.empty:
                    st.error("The uploaded CSV file is empty.")
                else:
                    # Check number of columns in the first row
                    num_cols = df_upload.shape[1]
                    if num_cols != EXPECTED_RAW_FEATURES:
                        st.error(f"Uploaded CSV has {num_cols} columns, but expected {EXPECTED_RAW_FEATURES}. Please check the file format.")
                    else:
                        # Get the first row as a numpy array, then list of floats
                        raw_flux_values_to_predict = df_upload.iloc[0].values.astype(float).tolist()
                        st.info(f"Using data from the first row of the uploaded file: `{uploaded_file.name}`")

            except pd.errors.EmptyDataError:
                st.error("The uploaded CSV file appears to be empty or improperly formatted.")
            except Exception as e:
                st.error(f"Error reading or processing CSV file: {e}")

        # --- Use text area if no file was uploaded (and text area not empty) ---
        elif flux_input_text:
            source = "Pasted Text Area"
            try:
                flux_values_str = flux_input_text.strip().split(',')
                raw_flux_values_list = [float(v.strip()) for v in flux_values_str if v.strip()]

                if len(raw_flux_values_list) != EXPECTED_RAW_FEATURES:
                    st.error(f"Incorrect number of features in pasted data. Expected {EXPECTED_RAW_FEATURES}, but got {len(raw_flux_values_list)}.")
                else:
                    raw_flux_values_to_predict = raw_flux_values_list

            except ValueError:
                st.error("Invalid input in text area: Please ensure all values are numbers separated by commas.")
            except Exception as e:
                st.error(f"Error processing pasted data: {e}")

        # --- No input provided ---
        else:
            st.warning("Please either upload a CSV file or paste flux data into the text area.")

        # --- Perform Prediction if data is ready ---
        if raw_flux_values_to_predict is not None:
            st.write(f"**Source of data for prediction:** {source}")
            # Prepare data for API, including the selected model
            payload = {'flux_data': raw_flux_values_to_predict, 'model_name': selected_model}

            # Send request to Flask API
            with st.spinner('Sending data, engineering features, and predicting...'):
                try:
                    response = requests.post(API_URL, json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()

                    st.subheader("Prediction Result:")
                    prediction = result.get('prediction')
                    label = result.get('class_label', 'N/A')
                    confidence = result.get('confidence', 0)
                    model_used = result.get('model_used', 'N/A')

                    st.write(f"(Predicted using model: `{model_used}`)")

                    if prediction == 1: # Planet
                        st.success(f"**Prediction: {label} (Class {prediction})**")
                        st.progress(confidence)
                        st.metric(label="Confidence", value=f"{confidence:.2%}")
                    elif prediction == 0: # No Planet
                        st.info(f"**Prediction: {label} (Class {prediction})**")
                        st.progress(confidence if label=="No Planet" else 1-confidence)
                        st.metric(label="Confidence", value=f"{confidence if label=='No Planet' else 1-confidence:.2%}")
                    else:
                        st.warning(f"Received unexpected prediction: {prediction}")

                    # Visualize the INPUT raw flux data used for prediction
                    st.subheader("Input Raw Flux Data Visualization")
                    fig = plot_flux(raw_flux_values_to_predict, title="Input Star Raw Flux Variation (Used for Prediction)")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                except requests.exceptions.ConnectionError:
                    st.error(f"Connection Error: Could not connect to the API at {API_URL}. Is the Flask API (`Api.py`) running?")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The API might be busy or unresponsive.")
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")
                    try:
                        error_detail = response.json().get('error', 'No details provided.')
                        st.error(f"API Error Message: {error_detail}")
                    except: pass

# --- Data Exploration Mode ---
elif app_mode == "📊 Explore Raw Training Data":
    st.header(" explore raw data")
    st.markdown(f"Visualizing raw flux samples from `{DATA_FILE}`.")

    df_sample = load_sample_data(DATA_FILE, nrows=200)

    if df_sample is not None:
        st.subheader("Sample Raw Training Data")
        st.dataframe(df_sample.head())

        st.subheader("Label Distribution (in Sample)")
        df_sample['MAPPED_LABEL'] = df_sample['LABEL'].map({1: 0, 2: 1})
        label_counts = df_sample['MAPPED_LABEL'].value_counts().reset_index()
        label_counts.columns = ['LABEL', 'Count']
        label_counts['LABEL'] = label_counts['LABEL'].map({0: 'No Planet (0)', 1: 'Planet (1)'})
        fig_dist = px.bar(label_counts, x='LABEL', y='Count', title="Distribution of Stars in Sampled Training Data", text_auto=True)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Example Raw Flux Curves from Training Data")
        hosts = df_sample[df_sample['LABEL'] == 2]
        non_hosts = df_sample[df_sample['LABEL'] == 1]

        if not hosts.empty:
            # Ensure column index range is correct, exclude LABEL and MAPPED_LABEL
            flux_cols = df_sample.columns.drop(['LABEL', 'MAPPED_LABEL'])
            host_example_flux = hosts.iloc[0][flux_cols].values
            fig_host = plot_flux(host_example_flux, title="Example Raw Flux Curve: Planet (Original Label 2)")
            if fig_host: st.plotly_chart(fig_host, use_container_width=True)
        else: st.write("No Planet examples (Original Label 2) found in this sample.")

        if not non_hosts.empty:
            flux_cols = df_sample.columns.drop(['LABEL', 'MAPPED_LABEL'])
            non_host_example_flux = non_hosts.iloc[0][flux_cols].values
            fig_non_host = plot_flux(non_host_example_flux, title="Example Raw Flux Curve: No Planet (Original Label 1)")
            if fig_non_host: st.plotly_chart(fig_non_host, use_container_width=True)
        else: st.write("No No Planet examples (Original Label 1) found in this sample.")

        st.markdown("---")
        st.markdown("**Note:** The training script evaluates models but doesn't save report files for display here.")

import gradio as gr
import joblib
import numpy as np

# Load trained model
model = joblib.load("diabetes_model.pkl")


def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "The person is Diabetic."
    else:
        return "The person is Not Diabetic."


# Create Gradio interface
diabetes_interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction System",
    description="Predict whether a person is diabetic or not.",
)

# Launch app
diabetes_interface.launch(
    server_name="0.0.0.0",
    server_port=7860
)

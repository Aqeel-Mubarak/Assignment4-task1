# app.py
import joblib
import gradio as gr
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load("iris_model.pkl")
iris = load_iris()

def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Make prediction
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)
    return iris.target_names[prediction[0]]

# Define the Gradio interface
inputs = [
    gr.Number(label="Sepal Length"),
    gr.Number(label="Sepal Width"),
    gr.Number(label="Petal Length"),
    gr.Number(label="Petal Width"),
]
outputs = gr.Textbox(label="Predicted Species")

app = gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title="Iris Flower Predictor")

if __name__ == "__main__":
    app.launch()

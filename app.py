import spaces  # must be imported before torch/transformers on HF ZeroGPU Spaces
import gradio as gr
from transformers import pipeline

MODEL_NAME = "davidit17/e5-grocery-finetuned-v2"
CONFIDENCE_THRESHOLD = 0.5

# Load the classifier once at startup
classifier = pipeline("text-classification", model=MODEL_NAME)


@spaces.GPU
def classify(text: str):
    if not text or not text.strip():
        return "Please enter a grocery item."

    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if score < CONFIDENCE_THRESHOLD:
        return f"אחר (low confidence: {score:.2f}, raw prediction: {label})"

    return f"{label} ({score:.2f})"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Grocery item (Hebrew)", placeholder="e.g. גבינה צהובה"),
    outputs=gr.Textbox(label="Category"),
    title="Hebrew Grocery Item Classifier",
    description="Classifies a Hebrew grocery shopping list item into one of 12 supermarket categories.",
    examples=["גבינה צהובה", "עגבניות שרי", "שמפו", "קוקה קולה", "חזה עוף"],
)

if __name__ == "__main__":
    demo.launch()
import time
import spaces  
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForFeatureExtraction

MODEL_NAME = "davidit17/e5-grocery-finetuned-v2"
BASE_TOKENIZER_NAME = "intfloat/multilingual-e5-small"

tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME)

# Lazily-populated cache so we only download/load a variant the first time it's used
_model_cache = {}

EXAMPLE_ITEMS = [
    "גבינה צהובה", "עגבניות שרי", "שמפו", "קוקה קולה", "חזה עוף",
    "לחם אחיד", "חלב 3%", "ביצים L", "תפוחי אדמה", "בננות",
    "אורז בסמטי", "פסטה פנה", "שמן זית", "קפה נמס", "תה ירוק",
    "נייר טואלט", "אבקת כביסה", "סבון כלים", "יוגורט תות", "טחינה גולמית",
]


THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
    # 1. Clean, modern system fonts
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    # 2. Layout & Backgrounds (Subtle nesting for visual depth)
    body_background_fill="#f9fafb",        # Soft light gray background
    body_background_fill_dark="#0b0f19",   # Deep rich dark background
    
    background_fill_primary="#ffffff",     # App containers remain crisp white
    background_fill_primary_dark="#111827",
    
    background_fill_secondary="#f3f4f6",   # Slightly darker for inner wells
    background_fill_secondary_dark="#1f2937",
    
    # 3. Card/Block Styling
    block_background_fill="#ffffff",
    block_background_fill_dark="#111827",
    block_border_width="1px",
    block_border_color="#e5e7eb",
    block_border_color_dark="#374151",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.05)", # Tiny shadow makes blocks "pop"
    block_radius="12px",                   # Slightly softer corners
    
    # 4. Form Inputs
    input_background_fill="#ffffff",
    input_background_fill_dark="#1f2937",
    input_radius="8px",
    border_color_primary="#d1d5db",
    border_color_primary_dark="#4b5563",
    
    # 5. Primary Action Button
    button_primary_background_fill="#2563eb",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_dark="#3b82f6",
    button_primary_background_fill_hover_dark="#2563eb",
    
    # 6. Smooth Transitions
    link_text_color="#2563eb",
    link_text_color_dark="#60a5fa"
)
CSS = """
#items-textbox {
    max-width: 500px;
}
.results-table {
    max-width: 400px;
}
"""


def parse_items(items_text):
    return [line.strip() for line in items_text.splitlines() if line.strip()]


def get_pytorch_classifier():
    if "pytorch" not in _model_cache:
        _model_cache["pytorch"] = pipeline("text-classification", model=MODEL_NAME, tokenizer=tokenizer)
    return _model_cache["pytorch"]


def get_onnx_model(subfolder):
    key = f"onnx:{subfolder}"
    if key not in _model_cache:
        _model_cache[key] = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, subfolder=subfolder)
    return _model_cache[key]


def classify_pytorch(items):
    clf = get_pytorch_classifier()
    predictions = clf(items)
    return [[item, pred["label"], f"{pred['score']:.2f}"] for item, pred in zip(items, predictions)]


def classify_onnx(items, subfolder):
    model = get_onnx_model(subfolder)

    inputs = tokenizer(items, return_tensors="np", padding=True, truncation=True)
    input_names = [i.name for i in model.model.get_inputs()]
    if "token_type_ids" in input_names and "token_type_ids" not in inputs:
        inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
    onnx_inputs = {k: v for k, v in inputs.items() if k in input_names}

    logits = model.model.run(None, onnx_inputs)[0]
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    pred_ids = np.argmax(probs, axis=-1)

    id2label = model.config.id2label
    results = []
    for i, item in enumerate(items):
        label = id2label[int(pred_ids[i])]
        score = float(probs[i][pred_ids[i]])
        results.append([item, label, f"{score:.2f}"])
    return results


def _run(items_text, classify_fn):
    items = parse_items(items_text)
    if not items:
        return [], ""
    start = time.perf_counter()
    results = classify_fn(items)
    elapsed = time.perf_counter() - start
    return results, f"{elapsed:.2f}s"


@spaces.GPU
def run_pytorch(items_text):
    return _run(items_text, classify_pytorch)


@spaces.GPU
def run_onnx_fp32(items_text):
    return _run(items_text, lambda items: classify_onnx(items, "onnx"))


@spaces.GPU
def run_onnx_int8(items_text):
    return _run(items_text, lambda items: classify_onnx(items, "onnx-int8"))


def make_model_column(label, run_fn):
    with gr.Column():
        btn = gr.Button(f"Classify — {label}", variant="primary")
        time_box = gr.Textbox(label="Time", interactive=False)
        results = gr.Dataframe(
            headers=["Item", "Category", "Confidence"],
            datatype=["str", "str", "str"],
            interactive=False,
            label=f"{label} Results",
            elem_classes=["results-table"],
        )
        return btn, time_box, results, run_fn


with gr.Blocks(title="Hebrew Grocery Item Classifier", css=CSS, theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# Hebrew Grocery Item Classifier")
    gr.Markdown(
        "Add or remove lines below (one item per line), then classify with any "
        "or all of the models below to compare results and speed."
    )

    with gr.Column(elem_id="items-wrapper"):
        items_input = gr.Textbox(
            value="\n".join(EXAMPLE_ITEMS),
            lines=20,
            label="Grocery items (one per line)",
            elem_id="items-textbox",
        )

    with gr.Row():
        pt_btn, pt_time, pt_results, pt_fn = make_model_column("PyTorch", run_pytorch)
        fp32_btn, fp32_time, fp32_results, fp32_fn = make_model_column("ONNX (fp32)", run_onnx_fp32)
        int8_btn, int8_time, int8_results, int8_fn = make_model_column("ONNX (int8)", run_onnx_int8)

    pt_btn.click(fn=pt_fn, inputs=items_input, outputs=[pt_results, pt_time])
    fp32_btn.click(fn=fp32_fn, inputs=items_input, outputs=[fp32_results, fp32_time])
    int8_btn.click(fn=int8_fn, inputs=items_input, outputs=[int8_results, int8_time])

if __name__ == "__main__":
    demo.launch()
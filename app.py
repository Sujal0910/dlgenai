import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice

st.set_page_config(page_title="Smart MCQ Solver", layout="centered")
st.title("🧠 Smart MCQ Solver Challenge")
st.write("Enter a prompt and 5 options to get ranked top-3 predictions.")

prompt = st.text_area("Question Prompt")
c1, c2 = st.columns(2)
with c1:
    opt_a, opt_c, opt_e = st.text_input("A"), st.text_input("C"), st.text_input("E")
with c2:
    opt_b, opt_d = st.text_input("B"), st.text_input("D")

if st.button("Predict"):
    if prompt and all([opt_a, opt_b, opt_c, opt_d, opt_e]):
        @st.cache_resource
        def load_model():
            name = "roberta-large"
            return AutoTokenizer.from_pretrained(name), AutoModelForMultipleChoice.from_pretrained(name)

        tokenizer, model = load_model()
        inputs = tokenizer([prompt]*5, [opt_a, opt_b, opt_c, opt_d, opt_e], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()

        top_3 = np.argsort(logits)[::-1][:3]
        labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        opts = [opt_a, opt_b, opt_c, opt_d, opt_e]

        st.success("Predictions:")
        for i, idx in enumerate(top_3):
            st.write(f"**Rank {i+1}:** {labels[idx]} - {opts[idx]}")

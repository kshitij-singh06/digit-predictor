import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from scipy import ndimage
import os
import io

if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = 0

@st.cache_resource
def load_digit_model(model_path):
    try:
       
        model = load_model(model_path, compile=False)
        st.sidebar.success(f" Model loaded: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        st.sidebar.error(f" Failed to load model from path {model_path}: {e}")
        st.stop()


def preprocess_canvas_image(canvas_image_rgba):
    
    img = Image.fromarray(canvas_image_rgba.astype('uint8'), mode='RGBA').convert('L')
    arr = np.array(img)

    nonzero = np.where(arr > 10)
    if nonzero[0].size == 0:
      
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    top, left = np.min(nonzero[0]), np.min(nonzero[1])
    bottom, right = np.max(nonzero[0]), np.max(nonzero[1])
    cropped = arr[top:bottom+1, left:right+1]

    h, w = cropped.shape
    target_size = 20
    if h > w:
        new_h = target_size
        new_w = int(round((w * target_size) / h))
    else:
        new_w = target_size
        new_h = int(round((h * target_size) / w))
        
    new_w, new_h = max(1, new_w), max(1, new_h)
    
    resized = Image.fromarray(cropped).resize((new_w, new_h), Image.BILINEAR)

    new_img = Image.new('L', (28, 28), color=0)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    new_img.paste(resized, (x_offset, y_offset))

    final_arr = np.array(new_img).astype(np.float32)
    
    cy, cx = ndimage.center_of_mass(final_arr)
    if np.isnan(cx) or np.isnan(cy):
        cx, cy = 14, 14
    shift_x = np.round(14 - cx).astype(int)
    shift_y = np.round(14 - cy).astype(int)
    final_arr = ndimage.shift(final_arr, shift=(shift_y, shift_x), mode='constant', cval=0.0)

    final_arr = final_arr / 255.0
    final_arr = final_arr.reshape(1, 28, 28, 1)
    return final_arr

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")

st.title(" Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) below and click **Predict** to see the model's guess.")

model_path = "digit_classifier.keras"
if not os.path.exists(model_path):
    model_path = "digit_classifier.h5"

model = load_digit_model(model_path)

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=30,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state['canvas_key']}", 
    display_toolbar=False,
)

col1, col2 = st.columns([1, 2])
with col1:
    predict_btn = st.button("Predict")
with col2:
    clear_btn = st.button("Clear Canvas", key="clear_button")

if clear_btn:
    st.session_state['canvas_key'] += 1
    st.rerun()

if predict_btn:
    if canvas_result.image_data is not None:
        img_input = preprocess_canvas_image(canvas_result.image_data)
        
        if np.sum(img_input) == 0:
            st.warning("Please draw a digit before predicting!")
        else:
            prediction = model.predict(img_input)
            pred_class = int(np.argmax(prediction))
            confidence = float(np.max(prediction))

            st.subheader(" Model Prediction")
            st.success(f"**Digit:** {pred_class} \n\nConfidence: {confidence*100:.2f}%")

            st.write("### Prediction Probabilities")
            
            prob_df = {
                'Digit': list(range(10)),
                'Probability': prediction[0].tolist()
            }
            st.bar_chart(prob_df, x='Digit', y='Probability')

            st.write("### Processed Input (28x28)")
            img_display = (img_input[0].reshape(28,28) * 255).astype(np.uint8)
            st.image(img_display, width=100, caption="28x28 input to CNN")
            
    else:
        st.warning("Please draw something first!")
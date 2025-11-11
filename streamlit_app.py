import streamlit as st
import requests
import io
from PIL import Image
import numpy as np
import cv2 
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 🎨 قسم التجميل وتخصيص CSS (النهائي والمُصحَّح)
# -----------------------------------------------------------
st.markdown("""
<style>
/* 1. تخصيص الألوان الأساسية */
:root {
    --primary-color: #FF4B4B;      
    --background-color: #F0F2F7;   
    --secondary-background-color: #FFFFFF; 
    --text-color: #31333F;         
    --font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
}

.stApp {
    background-color: var(--background-color);
}

/* 2. تخصيص Header Streamlit (الرأس العلوي) - داكن وأنيق */
header {
    background-color: #004488 !important; /* 🛑 اللون الأزرق الداكن */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    visibility: visible !important; 
    height: auto !important;
    display: flex !important;
}

/* 2ب. جعل النص والروابط والأيقونات في Header بيضاء */
header a, header button, #MainMenu, header .stButton > button, header .stToolbar {
    color: #FFFFFF !important;
    border-color: #FFFFFF !important; 
}


/* 3. إخفاء التذييل الافتراضي لـ Streamlit */
footer {
    visibility: hidden; 
}

/* 4. Footer المخصص (داكن، ممتد، خط كبير) */
.footer-custom {
    color: #FFFFFF; /* لون الكتابة أبيض */
    text-align: center; 
    padding: 20px; 
    font-size: 20px; 
    font-weight: bold; 
    background-color: #004488; /* اللون الأزرق الداكن */
    position: fixed; /* لتثبيته في الأسفل */
    bottom: 0;
    width: 100%;
    left: 0; /* 🛑 التعديل للامتداد حتى اليسار */
    right: 0; /* 🛑 التعديل للامتداد حتى اليمين */
    z-index: 1000; 
}

/* 5. تخصيص مربع العنوان (Container) */
.title-container {
    padding: 20px;
    margin-bottom: 20px;
    background-color: var(--secondary-background-color); 
    border: 1px solid #ddd; 
    border-radius: 10px; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); 
    text-align: center;
}
.title-container h1 {
    margin: 0; 
    color: var(--text-color);
}
</style>
""", unsafe_allow_html=True)


# 🛑 استيرادات Grad-CAM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image_processing
from tensorflow.keras.applications import EfficientNetB0 
from tensorflow.keras.applications.efficientnet import preprocess_input 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input 

# ⚠️ المسارات وأسماء الطبقات
MODEL_PATH = r"C:/Users/HP/last modle v/final_web_compatible_model.h5" 
LAST_CONV_LAYER_NAME = "efficientnetb0" 
TARGET_SIZE = (224, 224)
API_URL = "https://amany-s-lung-cancer-api-fastapi.hf.space/predict"

# -----------------------------------------------------------
# 🛑 دالة تحميل الأوزان وبناء النموذج النظيف
# -----------------------------------------------------------
@st.cache_resource
def load_model_for_gradcam():
    full_model = None
    clean_gradcam_model = None
    last_conv_clean_name = None 

    try:
        full_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ تم تحميل النموذج الأصلي بنجاح.")

        efficientnet_weights = None
        try:
            efficientnet_layer = full_model.get_layer(LAST_CONV_LAYER_NAME)
            efficientnet_weights = efficientnet_layer.get_weights()
        except ValueError:
            pass 
            
        dense_layer = full_model.get_layer('dense')
        dense_weights = dense_layer.get_weights()
        
        input_tensor = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), name="clean_input")

        base_model_clean = EfficientNetB0(
            weights='imagenet', 
            include_top=False,  
            input_tensor=input_tensor
        )
        
        if efficientnet_weights:
            base_model_clean.set_weights(efficientnet_weights) 
            
        x = base_model_clean.output
        x = GlobalAveragePooling2D(name="global_average_pooling2d_clean")(x) 
        x = Dropout(0.5, name="dropout_clean")(x) 
        output_tensor = Dense(1, activation='sigmoid', name="dense_clean")(x) 
        
        clean_gradcam_model = Model(inputs=input_tensor, outputs=output_tensor)
        clean_gradcam_model.get_layer('dense_clean').set_weights(dense_weights)
        
        last_conv_clean_name = base_model_clean.layers[-1].name
        print(f"✅ تم بناء نموذج Grad-CAM النظيف. آخر طبقة Conv هي: {last_conv_clean_name}")
        
        return full_model, clean_gradcam_model, last_conv_clean_name
        
    except Exception as e:
        st.error(f"❌ فشل حاسم في تحميل النماذج. لا يمكن متابعة التطبيق. الخطأ: {e}")
        return None, None, None

# -----------------------------------------------------------
# تحميل النماذج
# -----------------------------------------------------------
try:
    full_model_original, clean_gradcam_model, LAST_CONV_CLEAN_NAME = load_model_for_gradcam()
except Exception:
    full_model_original, clean_gradcam_model, LAST_CONV_CLEAN_NAME = None, None, None

# -----------------------------------------------------------
# دالة Grad-CAM 
# -----------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    if model is None:
        return np.zeros(TARGET_SIZE)

    if isinstance(img_array, np.ndarray):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    if len(img_array.shape) == 3:
        img_array = tf.expand_dims(img_array, axis=0)

    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input, 
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception as e:
        st.warning(f"⚠️ فشل بناء Grad-CAM Model: {e}")
        return np.zeros(TARGET_SIZE)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        if pred_index is None: 
            pred_index = tf.argmax(preds[0]) 
            
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    if last_conv_layer_output.ndim == 4:
        last_conv_layer_output = last_conv_layer_output[0]
    
    pooled_grads = tf.expand_dims(pooled_grads, axis=-1)
    
    try:
        heatmap = last_conv_layer_output @ pooled_grads
    except Exception:
        return np.zeros(TARGET_SIZE)
        
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) 
    return heatmap.numpy()

# -----------------------------------------------------------
# واجهة Streamlit الرئيسية
# -----------------------------------------------------------

st.set_page_config(
    page_title="تنبؤ سرطان الرئة",
    page_icon="🩺",
    layout="centered"
)

# 🛑 رأس الصفحة داخل مربع مخصص (يعمل الآن بشكل صحيح مع CSS)
st.markdown(
    """
    <div class="title-container">
        <h1>🩺 تطبيق تنبؤ سرطان الرئة (الواجهة البسيطة)</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.subheader("تحميل صورة أشعة الرئة")
uploaded_file = st.file_uploader("اختر صورة بصيغة JPG أو PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="الصورة المحملة", use_container_width=True) 
    
    with col2:
        if st.button("🚀 تنبؤ وحرارة (Grad-CAM)", use_container_width=True):
            with st.spinner("جاري تحليل الصورة وتوليد الخريطة الحرارية..."):
                files = {"file": (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prob = float(result['prediction_probability'])
                    
                    st.subheader("✅ نتيجة التشخيص")
                    
                    if result['class'] == "Positive (مصابة)":
                        st.error(f"**النتيجة: {result['class']}** - **الاحتمالية: {prob * 100:.2f}%** 🚨")
                    else:
                        st.success(f"**النتيجة: {result['class']}** - **الاحتمالية: {prob * 100:.2f}%** ✅")
                    
                    st.markdown("---")

                    # ----------------- Grad-CAM -----------------
                    if clean_gradcam_model and LAST_CONV_CLEAN_NAME:
                        img_array_for_gradcam = keras_image_processing.img_to_array(image.resize(TARGET_SIZE))
                        img_array_for_gradcam = np.expand_dims(img_array_for_gradcam, axis=0)
                        
                        img_preprocessed_for_gradcam = preprocess_input(img_array_for_gradcam) 

                        heatmap = make_gradcam_heatmap(
                            img_preprocessed_for_gradcam,
                            clean_gradcam_model, 
                            LAST_CONV_CLEAN_NAME,
                            pred_index=None 
                        )
                        
                        if np.all(heatmap == 0):
                            st.warning("⚠️ فشل توليد خريطة Grad-CAM أو لا توجد مناطق نشطة. (قد تكون المشكلة في التدرجات الصفرية).")
                        else:
                            img_cv = cv2.cvtColor(np.array(image.resize(TARGET_SIZE)), cv2.COLOR_RGB2BGR)
                            heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
                            heatmap_resized = np.uint8(255 * heatmap_resized)
                            heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                            superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)
                            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
                            
                            st.subheader("🔥 مناطق الاهتمام (Grad-CAM)")
                            st.image(superimposed_img, caption="المناطق الحمراء/الصفراء هي الأكثر تأثيراً في قرار النموذج", use_container_width=True)

                    else:
                        st.warning("⚠️ لا يمكن عرض Grad-CAM لعدم توفر النموذج النظيف.")

                else:
                    st.error(f"❌ خطأ في الاتصال بالـ API (تحقق من تشغيل Uvicorn): {response.status_code}")

# -----------------------------------------------------------
# 🦶 Footer (التذييل) - استخدام الفئة المخصصة الممتدة
# -----------------------------------------------------------
st.markdown(
    """
    <div class="footer-custom"> 
        © 2025 تطبيق تشخيص سرطان الرئة (Grad-CAM) | مشروع تخرج
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image_processing
from tensorflow.keras.applications.efficientnet import preprocess_input # 🛑 المعالجة المسبقة الصحيحة

import numpy as np
import io
from PIL import Image

# --- إعدادات FastAPI ---
app = FastAPI(title="Lung Cancer Prediction API", description="Serves the saved Keras model.")

# --- تحميل النموذج ---
MODEL_PATH = r"C:/Users/HP/last modle v/final_web_compatible_model.h5" 

try:
    model = load_model(MODEL_PATH)
    print("✅ تم تحميل النموذج بنجاح من:", MODEL_PATH)
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Check file path and TF 2.15 compatibility.")


# --- وظيفة معالجة الصورة ---
def preprocess_image(img: Image.Image):
    """
    تطبيق المعالجة المسبقة الصحيحة للنموذج (باستخدام preprocess_input).
    """
    img = img.resize((224, 224))
    img_array = keras_image_processing.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # تطبيق التهيئة الصحيحة للنموذج (تحويل إلى نطاق [-1, 1])
    img_preprocessed = preprocess_input(img_array) 
    return img_preprocessed


# --- نقطة النهاية (Endpoint) للتنبؤ ---
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف صورة (Image).")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_img = preprocess_image(img)
        
        predictions = model.predict(processed_img)
        
        # 🛑 استخدام الإخراج الخام للنموذج (احتمالية الفئة التي تم تدريب النموذج عليها)
        probability = predictions[0][0].item() 
        
        # تطبيق العتبة المعدلة (0.45) 
        DECISION_THRESHOLD = 0.45
        
        # 🔄 الإصلاح النهائي: عكس التسمية
        # إذا كانت القيمة عالية، فهذا يعني أن النموذج يرى سليمًا، لذا نصنفه كـ "سليمة" (Negative).
        # إذا كانت القيمة منخفضة، فهذا يعني أن النموذج يرى مصابًا، لذا نصنفه كـ "مصابة" (Positive).
        class_label = "Negative (سليمة)" if probability >= DECISION_THRESHOLD else "Positive (مصابة)"

        return {
            "filename": file.filename,
            "prediction_probability": f"{probability:.4f}",
            "class": class_label,
            "threshold_used": DECISION_THRESHOLD,
            "message": "تم إصلاح منطق الفئات بنجاح."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"حدث خطأ داخلي أثناء التنبؤ: {e}")

# --- نقطة النهاية الأساسية (Test) ---
@app.get("/")
def read_root():
    return {"status": "API is running and ready", "model_loaded": True}

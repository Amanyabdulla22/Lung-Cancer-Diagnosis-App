import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np

# 🚨 الرابط الصحيح لنقطة نهاية FastAPI على Hugging Face Spaces
API_URL = "https://amany-s-lung-cancer-api-fastapi.hf.space/"
st.set_page_config(
    page_title="تطبيق تنبؤ سرطان الرئة (الواجهة البسيطة)",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- واجهة المستخدم ---
st.title("🩺 تطبيق تنبؤ سرطان الرئة (الواجهة البسيطة) 2025")
st.markdown("---")

st.subheader("تحميل صورة الأشعة السينية")
uploaded_file = st.file_uploader("اختر صورة أشعة سينية للصدر (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المحملة
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة المحملة", use_column_width=True)

    # زر التنبؤ
    if st.button("تشخيص الحالة (تنبؤ و Grad-CAM)"):
        with st.spinner('جاري إرسال الصورة إلى خادم API...'):
            try:
                # تحويل الصورة إلى بيانات ثنائية
                image_bytes = uploaded_file.getvalue()

                # إرسال طلب POST إلى API
                files = {'file': (uploaded_file.name, image_bytes, uploaded_file.type)}
                response = requests.post(API_URL, files=files)
                
                # التحقق من استجابة API
                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ تم استلام نتائج التشخيص بنجاح!")
                    
                    # 1. عرض النتيجة النهائية
                    st.markdown("## 📊 النتيجة النهائية")
                    prediction_text = data.get("prediction", "غير متوفر")
                    probability = data.get("probability", 0)
                    
                    st.metric(label="احتمالية الإصابة بالسرطان", value=f"{probability:.2f}%")
                    st.info(f"النتيجة المتوقعة: **{prediction_text}**")
                    
                    # 2. عرض خريطة Grad-CAM
                    st.markdown("## 🔬 خريطة التفسير (Grad-CAM)")
                    
                    # استلام صورة Grad-CAM كبيانات Base64
                    grad_cam_base64 = data.get("grad_cam_image", None)
                    
                    if grad_cam_base64:
                        import base64
                        from io import BytesIO
                        
                        # فك تشفير صورة Grad-CAM من Base64
                        grad_cam_bytes = base64.b64decode(grad_cam_base64)
                        grad_cam_image = Image.open(BytesIO(grad_cam_bytes))
                        
                        st.image(grad_cam_image, caption="خريطة Grad-CAM تُظهر المنطقة الحرجة", use_column_width=True)
                        st.caption("اللون الأكثر سخونة يشير إلى المنطقة التي اعتمد عليها النموذج للتنبؤ.")
                    else:
                        st.warning("تعذر استلام خريطة Grad-CAM من خادم API.")

                else:
                    st.error(f"❌ خطأ في الاتصال بخادم API. رمز الحالة: {response.status_code}")
                    st.error(f"الرسالة التفصيلية من الخادم: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ فشل في الاتصال بالـ API. يرجى التأكد من أن Hugging Face Space قيد التشغيل (Running).")
            except Exception as e:
                st.error(f"❌ حدث خطأ غير متوقع: {e}")

# --- تذييل ---
st.markdown("---")
st.caption("مشروع تخرج (Grad-CAM) | تطبيق تشخيص سرطان الرئة 2025")

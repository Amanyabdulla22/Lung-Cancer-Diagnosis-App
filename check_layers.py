import tensorflow as tf

# ⚠️ تأكدي من أن هذا المسار صحيح تمامًا!
MODEL_PATH = r"C:/Users/HP/last modle v/final_web_compatible_model.h5" 

try:
    # نستخدم compile=False لتجنب مشاكل المترجم (Optimizer)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
    
    print("------------------------------------------------------")
    print("✅ تم تحميل النموذج. قائمة بأسماء طبقات النموذج:")
    print("------------------------------------------------------")
    
    # طباعة أسماء الطبقات
    for i, layer in enumerate(model.layers):
        # سنعرض آخر 10 طبقات تقريباً، لأن الطبقات المهمة تكون في النهاية
        if i >= len(model.layers) - 20: 
            print(f"Index {i}: {layer.name} (Type: {type(layer).__name__})")
            
    print("\n------------------------------------------------------")
    print("💡 ابحثي عن آخر طبقة من نوع Convolution (Conv2D) قبل طبقة GlobalAveragePooling2D أو Dropout أو Dense.")
    
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
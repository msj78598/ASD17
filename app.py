import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# تحميل النموذج المدرب
# تحديث المسار إلى النموذج المدرب
model_path = 'XGBoost_with_SMOTE.pkl'

# تحميل النموذج المدرب
model = joblib.load(model_path)


# دالة لتحليل البيانات باستخدام النموذج المدرب
def analyze_data(data):
    # التنبؤ
    X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]
    predictions = model.predict(X)
    data['Predicted_Loss'] = predictions
    
    # إضافة سبب الفقد
    data['Loss_Reason'] = data.apply(add_loss_reason, axis=1)

    return data

# إضافة تفسير لحالات الفقد
def add_loss_reason(row):
    if row['V1'] == 0 and row['A1'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V1'
    elif row['V2'] == 0 and row['A2'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V2'
    elif row['V3'] == 0 and row['A3'] > 0:
        return '⚠️ فقد بسبب جهد صفر وتيار على V3'
    else:
        return '✅ لا توجد حالة فقد مؤكدة'

# واجهة المستخدم باستخدام Streamlit
st.title("نظام اكتشاف الفاقد الكهربائي")

st.markdown("### 📤 قم برفع ملف البيانات للتحليل (Excel)")
uploaded_file = st.file_uploader("رفع الملف", type=["xlsx"])

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)

        # تطبيق النموذج لتحليل البيانات
        analyzed_data = analyze_data(data)

        # عرض النتائج
        st.subheader("📋 نتائج الفقد المحللة")
        st.dataframe(analyzed_data)

        # توفير رابط لتحميل النتائج
        output = BytesIO()
        analyzed_data.to_excel(output, index=False)
        output.seek(0)

        st.download_button(
            label="📥 تحميل نتائج التحليل",
            data=output,
            file_name="predicted_loss_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تحميل الملف: {str(e)}")

st.markdown("---")
st.markdown("👨‍💻 **المطور: مشهور العباس -00966553339838**")

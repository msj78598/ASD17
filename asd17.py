import os
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE  # لاستخدام SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# إنشاء المجلدات إذا كانت غير موجودة
model_folder = "C:\\Users\\Sec\\Documents\\ASD17\\models"
metrics_folder = "C:\\Users\\Sec\\Documents\\ASD17\\metrics"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
if not os.path.exists(metrics_folder):
    os.makedirs(metrics_folder)

# تحميل البيانات من ملف Excel
def load_data(file_path):
    data = pd.read_excel(file_path)
    
    # التأكد من وجود قيم مفقودة وتعويضها باستخدام المتوسط
    imputer = SimpleImputer(strategy='mean')  # استخدام المتوسط لتعويض القيم المفقودة
    data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']] = imputer.fit_transform(data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']])
    
    X = data[['V1', 'V2', 'V3', 'A1', 'A2', 'A3']]  # الميزات (الفولت والتيار)
    y = data['results']  # الهدف (1 للفاقد، 0 للطبيعي)
    
    return X, y

# استخدام Cross-validation لتقييم النموذج
def train_and_save_model_with_cv(X_train, y_train, X_test, y_test, model_name):
    model = XGBClassifier(n_estimators=100, max_depth=10, random_state=42, scale_pos_weight=10)  # XGBoost مع التعامل مع البيانات غير المتوازنة
    
    # استخدام Cross-validation لتقييم النموذج
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # طباعة نتائج Cross-validation
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    
    # تدريب النموذج على كامل بيانات التدريب
    model.fit(X_train, y_train)
    
    # التنبؤ على مجموعة الاختبار
    y_pred = model.predict(X_test)
    
    # حساب دقة النموذج على مجموعة الاختبار
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy:.2f}")
    
    # حفظ النموذج
    model_path = os.path.join(model_folder, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    
    # حفظ المقاييس
    metrics_path = os.path.join(metrics_folder, f"{model_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write(f"Accuracy: {accuracy:.2f}\n")
    
    return accuracy

# تحميل البيانات
file_path = 'C:\\Users\\Sec\\Documents\\ASD17\\ALL.xlsx'  # مسار الملف
X, y = load_data(file_path)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# طباعة توزيع الفئات قبل تطبيق SMOTE
print(f"Distribution in training data before SMOTE:\n{y_train.value_counts()}")

# استخدام SMOTE لزيادة البيانات في الفئة الأقل
if len(y_train.value_counts()) > 1:
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # طباعة توزيع البيانات بعد تطبيق SMOTE
    print(f"Training data size after SMOTE: {len(X_train_res)}")
    print(f"Distribution after SMOTE:\n{pd.Series(y_train_res).value_counts()}")
else:
    print("Warning: SMOTE cannot be applied as there is only one class in the training data.")

# تدريب النموذج وحفظه
accuracy = train_and_save_model_with_cv(X_train_res, y_train_res, X_test, y_test, "XGBoost_with_SMOTE")

# عرض الدقة
print(f"Model Accuracy after SMOTE: {accuracy:.2f}")

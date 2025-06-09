# 25619005-
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

features = train.loc[:, ~train.columns.isin(['ID', 'Cancer'])]
labels = train['Cancer']

test_features = test.drop(columns=['ID'])

from sklearn.preprocessing import LabelEncoder

# 범주형 변수 식별
categorical_cols = X.select_dtypes(include='object').columns

# 각 범주형 컬럼별로 Label Encoding 수행
for column in categorical_cols:
    encoder = LabelEncoder()
    
    # 학습 데이터에 대해 피팅
    X[column] = encoder.fit_transform(X[column])
    
    # 테스트 데이터에 존재하지만 학습 데이터에 없는 클래스 처리
    unseen_classes = set(x_test[column].unique()) - set(encoder.classes_)
    if unseen_classes:
        encoder.classes_ = np.concatenate([encoder.classes_, list(unseen_classes)])
    
    # 테스트 데이터 변환
    x_test[column] = encoder.transform(x_test[column])

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

   def train_and_eval(X_tr, y_tr, X_val, y_val, label):
    model = XGBClassifier(random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f"[{label}] Validation F1-score: {f1:.4f}")
    return model, f1 

    from imblearn.over_sampling import SMOTE

# (1) SMOTE 미적용 모델 학습 및 평가
model_no_smote, f1_no_smote = train_and_eval(
    X_train, y_train, X_val, y_val, label="No_SMOTE"
)

# (2) SMOTE 적용 후 모델 학습 및 평가
smote_sampler = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote_sampler.fit_resample(X_train, y_train)

model_with_smote, f1_with_smote = train_and_eval(
    X_train_balanced, y_train_balanced, X_val, y_val, label="With_SMOTE"
)

# SMOTE 적용 여부에 따라 최종 학습 데이터 구성
if f1_smote >= f1_raw:
    smote_full = SMOTE(random_state=42)
    X_final, y_final = smote_full.fit_resample(X, y)
else:
    X_final, y_final = X, y

# 최종 모델 학습
final_model = XGBClassifier(random_state=42)
final_model.fit(X_final, y_final)

final_pred = final_model.predict(x_test)

submission = pd.read_csv('sample_submission.csv')

submission['Cancer'] = final_pred

submission.to_csv('/content/sample_data/baseline_submit.csv', index=False)

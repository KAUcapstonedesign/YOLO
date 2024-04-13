import json
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# 데이터 로드 및 전처리
directory = 'C:\capstonedesign\landmark'  # JSON 파일이 저장된 디렉토리 경로

def load_data_from_json(directory): # JSON 파일을 로드하는 함수
    features = []   # 특성 벡터를 저장할 리스트
    labels = []     # label을 저장할 리스트
    for filename in os.listdir(directory):  # 디렉토리 내의 모든 JSON 파일을 반복 처리           
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)     # 파일 경로 생성      
            with open(filepath, 'r') as file:  
                data = json.load(file)  # JSON 데이터 로드
                landmarks = data['person_identifier']['landmarks']  # person_identifier list에서 landmarks를 가져옴
                person_features = []    # 사람의 특성 벡터를 저장할 리스트
                for landmark in landmarks:
                    person_features += [landmark['x'], landmark['y']]   # x, y 좌표를 특성 벡터에 추가
                features.append(person_features)  # 특성 벡터를 features 리스트에 추가
            labels.append(data['person_identifier']['name'])        
    return np.array(features), np.array(labels)              

X, y = load_data_from_json(directory)  # 데이터 로드

# 데이터를 학습 세트와 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)   # test_size=0.2로 80:20 비율로 분리, random_state=10으로 고정

# 모델 선택 및 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)   # 랜덤 포레스트 모델 선택, n_estimators는 트리 개수, random_state는 시드값
model.fit(X_train, y_train) # 모델 학습

# 학습된 모델을 사용하여 테스트 세트의 정확도 평가
predictions = model.predict(X_test)  # 테스트 세트에 대한 예측
print("Accuracy:", accuracy_score(y_test, predictions))  # 정확도 출력, 정확도는 0~1 사이의 값, 높을수록 정확도가 높음

# 모델 저장
joblib.dump(model, 'face_landmarks_trained_model.pkl')

# Confusion Matrix
cm = confusion_matrix(y_test, predictions, labels=model.classes_)
plt.figure(figsize=(10, 7))
plt.title('Confusion Matrix')
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plotting the results
fig, ax = plt.subplots()
for i in range(len(X_test)):
    if predictions[i] == y_test[i]:
        ax.scatter(X_test[i, 0], X_test[i, 1], color='blue')  # Correct predictions in blue
    else:
        ax.scatter(X_test[i, 0], X_test[i, 1], color='red')  # Incorrect predictions in red

ax.set_title('Test Set Facial Landmarks Classification Result')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
plt.show()
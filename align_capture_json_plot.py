import matplotlib.pyplot as plt
import cv2
import dlib
import json

# 모델 및 카메라 초기화
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
capture = cv2.VideoCapture(0)
capture.set(3, 256)  # 너비
capture.set(4, 256)  # 높이

face_id = input('\n enter user id and press <enter> ==> ')
user_name = input('\n enter user name and press <enter> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0
landmark_labels = [str(i) for i in range(68)]  # 0부터 67까지의 라벨
json_files = [] # JSON 파일 경로를 저장할 리스트

# 얼굴 정렬 함수
def align_faces(frame):
    dets = detector(frame, 1)
    faces = []
    for det in dets:
        shape = predictor(frame, det)
        face_chip = dlib.get_face_chip(frame, shape, size=256, padding=0.3)
        faces.append(face_chip)
    return faces

# 얼굴 캡처 및 랜드마크 추출
while True:
    ret, frame = capture.read()
    if not ret:
        print("[ERROR] Unable to capture video")
        break
    
    # 얼굴 정렬
    aligned_faces = align_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    for aligned_face in aligned_faces:
        count += 1
        # 정렬된 얼굴 사진을 흑백으로 변환
        aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
        img_path = f"dataset/face_{face_id}_{count}.jpg"
        cv2.imwrite(img_path, aligned_gray)  # 흑백 이미지 저장
        
        # 랜드마크 추출 및 라벨 추가
        rects = detector(aligned_gray, 1)
        for rect in rects:
            landmarks = predictor(aligned_gray, rect)
            landmark_list = [{"x": p.x, "y": p.y, "label": landmark_labels[i]} for i, p in enumerate(landmarks.parts())]
            
            # 랜드마크 JSON 데이터 구조에 저장
            json_data = {
                "person_identifier": {
                    "name": user_name,
                    "id": face_id,
                    "landmarks": landmark_list
                }
            }
            
            json_path = f"dataset/face_{face_id}_{count}.json"
            json_files.append(json_path)  # JSON 파일 경로 저장
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27 or count >= 68:  # ESC 키를 누르거나 68회 캡처 시 종료
        break

print("\n [INFO] Exiting Program and cleanup stuff")
capture.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))  # 그래프 크기 설정

for json_path in json_files:
    with open(json_path) as file:
        data = json.load(file)
        landmarks = data["person_identifier"]["landmarks"]
        
        # x, y 좌표 분리
        x = [landmark['x'] for landmark in landmarks]
        y = [landmark['y'] for landmark in landmarks]

        plt.scatter(x, y, alpha=0.5)

plt.gca().invert_yaxis()  # y축을 상단이 0이 되도록 변경
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Facial Landmarks')
plt.show()

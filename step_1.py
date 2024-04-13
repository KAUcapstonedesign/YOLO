# Code 있는 폴더에 image, landmark 폴더 생성 후 실행
import matplotlib.pyplot as plt
import cv2
import dlib
import json
from scipy.stats import gaussian_kde
import numpy as np

# Dlib의 얼굴 감지기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector() # 얼굴 감지기
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 랜드마크 예측기  
capture = cv2.VideoCapture(0)   # webcam(0번) 연결
capture.set(3, 256)
capture.set(4, 256)

face_id = input('\n enter user id and press <enter> ==> ')
user_name = input('\n enter user name and press <enter> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
count = 0   # 캡처 횟수 저장할 변수 초기화
landmark_labels = [str(i) for i in range(68)]  # 0부터 67까지의 landmark 라벨
json_files = [] # JSON 파일 경로를 저장할 리스트

# 얼굴 정렬 함수 선언
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
        aligned_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)   # 정렬된 얼굴 사진을 흑백으로 변환
        img_path = f"image/face_{face_id}_{count}.jpg"
        cv2.imwrite(img_path, aligned_gray)  # 흑백 이미지 저장
        
        # 랜드마크 추출
        rects = detector(aligned_gray, 1)
        for rect in rects:
            landmarks = predictor(aligned_gray, rect)
            landmark_list = [{"x": p.x, "y": p.y, "label": landmark_labels[i]} for i, p in enumerate(landmarks.parts())]
            
            # Data label 생성
            json_data = {
                "person_identifier": {
                    "name": user_name,
                    "id": face_id,
                    "landmarks": landmark_list
                }
            }
            json_path = f"landmark/face_{face_id}_{count}.json"
            json_files.append(json_path)  # 랜드마크 좌표 JSON 파일로 저장
            with open(json_path, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == 27 or count >= 100:  # ESC 키를 누르거나 캡처 완료 시 종료
        break

print("\n [INFO] Exiting Program and cleanup stuff")
capture.release()
cv2.destroyAllWindows()

plt.figure(figsize=(10, 10))
for json_path in json_files:
    with open(json_path) as file:
        data = json.load(file)
        landmarks = data["person_identifier"]["landmarks"]  # JSON 파일에서 랜드마크 가져오기
        x = [landmark['x'] for landmark in landmarks]
        y = [landmark['y'] for landmark in landmarks]
        plt.scatter(x, y, alpha=0.5)

plt.gca().invert_yaxis()  # y축을 상단이 0이 되도록 변경
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Distribution of Facial Landmarks')
plt.show()

def gaussian_graph_for_landmarks(json_files):
    # Initialize a dictionary to hold all coordinates for each landmark
    landmarks_data = {i: {'x': [], 'y': []} for i in range(68)}
    
    # Load the data from the JSON files
    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)
            landmarks = data['person_identifier']['landmarks']
            for landmark in landmarks:
                idx = int(landmark['label'])
                landmarks_data[idx]['x'].append(landmark['x'])
                landmarks_data[idx]['y'].append(landmark['y'])
                
    # Plotting the Gaussian graph using KDE for each landmark number
    fig, axs = plt.subplots(8, 9, figsize=(20, 20))  # Adjust the grid size according to the number of landmarks
    axs = axs.ravel()
    for i, coords in landmarks_data.items():
        # Perform a Gaussian KDE on the data:
        x = np.array(coords['x'])
        y = np.array(coords['y'])
        xy = np.vstack([x,y])
        kde = gaussian_kde(xy)
        
        # Create a grid over which we'll evaluate the KDE:
        x_grid = np.linspace(np.min(x), np.max(x), 100)
        y_grid = np.linspace(np.min(y), np.max(y), 100)
        Xgrid, Ygrid = np.meshgrid(x_grid, y_grid)
        
        # Evaluate the KDE on the grid:
        Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        
        # Plot the result as an image:
        axs[i].imshow(Z.reshape(Xgrid.shape),
                      origin='lower', aspect='auto',
                      extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
                      cmap='viridis')
        axs[i].scatter(x, y, s=5, facecolor='white')
        axs[i].set_title(f'Landmark {i}', fontsize=8)
        axs[i].tick_params(axis='both', which='both', labelsize=6)
        axs[i].invert_yaxis()  # Invert the y-axis to match the image coordinates
        
    plt.tight_layout()
    plt.show()
gaussian_graph_for_landmarks(json_files)

def line_gaussian_for_landmarks(json_files):
    # Initialize a dictionary to hold all coordinates for each landmark
    landmarks_data = {i: {'x': [], 'y': []} for i in range(68)}
    
    # Load the data from the JSON files
    for json_path in json_files:
        with open(json_path, 'r') as file:
            data = json.load(file)
            landmarks = data['person_identifier']['landmarks']
            for landmark in landmarks:
                idx = int(landmark['label'])
                landmarks_data[idx]['x'].append(landmark['x'])
                landmarks_data[idx]['y'].append(landmark['y'])

    # Plotting the Gaussian graph for each landmark number
    fig, axs = plt.subplots(17, 4, figsize=(20, 40))  # Adjust grid size for the number of landmarks
    axs = axs.ravel()
    for i, coords in landmarks_data.items():
        x = np.array(coords['x'])
        y = np.array(coords['y'])

        # Create gaussian KDEs for x and y coordinates
        x_kde = gaussian_kde(x)
        y_kde = gaussian_kde(y)

        # Create a range of values over which to evaluate the KDEs
        x_range = np.linspace(np.min(x), np.max(x), 500)
        y_range = np.linspace(np.min(y), np.max(y), 500)

        # Evaluate KDEs over the range
        x_density = x_kde(x_range)
        y_density = y_kde(y_range)

        # Plot the densities
        axs[i].plot(x_range, x_density, label='X Density', color='blue')
        axs[i].plot(y_range, y_density, label='Y Density', color='green')
        axs[i].fill_between(x_range, x_density, alpha=0.5, color='blue')
        axs[i].fill_between(y_range, y_density, alpha=0.5, color='green')
        axs[i].set_title(f'Landmark {i}', fontsize=8)
        axs[i].tick_params(axis='both', which='both', labelsize=6)

    plt.tight_layout()
    plt.show()
line_gaussian_for_landmarks(json_files)

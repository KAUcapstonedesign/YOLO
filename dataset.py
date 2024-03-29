import cv2  # opencv 사용

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # 얼굴 인식

capture = cv2.VideoCapture(0) # 0번 카메라 실행(노트북 기본 카메라)  
capture.set(cv2.CAP_PROP_FRAME_WIDTH,256) # CAP_PROP_FRAME_WIDTH == 3 / 카메라 너비 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,256) # CAP_PROP_FRAME_HEIGHT == 4 / 카메라 높이 설정

face_id = input('\n enter user id end press <enter> ==> ') # 사용자 이름 입력(숫자)
print("\n [INFO] Initializing face capture. Look the camera and wait ...")  # 진행 상황 알려주는 메시지

count = 0   # 사진 저장할 때 사용할 변수

while True: 
    ret, frame = capture.read() # 카메라 상태(정상 작동할 경우 ret은 true) 및 현재 프레임(이미지) 저장3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 컬러 -> 흑백으로 변환
    faces = faceCascade.detectMultiScale(   
        gray,   # gray scale로 변환한 이미지
        scaleFactor = 1.05,  # 검색 윈도우 확대 비율, 1보다 커야 한다 / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 1.05~1.4 정도로 설정
        minNeighbors = 3,   # 얼굴 사이 최소 간격(픽셀) / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 3~6 정도로 설정
        minSize=(20,20) # 얼굴 최소 크기, 이 값보다 작으면 무시 / (30,30)으로 설정하면 얼굴이 너무 작아서 검출이 안됨 
    )
    
    for (x,y,w,h) in faces: # 얼굴이 검출되면 / (x,y): 얼굴 좌표 / w, h: 얼굴 너비와 높이
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)    # 얼굴에 사각형 표시 / (x+w,y+h): 얼굴 우측 하단 좌표, (255,255,255): white, 2: 선 두께 
        count += 1  # 얼굴 사진 개수 + 1
        cv2.imwrite(f"dataset/face_{face_id}_{count}.jpg", gray[y:y+h, x:x+w])  # dataset 폴더에 얼굴 사진 jpg 형식으로 저장
    cv2.imshow('image',frame)   # 노트북 카메라 영상 출력

    # 종료 조건
    if cv2.waitKey(1) > 0 : break # 키 입력이 있을 때 종료
    elif count >= 100 : break # 저장된 얼굴 사진이 100개가 넘어가면 종료
'''
# 폴더에 labeling txt 파일 생성
for count_label in range(1, 101):   # range(시작, 끝+1)만큼 반복 / 라벨링 txt 파일 생성
    folder = "train" if count_label <= 70 else ("valid" if count_label <= 90 else "test")   # 1~70: train, 71~90: valid, 91~100: test로 나누기 위함
    file_path = f"data/{folder}/labels/Face_{face_id}_{count_label}.txt"    # 파일 경로
    with open(file_path, "w", encoding="utf8") as file: # 파일이 없으면 생성, 있으면 덮어쓰기
        file.write(f"{face_id} 0.490000 0.490000 0.999000 0.999000")    # 파일에 라벨링 정보 입력
'''
print("\n [INFO] Exiting Program and cleanup stuff")    # 진행 상황 알려주는 메시지

capture.release()   # 메모리 해제
cv2.destroyAllWindows() # 모든 윈도우 창 닫기

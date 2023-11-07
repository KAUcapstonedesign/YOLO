import cv2  # opencv 사용

faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # 얼굴 인식

capture = cv2.VideoCapture(0) # 0번 카메라 실행(노트북 기본 카메라)  
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) # CAP_PROP_FRAME_WIDTH == 3   
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) # CAP_PROP_FRAME_HEIGHT == 4

face_id = input('\n enter user id end press <return> ==> ') # 사용자 이름 입력(숫자)
print("\n [INFO] Initializing face capture. Look the camera and wait ...")  # 진행 상황 알려주는 메시지

count = 0   # 사진 저장할 때 사용할 변수

# 영상 처리 및 출력
while True: 
    ret, frame = capture.read() # 카메라 상태(정상 작동할 경우 ret은 true) 및 현재 프레임(이미지) 저장3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 컬러 -> 흑백으로 변환
    faces = faceCascade.detectMultiScale(   
        gray,   # gray scale로 변환한 이미지
        scaleFactor = 1.2,  # 검색 윈도우 확대 비율, 1보다 커야 한다 / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 1.05~1.4 정도로 설정
        minNeighbors = 6,   # 얼굴 사이 최소 간격(픽셀) / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 3~6 정도로 설정
        minSize=(20,20) # 얼굴 최소 크기, 이 값보다 작으면 무시
    )

    for (x,y,w,h) in faces: # 얼굴이 검출되면 / (x,y) : 얼굴 좌표, w,h : 얼굴 너비와 높이
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)    # 얼굴에 사각형 표시 / (x+w,y+h) : 얼굴 우측 하단 좌표, (255,0,0) : blue, 2 : 선 두께 
        count += 1  # 얼굴 사진 개수 + 1
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])  # 얼굴 사진 jpg 형식으로 저장
    cv2.imshow('image',frame)   # 노트북 카메라 영상 출력

    # 종료 조건
    if cv2.waitKey(1) > 0 : break # 키 입력이 있을 때 반복문 종료
    elif count >= 100 : break # 저장된 얼굴 사진이 100개가 넘어가면 종료

print("\n [INFO] Exiting Program and cleanup stuff")    # 진행 상황 알려주는 메시지

capture.release()   # 메모리 해제
cv2.destroyAllWindows() # 모든 윈도우 창 닫기

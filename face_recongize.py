import cv2  # opencv 사용
import numpy as np  # numpy 사용

recognizer = cv2.face.LBPHFaceRecognizer_create()   # LBPH 알고리즘 사용
recognizer.read('trainer/trainer.yml')  # 학습 결과 불러오기
cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'    # 얼굴 인식
faceCascade = cv2.CascadeClassifier(cascadePath)    # 얼굴 인식

font = cv2.FONT_HERSHEY_SIMPLEX   # 폰트 설정 

id = 0  # id 초기화

names = ['hyunsoo','kichul','youngjin','jungsub']   # 사용자 이름 리스트

cam = cv2.VideoCapture(0)   # 0번 카메라 실행(노트북 기본 카메라)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980) # 카메라 너비 설정
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 카메라 높이 설정

minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # 카메라 너비의 10%
minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT) # 카메라 높이의 10%

while True: # 무한 반복
    ret, img = cam.read()   # 카메라 상태 및 현재 프레임(이미지) 저장
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 컬러 -> 흑백으로 변환

    faces = faceCascade.detectMultiScale(   # 얼굴 검출
        gray,   # gray scale로 변환한 이미지
        scaleFactor=1.2,    # 검색 윈도우 확대 비율, 1보다 커야 한다 / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 1.05~1.4 정도로 설정
        minNeighbors=6, # 얼굴 사이 최소 간격(픽셀) / 값이 높을수록 정확도는 떨어지지만 검출률이 높아진다 / 3~6 정도로 설정
        minSize=(int(minW), int(minH))  # 얼굴 최소 크기, 이 값보다 작으면 무시(카메라 너비의 10%, 높이의 10%보단 커야 함)
    )

    for(x,y,w,h) in faces:  # 얼굴이 검출되면
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)   # 얼굴에 사각형 표시 / (x+w,y+h) : 얼굴 우측 하단 좌표, (255,0,0) : blue, 2 : 선 두께
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])   # 검출된 얼굴을 학습한 결과와 비교하여 id, confidence 저장
        # confidence : 정확도 (0~100) / 0에 가까울수록 label과 일치한다는 의미(label과 얼마나 가까운지 생각하면 됨)

        if confidence < 50 :    # 50 이하의 정확도만 출력  
            id = names[id]  # id에 해당하는 이름 저장
        else:
            id = "unknown"  # 정확도가 높을수록 값이 작아지므로 50 이상이면 unknown으로 표시 
        
        confidence = "  {0}%".format(round(100-confidence)) # format으로 정확도 표시 (반올림하여 소수점 2자리까지) / 정확도가 높을수록 값이 작아지므로 100에서 빼야 일치율이 높아짐

        cv2.putText(img,str(id), (x+5,y-5),font,1,(255,255,255),2)  # 이름 표시 / (x+5,y-5) : 얼굴 좌측 상단 좌표, (255,255,255) : white, 2 : 선 두께
        cv2.putText(img,str(confidence), (x+5,y+h-5),font,1,(255,255,0),1)  # 정확도 표시 / (x+5,y+h-5) : 얼굴 좌측 하단 좌표, (255,255,0) : yellow, 1 : 선 두께
    
    cv2.imshow('camera',img)    # 카메라 영상 출력
    if cv2.waitKey(1) > 0 : break   # 키 입력이 있을 때 반복문 종료

print("\n [INFO] Exiting Program and cleanup stuff")    # 진행 상황 알려주는 메시지
cam.release()   # 메모리 해제
cv2.destroyAllWindows() # 모든 윈도우 창 닫기

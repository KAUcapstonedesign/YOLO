import cv2 # opencv 사용

capture = cv2.VideoCapture(0)    # 0번 카메라를 사용하겠다는 의미
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # 카메라의 해상도를 1280x720으로 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 카메라의 해상도를 1280x720으로 설정

while True:                         # 무한루프
    ret, frame = capture.read()     # 카메라의 ret(정상작동여부), frame(카메라 영상)을 받아옴
    cv2.imshow("original", frame)   # frame(카메라 영상)을 original 이라는 창에 띄워줌 
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프 탈출
            break                   # 무한루프 탈출
    
capture.release()                   # 카메라 객체 해제
cv2.destroyAllWindows()             # 모든 윈도우창을 없애줌
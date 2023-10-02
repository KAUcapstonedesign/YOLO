import cv2 # opencv 사용

img = cv2.imread('cat.jpg') # 이미지 읽기

cv2.imshow('image', img) # 이미지 출력
cv2.waitKey() # 키 입력 대기

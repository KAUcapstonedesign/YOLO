import cv2  # opencv 사용

img = cv2.imread("cat.jpg") # 이미지 읽기
print("img.shape = {0}".format(img.shape))  # 이미지 크기 출력

resize_img = cv2.resize(img, (1000, 500))    # 이미지 크기 조절(가로 1000px, 세로 500px)
print("resize_img.shape = {0}".format(resize_img.shape))    # 이미지 크기 출력

cv2.imshow("resize img", resize_img)    # 이미지 출력
cv2.waitKey()   # 키 입력 대기
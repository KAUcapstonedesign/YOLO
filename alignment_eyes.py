# **MTCNN을 이용한 눈 기준으로 얼굴 정렬(얼굴만 뽑는게 아니고 사진 전체를 돌림)**
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np

def align_face(image):      # 얼굴 정렬 함수
    detector = MTCNN()          # MTCNN 디텍터 생성
    detection = detector.detect_faces(image)            # 얼굴 검출
    print(detection)        # 검출 결과 출력

    if detection:    # 얼굴 검출 성공 시
        keypoints = detection[0]['keypoints']       # 얼굴 특징점 추출
        left_eye = keypoints['left_eye']        # 왼쪽 눈 좌표
        right_eye = keypoints['right_eye']      # 오른쪽 눈 좌표

        # 두 눈의 중점 계산
        eye_center = np.array([(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2], dtype=np.float32)       # 두 눈의 중점 계산

        # 두 눈 사이의 각도 계산
        delta_x = right_eye[0] - left_eye[0]    # 두 눈의 x 좌표 차이
        delta_y = right_eye[1] - left_eye[1]        # 두 눈의 y 좌표 차이
        angle = np.arctan(delta_y / delta_x) * 180 / np.pi     # 두 눈 사이의 각도 계산

        # 회전을 위한 변환 행렬 준비
        M = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1)      # 회전 변환 행렬 계산

        # 이미지 회전
        aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))      # 이미지 회전   

        cv2.imwrite('C:/capstonedesign/alignedhi.jpg', aligned_image)     # 이미지 저장
        return aligned_image        # 정렬된 이미지 반환

    return None     # 얼굴 검출 실패 시 None 반환

image = cv2.imread('C:/capstonedesign/iu.jpg')  # 이미지 로드
aligned_image = align_face(image)   # 얼굴 정렬

if aligned_image is not None:
    print("얼굴 정렬 및 저장 완료")     # 정렬된 이미지 저장 완료
else:
    print("얼굴을 검출하지 못했습니다.")        # 얼굴 검출 실패
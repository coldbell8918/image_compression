import cv2
from PIL import Image

def convert_jp2_to_jpg(input_path, output_path):
    """
    JPEG 2000 (.jp2) 이미지를 JPEG (.jpg)로 변환.

    Args:
        input_path (str): JPEG 2000 파일 경로.
        output_path (str): 변환된 JPEG 파일 저장 경로.
    """
    # JPEG 2000 파일 읽기
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError(f"이미지 파일을 읽을 수 없습니다: {input_path}")

    # JPEG 형식으로 저장
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"파일이 변환되었습니다: {input_path} -> {output_path}")


# 입력 및 출력 경로 설정
input_jp2 = "/home/park/IC/iclr_17_compression/kodak_result/jpeg2000_result_4096.jp2"  # JPEG 2000 파일 경로
output_jpg = "/home/park/IC/iclr_17_compression/kodak_result/jpeg2000_result_4096.jpg"  # 변환된 JPEG 파일 경로

# 변환 함수 호출
convert_jp2_to_jpg(input_jp2, output_jpg)

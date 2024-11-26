import cv2
import os

def compress_to_bpp(input_path, output_path, bpp, width, height):
    # 이미지 읽기
    img = cv2.imread(input_path)
    
    # 픽셀 수 계산
    num_pixels = width * height
    
    # 목표 파일 크기 계산 (Bytes 단위)
    target_file_size = (bpp * num_pixels) / 8  # Bytes로 변환
    
    # 초기 품질 설정 (중간값)
    quality = 50
    
    # 압축 실행 및 파일 크기 비교
    while True:
        # 임시 파일로 저장
        temp_path = "temp_compressed.jpg"
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        # 현재 파일 크기 측정
        current_file_size = os.path.getsize(temp_path)
        
        # 파일 크기가 목표에 근접하면 종료
        if abs(current_file_size - target_file_size) < 1024:  # 1KB 오차 허용
            os.rename(temp_path, output_path)
            break
        
        # 품질 조정
        if current_file_size > target_file_size:
            quality -= 1  # 파일이 크면 품질 낮춤
        else:
            quality += 1  # 파일이 작으면 품질 높임

        if quality < 1 or quality > 100:  # 품질 범위 벗어나면 종료
            raise ValueError("Cannot achieve the desired bpp with JPEG compression.")
    
    print(f"Image saved to {output_path} with bpp={bpp:.2f}, quality={quality}")

# 입력 이미지 경로, 출력 경로, 목표 bpp 설정
input_image = "/home/park/IC/CompressionData/kodak/kodim15.png"
output_image = "compressed.jpg"
desired_bpp = 0.5  # 원하는 bpp (예: 0.5 bits per pixel)

# 이미지 해상도 (예제: 1920x1080)
width, height = 1920, 1080

compress_to_bpp(input_image, output_image, desired_bpp, width, height)

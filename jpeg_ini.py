from PIL import Image

def compress_image(input_path, output_path, target_size_kb, quality_range=(10, 95)):
    """
    PNG 이미지를 JPEG로 압축하는 함수.

    Args:
        input_path (str): 원본 이미지 경로.
        output_path (str): 압축된 JPEG 저장 경로.
        target_size_kb (float): 목표 크기 (킬로바이트).
        quality_range (tuple): JPEG 품질의 최소 및 최대 값.

    Returns:
        None
    """
    # 이미지 열기
    img = Image.open(input_path).convert("RGB")

    # 이진 탐색으로 JPEG 품질 찾기
    low, high = quality_range
    best_quality = low
    while low <= high:
        quality = (low + high) // 2
        # 임시로 이미지를 저장
        img.save(output_path, "JPEG", quality=quality)
        # 저장된 파일 크기 확인
        file_size_kb = os.path.getsize(output_path) / 1024  # 바이트를 KB로 변환

        if file_size_kb <= target_size_kb:
            best_quality = quality  # 목표 크기를 만족하는 품질 저장
            low = quality + 1       # 품질을 높여 더 압축된 크기 탐색
        else:
            high = quality - 1      # 품질을 낮춰 크기 줄이기

    # 최적 품질로 최종 저장
    img.save(output_path, "JPEG", quality=best_quality)
    print(f"압축 완료: {output_path} (크기: {os.path.getsize(output_path) / 1024:.2f} KB, 품질: {best_quality})")

# 실행 코드
import os

input_path = "/home/park/IC/CompressionData/kodak/kodim15.png"  # 원본 PNG 이미지 경로
output_path = "/home/park/IC/iclr_17_compression/kodak_result/jpeg_result.png"  # 저장할 JPEG 경로
target_size_kb = 415.5  # 목표 크기 (킬로바이트)

compress_image(input_path, output_path, target_size_kb)

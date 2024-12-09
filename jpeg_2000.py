import cv2
import os
from PIL import Image
import numpy as np
from pytorch_msssim import ms_ssim
import torch
from torchvision import transforms

def compress_to_target_bpp(original_path, output_path, target_bpp):
    """
    JPEG 2000 압축을 목표 BPP로 수행.
    
    Args:
        original_path (str): 원본 이미지 경로.
        output_path (str): 압축된 JPEG 2000 이미지 저장 경로.
        target_bpp (float): 목표 Bits Per Pixel (BPP).

    Returns:
        float: 최종 BPP.
    """
    # 원본 이미지 로드
    img = cv2.imread(original_path)
    height, width, _ = img.shape
    total_pixels = height * width
    
    # 목표 파일 크기 계산
    target_file_size = (target_bpp * total_pixels) / 8  # Bytes로 변환

    # JPEG 2000 압축 품질 초기값 설정
    compression_ratio = 37  # 압축률 초기값
    step = 5  # 압축률 조정 간격

    # 반복적으로 파일 크기 조정
    while True:
        # 임시 파일 저장
        temp_path = "temp_output.jp2"
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio])
        os.rename(temp_path, output_path)
        break

        # 현재 파일 크기 확인
        # file_size = os.path.getsize(temp_path)

        # # 목표 파일 크기에 근접하면 저장 후 종료
        # # if abs(file_size - target_file_size) < 2048:  # 오차 허용 (1KB)
        # if file_size < target_file_size:
        #     os.rename(temp_path, output_path)
        #     break

        # # 압축률 조정
        # if file_size > target_file_size:
        #     compression_ratio += step  # 파일이 크면 압축 강화
        # else:
        #     compression_ratio -= step  # 파일이 작으면 압축 완화

        # # 압축률 범위 제한
        # if compression_ratio < 1:
        #     # 압축률 최저로 설정 후 마지막 시도
        #     compression_ratio = 1
        #     cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression_ratio])
        #     final_size = os.path.getsize(temp_path)
        #     if abs(final_size - target_file_size) < 1024:
        #         os.rename(temp_path, output_path)
        #         break
        #     raise ValueError("Cannot compress to the desired BPP even at the highest compression.")

    # 최종 BPP 계산
    final_bpp = (os.path.getsize(output_path) * 8) / total_pixels
    return final_bpp

def calculate_ms_ssim(original_path, compressed_path):
    """
    Multi-Scale SSIM 계산.
    
    Args:
        original_path (str): 원본 이미지 경로.
        compressed_path (str): 압축된 이미지 경로.
    
    Returns:
        float: MS-SSIM 값.
    """
    # 원본 및 압축 이미지 로드 (RGB로 변환)
    original_img = cv2.cvtColor(cv2.imread(original_path), cv2.COLOR_BGR2RGB)
    compressed_img = cv2.cvtColor(cv2.imread(compressed_path), cv2.COLOR_BGR2RGB)

    # 이미지 텐서로 변환
    original_tensor = torch.tensor(original_img / 255.0).permute(2, 0, 1).unsqueeze(0)
    compressed_tensor = torch.tensor(compressed_img / 255.0).permute(2, 0, 1).unsqueeze(0)

    # MS-SSIM 계산
    ms_ssim_value = ms_ssim(original_tensor, compressed_tensor, data_range=1.0).item()
    return ms_ssim_value

def calculate_psnr(input_tensor, recon_tensor):
    mse = torch.mean((input_tensor - recon_tensor) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

# 원본 이미지 경로 및 압축 이미지 경로
original_path = "/home/park/IC/CompressionData/kodak/kodim15.png"  # 원본 이미지
output_path = "/home/park/IC/iclr_17_compression/kodak_result/jpeg2000_result_4096.jp2"  # 압축된 이미지 (.jp2 확장자)

# 목표 BPP 설정
target_bpp = 0.1924

# 압축 수행
final_bpp = compress_to_target_bpp(original_path, output_path, target_bpp)
ms_ssim_value = calculate_ms_ssim(original_path, output_path)

transform = transforms.Compose([
        transforms.ToTensor()  # 이미지를 Tensor로 변환
    ])

original_image = Image.open(original_path).convert('RGB')
input_tensor = transform(original_image).unsqueeze(0).cuda()  # 배치 차원 추가
input_tensor_cpu = input_tensor.cpu().squeeze(0)

recon_image = Image.open(output_path).convert('RGB')
recon_tensor = transform(recon_image).unsqueeze(0).cuda()  # 배치 차원 추가
recon_tensor_cpu = recon_tensor.cpu().squeeze(0)

final_luma = calculate_psnr(input_tensor_cpu, recon_tensor_cpu)
print(f"압축 완료: 최종 BPP = {final_bpp:.4f}, 최종 Luma PSNR = {final_luma:.4f}, 최종 MS-SSIM = {ms_ssim_value:.4f}")

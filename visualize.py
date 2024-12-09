import os
import torch
from torchvision import transforms
from model import ImageCompressor, load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.color import rgb2ycbcr
from pytorch_msssim import ms_ssim

def calculate_psnr(input_tensor, recon_tensor):
    mse = torch.mean((input_tensor - recon_tensor) ** 2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()

def calculate_chroma_psnr(input_tensor, recon_tensor):
    """
    Chroma(색상) 채널의 PSNR 계산 (YCbCr 변환 후).
    """
    input_ycbcr = rgb2ycbcr(input_tensor.permute(1, 2, 0).numpy())
    recon_ycbcr = rgb2ycbcr(recon_tensor.permute(1, 2, 0).numpy())

    # Cb, Cr 채널 추출
    input_cb = input_ycbcr[:, :, 1] / 255.0
    input_cr = input_ycbcr[:, :, 2] / 255.0
    recon_cb = recon_ycbcr[:, :, 1] / 255.0
    recon_cr = recon_ycbcr[:, :, 2] / 255.0

    # PSNR 계산
    cb_psnr = 10 * np.log10(1 / np.mean((input_cb - recon_cb) ** 2))
    cr_psnr = 10 * np.log10(1 / np.mean((input_cr - recon_cr) ** 2))

    return cb_psnr, cr_psnr

def visualize(pretrained_model_path, input_image_path, output_dir=None):
    """
    원본 이미지와 복원 이미지를 시각화하며, 압축 성능 지표를 출력.
    """
    # 모델 불러오기
    model = ImageCompressor()
    global_step = load_model(model, pretrained_model_path)
    model = model.cuda()
    model.eval()  # 평가 모드 설정

    # 입력 이미지 로드 및 전처리
    transform = transforms.Compose([
        transforms.ToTensor()  # 이미지를 Tensor로 변환
    ])
    input_image = Image.open(input_image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).cuda()  # 배치 차원 추가

    # 모델을 사용해 복원 이미지 생성
    with torch.no_grad():
        recon_image, _, bpp = model(input_tensor)
        recon_image = recon_image.clamp(0, 1).cpu().squeeze(0)  # Tensor에서 배치 차원 제거 (CPU 이동)

    # 압축 성능 계산 (모든 텐서를 CPU에서 처리)
    input_tensor_cpu = input_tensor.cpu().squeeze(0)  # CPU로 이동
    bit_per_pixel = bpp.item()  # BPP(Bit-Per-Pixel)
    byte_per_pixel = bit_per_pixel / 8  # Byte 계산
    psnr_luma = calculate_psnr(input_tensor_cpu, recon_image)  # PSNR (Luma)
    cb_psnr, cr_psnr = calculate_chroma_psnr(input_tensor_cpu, recon_image)  # PSNR (Chroma)
    ms_ssim_value = ms_ssim(input_tensor_cpu.unsqueeze(0), recon_image.unsqueeze(0), data_range=1.0).item()  # MS-SSIM

    # 결과 출력
    print(f"Compression Metrics for {os.path.basename(input_image_path)}:")
    print(f"- Bits per Pixel (bpp): {bit_per_pixel:.4f}")
    print(f"- Bytes per Pixel: {byte_per_pixel:.4f}")
    print(f"- PSNR (Luma): {psnr_luma:.2f} dB")
    print(f"- PSNR (Chroma Cb): {cb_psnr:.2f} dB")
    print(f"- PSNR (Chroma Cr): {cr_psnr:.2f} dB")
    print(f"- MS-SSIM: {ms_ssim_value:.4f}")

    # 복원 이미지 저장 (선택)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "iter_2502000_4096_image.png")
        recon_pil = transforms.ToPILImage()(recon_image)
        recon_pil.save(output_path)

    # 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis("off")
    plt.text(
        0, -30, 
        f"PSNR (Luma): N/A\nPSNR (Chroma Cb): N/A\nPSNR (Chroma Cr): N/A\nMS-SSIM: N/A\nBytes per Pixel: N/A\nBits per Pixel: N/A",
        fontsize=10, ha='left', wrap=True
    )

    # 복원 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(recon_image.permute(1, 2, 0))  # 채널 순서 변경
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.text(
        0, -30,
        f"PSNR (Luma): {psnr_luma:.2f} dB\n"
        f"PSNR (Chroma Cb): {cb_psnr:.2f} dB\n"
        f"PSNR (Chroma Cr): {cr_psnr:.2f} dB\n"
        f"MS-SSIM: {ms_ssim_value:.4f}\n"
        f"Bytes per Pixel: {byte_per_pixel:.4f}\n"
        f"Bits per Pixel: {bit_per_pixel:.4f}",
        fontsize=10, ha='left', wrap=True
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 학습된 모델 체크포인트 경로
    pretrained_model_path = "/home/park/IC/iclr_17_compression/checkpoints_4096/baseline/iter_2502000.pth.tar"
    # 원본 비압축 이미지 경로
    input_image_path = "/home/park/IC/CompressionData/kodak/kodim15.png"
    # 복원 이미지를 저장할 디렉토리 (선택 사항)
    output_dir = "/home/park/IC/iclr_17_compression/kodak_result"

    # 시각화 함수 호출
    visualize(pretrained_model_path, input_image_path, output_dir)
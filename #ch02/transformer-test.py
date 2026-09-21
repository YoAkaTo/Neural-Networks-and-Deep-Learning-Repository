import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
from safetensors.torch import load_file

# ========== 修改：模型文件夹名称改为 rmbg_model ==========
model_dir = os.path.join(os.getcwd(), "rmbg_model")
sys.path.append(model_dir)

from rmbg_model.briarmbg import BriaRMBG


def preprocess_image(pil_img, model_input_size=(1024, 1024)):
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(model_input_size, Image.Resampling.BILINEAR)
    im_np = np.array(pil_img, dtype=np.float32) / 255.0
    im_tensor = torch.from_numpy(im_np).permute(2, 0, 1).unsqueeze(0)
    im_tensor = im_tensor * 2.0 - 1.0
    return im_tensor


def postprocess_mask(pred_mask, orig_size):
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(pred_mask).resize(orig_size, Image.Resampling.BILINEAR)
    return np.array(mask_pil)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = BriaRMBG()
    weight_path = os.path.join(model_dir, "model.safetensors")
    # 记得自己去RMBG1.4下载这个文件到rmbg_model哦，太大了没上传到这个仓库
    # 算了我人这么好，贴上链接吧，就是不知道你用的时候还能打开不 https://hf-mirror.com/briaai/RMBG-1.4/tree/main

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件：{weight_path}")

    state_dict = load_file(weight_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_video = "dog1.mp4"
    output_video = "rmbg_dog1_output.mp4"

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"找不到视频：{input_video}")

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_count = 0
    save_frames = [20, 60, 100]
    orig_size = (width, height)

    print("开始视频逐帧抠图...")

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            img_tensor = preprocess_image(pil_img).to(device)
            output = model(img_tensor)
            pred_mask = torch.sigmoid(output[0][0])
            mask_np = postprocess_mask(pred_mask, orig_size)

            white_bg = np.full_like(frame, 255)
            mask_3ch = np.repeat(mask_np[:, :, np.newaxis], 3, axis=-1)
            out_frame = np.where(mask_3ch > 127, frame, white_bg)

            writer.write(out_frame)

            if frame_count in save_frames:
                save_name = f"rmbg_frame_{frame_count}.png"
                ##cv2.imwrite(save_name, out_frame)
                ret, buf = cv2.imencode(".png", out_frame)
                if ret:
                    buf.tofile(save_name)
                print(f"保存关键帧：{save_name}")

    cap.release()
    writer.release()
    print(f"✅ 视频处理完成，输出文件：{output_video}")


if __name__ == "__main__":
    main()

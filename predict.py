import os
import time
from PIL import Image
from mask_generator import RetinalCoNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":

    model = RetinalCoNet()

    # 定义是否进行目标的像素点计数与比例计算
    count = False
    # 定义区分的种类
    name_classes = ["background", "fish"]

    # 指定用于检测的图片的文件夹路径
    dir_origin_path = "./VOCdevkit/VOC2007/test_image"
    # 指定检测完图片的保存路径
    dir_save_path = "./VOCdevkit/VOC2007/1"

    # 确保保存文件夹存在
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    # 获取指定文件夹中的所有图片文件名
    img_names = os.listdir(dir_origin_path)

    # 添加FPS计算相关变量
    total_time = 0.0  # 记录总推理时间
    processed_count = 0  # 记录成功处理的图片数量

    print(f"开始处理 {len(img_names)} 张图片...")

    for img_name in img_names:
        # 检查文件扩展名是否为常见的图片格式
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            # 构建完整的图片文件路径
            image_path = os.path.join(dir_origin_path, img_name)
            try:
                # 打开图片
                image = Image.open(image_path)
            except Exception as e:
                print(f"Open Error for {img_name}! Error: {e}. Try again!")
                continue
            else:
                try:
                    # 记录推理开始时间
                    start_time = time.time()

                    # 调用Unet模型进行图片检测
                    r_image = model.detect_image(image, count=count, name_classes=name_classes)

                    # 记录推理结束时间
                    end_time = time.time()

                    # 计算并累加推理时间
                    inference_time = end_time - start_time
                    total_time += inference_time
                    processed_count += 1

                    # 计算当前FPS
                    current_fps = 1.0 / inference_time

                    # 保存处理后的图片
                    r_image.save(os.path.join(dir_save_path, img_name))

                    # 打印处理信息
                    print(f"Processed {img_name} | Time: {inference_time:.4f}s | FPS: {current_fps:.2f}")

                except Exception as e:
                    print(f"Processing Error for {img_name}! Error: {e}")

    # 计算平均FPS
    if processed_count > 0:
        average_fps = processed_count / total_time
        print("\n" + "=" * 50)
        print(f"处理完成! 总共处理图片: {processed_count}张")
        print(f"总推理时间: {total_time:.4f}秒")
        print(f"平均推理速度: {average_fps:.2f} FPS")
        print(f"每张图片平均处理时间: {total_time / processed_count:.4f}秒")
        print("=" * 50)
    else:
        print("没有成功处理的图片!")
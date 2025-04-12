import os
from PIL import Image

def convert_images(input_dir, output_dir, output_format='PNG'):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # 构建完整文件路径
            input_path = os.path.join(input_dir, filename)
            
            # 生成输出文件名（保持相同名称，仅修改扩展名）
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.{output_format.lower()}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # 打开图片并转换
                with Image.open(input_path) as img:
                    # 保持原始分辨率
                    img.save(output_path, format=output_format)
                print(f"转换成功: {filename} -> {output_filename}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_directory = "test_set"  # 替换为你的输入目录
    output_directory = "test_set"  # 替换为你的输出目录
    target_format = "WEBP"  # 可以改为"WEBP", "BMP"等
    
    convert_images(input_directory, output_directory, target_format)

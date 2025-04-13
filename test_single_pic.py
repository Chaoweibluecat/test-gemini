import os
import time
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import pandas as pd

# 初始化客户端
client = genai.Client(api_key="AIzaSyDw4yQR6Lhzkr4XP1BNrkZ5PkcoBeNWqm0")

# 测试数据配置
cat_images = [
    {"name": "cat_640*440", "res": (640, 440), "formats": ["png", "jpg", "webp"]},
    {"name": "cat_1920*1321", "res": (1920, 1321), "formats": ["png", "jpg", "webp"]},
    {"name": "cat_2400*1651", "res": (2400, 1651), "formats": ["png", "jpg", "webp"]},
    {"name": "cat_5026*3458", "res": (5026, 3458), "formats": ["jpg", "webp"]}
]

output_dir = "single_image_results"
os.makedirs(output_dir, exist_ok=True)

# 结果存储

def generate_with_retry(prompt, image, model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image],
            )
            return response
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指数退避
    return None

def run_model(model_name):
    results = []

    # 开始测试
    for cat in cat_images:
        for fmt in cat["formats"]:
            cat_path = f"cats/{cat['name']}.{fmt}"
            if not os.path.exists(cat_path):
                print(f"Warning: {cat_path} not found, skipping")
                continue
                
            print(f"\nProcessing: {cat_path}")
            
            try:
                # 1. 计算单个图片的Token
                cat_file = client.files.upload(file=cat_path)
                
                count_response = client.models.count_tokens(
                    model=model_name,
                    contents=["Describe this image", cat_file]
                )
                
                # 2. 实际生成
                response = generate_with_retry(
                    "Describe this image",
                    Image.open(cat_path), 
                    model_name=model_name
                )
                
                # 3. 记录结果
                usage = response.usage_metadata
                result = {
                    "image_name": cat['name'],
                    "image_format": fmt,
                    "image_resolution": f"{cat['res'][0]}x{cat['res'][1]}",
                    "count_tokens": count_response.total_tokens,
                    "actual_tokens": usage.prompt_token_count,
                    "text_tokens": next(m.token_count for m in usage.prompt_tokens_details if m.modality.name == "TEXT"),
                    "image_tokens": next(m.token_count for m in usage.prompt_tokens_details if m.modality.name == "IMAGE"),
                    "output_tokens": usage.candidates_token_count or 0,
                    "status": "success"
                }                    
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {cat_path}: {str(e)}")
                results.append({
                    "image_name": cat['name'],
                    "image_format": fmt,
                    "image_resolution": f"{cat['res'][0]}x{cat['res'][1]}",
                    "status": f"failed: {str(e)}"
                })
            finally:
                # 添加延迟避免速率限制
                time.sleep(1)

    # 生成结果表格
    df = pd.DataFrame(results)
    report_path = os.path.join(output_dir, f"{model_name}_single_image_token_report.csv")
    df.to_csv(report_path, index=False)

    print(f"\n测试完成！结果已保存到: {report_path}")
    print(df[['image_name', 'image_format', 'count_tokens', 'actual_tokens', 'status']])

for m in ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-1.5-flash"]:
    run_model(m)
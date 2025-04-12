# 创建虚拟环境（推荐放到项目目录）
python3 -m venv .venv

# 激活环境
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 现在安全安装包
pip install google-genai

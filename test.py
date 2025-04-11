import os

model_path = './model/全光谱/onedcnnattention/model.pth'  # 修改为你想保存的文件名
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # 确保目录存在


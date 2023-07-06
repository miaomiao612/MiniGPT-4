from predict import Predictor

# 准备输入图像
image_path = "D:\\MiniGPT-4\\finetune_data\\image\\82.jpg"  # 替换为你的输入图像路径

# 创建Predictor实例
predictor = Predictor()

message = "describe the image."
num_beams = 1
temperature = 0.75
max_new_tokens = 500

# 调用predict方法进行预测
result = predictor.predict(image=image_path, message=message, num_beams=num_beams, temperature=temperature, max_new_tokens=max_new_tokens)

# 使用结果
print("Prediction result:", result)
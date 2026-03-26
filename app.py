import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 启用 Hugging Face 国内镜像

import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer

# 导入你的主厨模型
from models.model import ContrastiveHateSpeechModel

app = Flask(__name__)

# ==========================================
# 1. 全局加载模型和分词器（只需加载一次）
# ==========================================
print("正在启动 AI 引擎，请稍候...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = ContrastiveHateSpeechModel('bert-base-chinese').to(device)

# 加载你刚才本地试跑生成的模型权重
model_path = 'checkpoints/best_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功加载本地权重: {model_path}")
else:
    print("【警告】未找到训练好的权重，模型目前是瞎猜状态！")

model.eval()  # 切换到预测模式


# ==========================================
# 2. 定义前后端交互的 API 接口
# ==========================================

# 访问主页时，返回 HTML 页面
@app.route('/')
def home():
    return render_template('index.html')


# 接收前端发送的文本，返回预测结果
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': '文本不能为空！'})

    # 把文字变成模型认识的张量
    inputs = tokenizer(
        text, return_tensors='pt', max_length=128,
        truncation=True, padding='max_length'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 不计算梯度，直接推理
    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        # 将输出转换为百分比概率
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_label = torch.argmax(probabilities).item()
        confidence = probabilities[pred_label].item()

    # 封装要返回给前端的数据包 (JSON)
    result = {
        'label': int(pred_label),
        'label_name': '不当言论 🚨' if pred_label == 1 else '正常言论 ✅',
        'confidence': f"{confidence * 100:.2f}%"
    }
    return jsonify(result)


if __name__ == '__main__':
    # 启动服务器，端口设为 5000
    app.run(debug=True, port=5000)
from flask import Flask, request, jsonify, render_template
from inference import HateSpeechInference

app = Flask(__name__)

# ==========================================
# 1. 初始化推理引擎
# ==========================================
inference = HateSpeechInference(model_path='checkpoints/best_model.pth')


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
    text = data.get('text', '')

    result = inference.predict(text)

    if 'error' in result:
        return jsonify(result)

    # 格式化返回给前端
    return jsonify({
        'label': result['label'],
        'label_name': result['label_name'],
        'confidence': f"{result['confidence'] * 100:.2f}%"
    })


if __name__ == '__main__':
    # 启动服务器，端口设为 5000
    app.run(debug=True, port=5000)
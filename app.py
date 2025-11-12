from flask import Flask, render_template, request, jsonify
from load_model import generate_response

app = Flask(__name__)

@app.route('/')
def index():
    """
    Webブラウザからトップページにアクセスがあった場合に
    'templates/editable.html' を表示します。
    """
    return render_template('editable.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    フロントエンドからのPOSTリクエストを処理するエンドポイント。
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        history = data.get('history', [])
        response = generate_response(user_message, history)

        return jsonify({'response': response})
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

# load_model.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_directml # DMLデバイスを有効化

# 1. グローバルなデバイス設定
try:
    # dmlデバイスが利用可能かチェック
    DEVICE = torch_directml.device(0) 
    print(f"✅ Llama Model will use device: {DEVICE}")
    
except Exception:
    # dmlデバイスが見つからなければCPUへフォールバック
    DEVICE = torch.device("cpu")
    print("❌ Llama Model will use device: cpu. DirectMLデバイスが見つかりませんでした。")

# 2. グローバルなモデルとトークナイザーのロード（一度だけ実行）
# モデルはVRAMを大量に消費するため、起動時に時間がかかる場合があります
try:
    print("Loading Llama-3.2-1B-Instruct model...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # モデルをGPU/DMLデバイスに転送
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model on {DEVICE}: {e}")
    # ロード失敗時のフォールバック処理が必要であれば追加

def generate_response(user_message: str, history: list) -> str:
    """
    ユーザーメッセージを受け取り、Llamaモデルで応答を生成する
    """
    
    # 既存の履歴をメッセージ形式に変換
    messages = []
    for item in history:
        # app.py側で履歴を管理する方法に依存しますが、ここではダミー処理
        if item.get("sender") == "user":
             messages.append({"role": "user", "content": item.get("content")})
        elif item.get("sender") == "bot":
             messages.append({"role": "assistant", "content": item.get("content")})
             
    # 新しいユーザーメッセージを追加
    messages.append({"role": "user", "content": user_message})

    # テンプレートを適用し、入力テンソルを生成
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(DEVICE) # 入力も同じデバイスへ転送

    # 生成を実行
    outputs = model.generate(**inputs, max_new_tokens=200)
    
    # モデルの応答部分のみをデコードして返す
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response

# テスト用の実行コード（app.pyがインポートしない場合にのみ実行）
if __name__ == '__main__':
    print("--- 起動テスト ---")
    dummy_history = []
    test_response = generate_response("Hello, what is your name?", dummy_history)
    print(f"Test Response: {test_response}")
import os
import sys
from dotenv import load_dotenv

# Dẫn luồng linh lực về thư mục gốc (Add src to path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.openrouter_provider import OpenRouterProvider

def test_openrouter():
    # Nạp linh thạch từ file .env
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    print(f"--- 🔮 Test OpenRouter (Qwen 3.6) ---")
    
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env file. Please check!")
        return

    try:
        # Initialize the provider
        provider = OpenRouterProvider()
        
        # ---------------------------------------------------------
        # 1. Test English Prompt
        # ---------------------------------------------------------
        prompt_en = "Explain the core difference between a simple LLM Chatbot and a ReAct Agent in one concise sentence."
        print(f"\n[English Prompt]\nUser: {prompt_en}")
        print("Assistant: ", end="", flush=True)
        
        # Thử nghiệm tính năng stream (trả chữ về từ từ như đang gõ)
        for chunk in provider.stream(prompt_en):
            print(chunk, end="", flush=True)
        print("\n" + "-"*50)

        # ---------------------------------------------------------
        # 2. Test Vietnamese Prompt (Vietnamese Formal)
        # ---------------------------------------------------------
        prompt_vn = "Xin chào, hãy đóng vai là một chuyên gia tuyển dụng. Xin hãy tóm tắt ngắn gọn 3 kỹ năng cốt lõi nhất mà một lập trình viên Full-stack cần có trong thời đại hiện nay."
        print(f"\n[Vietnamese Formal Prompt]\nUser: {prompt_vn}")
        print("Assistant: ", end="", flush=True)
        
        for chunk in provider.stream(prompt_vn):
            print(chunk, end="", flush=True)
            
        print("\n\n✅ Test completed! Qwen via OpenRouter is working smoothly.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    test_openrouter()
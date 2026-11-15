import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()                      # مقدارهای فایل .env را به متغیرهای محیطی اضافه می‌کند
key = os.environ.get("GEMINI_API_KEY")
if not key:
    sys.exit("GEMINI_API_KEY پیدا نشد. فایل .env را بسازید (نمونه: .env.example).")

question = sys.argv[1]
system = sys.argv[2] if len(sys.argv) > 2 else "تو یک دستیار مفید هستی و کوتاه و ساده جواب می‌دهی."
model = os.environ.get("QUIZ_AI_MODEL", "gemini-3.5-flash-lite")

client = genai.Client(api_key=key)
response = client.models.generate_content(
    model=model,
    contents=question,                                              # سؤال کاربر
    config=types.GenerateContentConfig(system_instruction=system),  # شخصیت و قواعد
)

print(response.text)
usage = response.usage_metadata
print(f"\n[توکن ورودی: {usage.prompt_token_count} | توکن خروجی: {usage.candidates_token_count}]")

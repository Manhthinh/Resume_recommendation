import argparse
import json
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(gap_result: dict, user_question: str) -> str:
    return f"""
Bạn là chatbot tư vấn nghề nghiệp và phân tích CV cho nhóm ngành Data/AI.

Nhiệm vụ của bạn:
- Trả lời tự nhiên, rõ ràng, dễ hiểu bằng tiếng Việt.
- Không bịa thông tin ngoài dữ liệu đã cho.
- Tập trung vào tư vấn nghề nghiệp và cải thiện CV.
- Nếu CV chưa phù hợp với nhóm Data/AI, hãy nói rõ điều đó một cách lịch sự.

Hãy trả lời theo đúng 5 mục sau:
1. Mức độ phù hợp
2. Điểm mạnh hiện tại
3. Điểm còn thiếu
4. Kỹ năng nên phát triển tiếp
5. Nghề phù hợp hơn nếu có

Dữ liệu phân tích CV:
{json.dumps(gap_result, ensure_ascii=False, indent=2)}

Câu hỏi người dùng:
{user_question}
""".strip()


def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def fallback_response(gap_result: dict, user_question: str) -> str:
    roles = gap_result.get("best_fit_roles", [])
    strengths = gap_result.get("strengths", [])
    missing = gap_result.get("missing_skills", [])
    plan = gap_result.get("development_plan", [])
    domain_fit = gap_result.get("domain_fit", "unknown")

    lines = []

    if roles:
        lines.append(f"1. Mức độ phù hợp")
        if domain_fit == "low":
            lines.append(
                f"CV của bạn hiện chưa thể hiện mức độ phù hợp cao với nhóm Data/AI. "
                f"Vai trò gần nhất hiện tại là {roles[0]}."
            )
        else:
            lines.append(f"CV của bạn hiện phù hợp nhất với vị trí {roles[0]}.")
    else:
        lines.append("1. Mức độ phù hợp")
        lines.append("Chưa xác định được vai trò phù hợp rõ ràng.")

    lines.append("\n2. Điểm mạnh hiện tại")
    if strengths:
        for s in strengths[:5]:
            lines.append(f"- {s}")
    else:
        lines.append("- Chưa có kỹ năng nổi bật khớp rõ với nhóm nghề mục tiêu.")

    lines.append("\n3. Điểm còn thiếu")
    if missing:
        for m in missing[:5]:
            lines.append(f"- {m}")
    else:
        lines.append("- Chưa phát hiện thiếu hụt nổi bật.")

    lines.append("\n4. Kỹ năng nên phát triển tiếp")
    if plan:
        for p in plan[:5]:
            lines.append(f"- {p}")
    else:
        lines.append("- Nên bổ sung thêm project thực tế và kỹ năng theo role mục tiêu.")

    lines.append("\n5. Nghề phù hợp hơn nếu có")
    if len(roles) >= 2:
        lines.append(f"- Bạn cũng có thể cân nhắc: {', '.join(roles[1:3])}")
    elif roles:
        lines.append(f"- Vai trò nên ưu tiên hiện tại là: {roles[0]}")
    else:
        lines.append("- Cần thêm thông tin CV để gợi ý nghề phù hợp hơn.")

    lines.append("\n[Ghi chú] Hệ thống đang dùng chế độ dự phòng vì chưa gọi được Ollama/Llama 3.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap_result", required=True, help="Path to gap analysis result JSON")
    parser.add_argument("--question", required=True, help="User question for the chatbot")
    args = parser.parse_args()

    gap_result_path = Path(args.gap_result)
    if not gap_result_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file gap result: {gap_result_path}")

    gap_result = load_json(str(gap_result_path))
    prompt = build_prompt(gap_result, args.question)

    try:
        answer = ask_ollama(prompt)
        if not answer:
            answer = fallback_response(gap_result, args.question)
    except Exception as e:
        print(f"[Warning] Không gọi được Ollama: {e}")
        answer = fallback_response(gap_result, args.question)

    print("\n===== CHATBOT ANSWER =====\n")
    print(answer)


if __name__ == "__main__":
    main()
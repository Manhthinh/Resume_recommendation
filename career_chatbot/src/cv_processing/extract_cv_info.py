# src/cv_processing/extract_cv_info.py

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import fitz  # PyMuPDF
from docx import Document


SKILL_VOCAB = [
    "python", "sql", "excel", "power bi", "tableau", "pandas", "numpy",
    "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
    "spark", "hadoop", "airflow", "etl", "nlp", "computer vision",
    "statistics", "data visualization", "dashboard", "git", "docker",
    "linux", "mysql", "postgresql", "mongodb", "aws", "azure", "gcp",
    "llm", "rag", "langchain", "streamlit", "flask", "fastapi"
]

ROLE_KEYWORDS = {
    "Data Analyst": [
        "data analyst", "business analyst", "bi analyst", "reporting", "dashboard"
    ],
    "Data Engineer": [
        "data engineer", "etl", "pipeline", "data warehouse", "airflow", "spark"
    ],
    "AI Engineer": [
        "ai engineer", "machine learning engineer", "ml engineer", "deployment", "llm"
    ],
    "AI Researcher": [
        "ai researcher", "research scientist", "research assistant", "paper", "experiment"
    ],
}

EDUCATION_KEYWORDS = [
    "university", "college", "đại học", "cao đẳng", "bachelor", "master",
    "cử nhân", "thạc sĩ", "khoa học dữ liệu", "data science", "computer science",
    "information technology", "công nghệ thông tin"
]


def read_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    pages = [page.get_text() for page in doc]
    return "\n".join(pages)


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_cv_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    if ext == ".docx":
        return read_docx(file_path)
    if ext == ".txt":
        return read_txt(file_path)
    raise ValueError(f"Unsupported CV format: {ext}")


def normalize_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-().]{8,}\d)", text)
    return match.group(0).strip() if match else ""


def extract_skills(text: str) -> List[str]:
    lowered = text.lower()
    found = []

    for skill in SKILL_VOCAB:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, lowered):
            found.append(skill)

    # loại trùng giữ thứ tự
    unique = []
    seen = set()
    for item in found:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def guess_target_role(text: str, skills: List[str]) -> str:
    lowered = text.lower()

    role_scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in lowered:
                score += 2
        for skill in skills:
            if skill.lower() in " ".join(keywords):
                score += 1
        role_scores[role] = score

    best_role = max(role_scores, key=role_scores.get)
    return best_role if role_scores[best_role] > 0 else "Unknown"


def extract_education_signals(text: str) -> List[str]:
    lowered = text.lower()
    found = [kw for kw in EDUCATION_KEYWORDS if kw in lowered]
    # loại trùng
    result = []
    seen = set()
    for item in found:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def guess_experience_years(text: str) -> str:
    lowered = text.lower()

    # tìm pattern dạng 1 year / 2 years / 3 năm
    patterns = [
        r"(\d+)\+?\s*(?:years|year)",
        r"(\d+)\+?\s*năm",
    ]
    values = []
    for pattern in patterns:
        matches = re.findall(pattern, lowered)
        for m in matches:
            try:
                values.append(int(m))
            except ValueError:
                pass

    if values:
        return str(max(values))

    # heuristic cơ bản
    if "intern" in lowered or "fresher" in lowered or "sinh viên" in lowered:
        return "0"

    return "Unknown"


def summarize_projects(text: str) -> List[str]:
    # bản MVP: chỉ lấy vài câu chứa project / dự án
    candidates = re.split(r"[.\n]", text)
    result = []
    for line in candidates:
        line_clean = line.strip()
        lowered = line_clean.lower()
        if any(k in lowered for k in ["project", "dự án", "dashboard", "analysis", "model"]):
            if 10 <= len(line_clean) <= 200:
                result.append(line_clean)

    # loại trùng
    unique = []
    seen = set()
    for item in result:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:5]


def extract_cv_info(file_path: str) -> Dict:
    raw_text = load_cv_text(file_path)
    cleaned_text = normalize_text(raw_text)

    skills = extract_skills(cleaned_text)
    target_role = guess_target_role(cleaned_text, skills)
    education_signals = extract_education_signals(cleaned_text)
    experience_years = guess_experience_years(cleaned_text)
    projects = summarize_projects(raw_text)

    result = {
        "file_name": os.path.basename(file_path),
        "email": extract_email(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "skills": skills,
        "target_role": target_role,
        "experience_years": experience_years,
        "education_signals": education_signals,
        "projects": projects,
        "raw_text_preview": cleaned_text[:1000],
    }
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cv_path", required=True, help="Path to CV file (.pdf, .docx, .txt)")
    parser.add_argument("--output_path", default="", help="Optional path to save extracted JSON")
    args = parser.parse_args()

    result = extract_cv_info(args.cv_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON to: {args.output_path}")


if __name__ == "__main__":
    main()
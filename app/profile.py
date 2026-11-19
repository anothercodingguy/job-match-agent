"""Resume text -> structured Profile.

The model is asked for JSON, and the result is validated with Pydantic. If
validation fails, the errors are sent back to the model once for a repair. If
the model is unavailable or still wrong, a keyword-based profile is returned
instead, so the rest of the app keeps working.
"""
import json
import re

from pydantic import ValidationError

from .llm import LLM, LLMError
from .schemas import Profile

MAX_RESUME_CHARS = 12000

SYSTEM_PROMPT = """You extract a candidate profile from resume text.
Return one JSON object with exactly these keys:
  "headline": short professional headline, no personal name (string)
  "years_experience": total work experience in years, internships included (number, e.g. 0.7)
  "seniority": one of "intern", "junior", "mid", "senior", "lead"
  "skills": up to 25 concrete skills and tools (array of strings)
  "target_roles": up to 5 job titles this person fits (array of strings)
  "locations": cities or countries the resume mentions (array of strings)
  "remote_ok": true unless the resume says otherwise (boolean)
  "summary": at most 60 words (string)
Use only facts from the resume. Do not invent skills or experience."""

KNOWN_SKILLS = [
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "SQL", "Bash",
    "React", "Next.js", "Node.js", "Express", "FastAPI", "Flask", "Django", "Spring Boot",
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Qdrant", "Elasticsearch", "Kafka",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Linux", "Git", "CI/CD", "Terraform",
    "PyTorch", "TensorFlow", "scikit-learn", "Pandas", "NumPy", "LLM", "RAG", "Embeddings",
    "OpenTelemetry", "GraphQL", "REST", "HTML", "CSS", "Tailwind", "Figma", "Excel",
    "Tableau", "Power BI", "Salesforce", "SEO", "Selenium", "PyTest", "Jest",
]


def fallback_profile(resume_text: str) -> Profile:
    """Keyword matching only. Used when the model cannot be reached."""
    skills = [
        skill
        for skill in KNOWN_SKILLS
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(skill)}(?![A-Za-z0-9+#])", resume_text, re.IGNORECASE)
    ]
    return Profile(skills=skills, source="fallback")


def extract_profile(llm: LLM | None, resume_text: str) -> Profile:
    text = resume_text.strip()[:MAX_RESUME_CHARS]
    if llm is None:
        return fallback_profile(text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Resume:\n{text}"},
    ]
    for _ in range(2):  # first try, then one repair
        try:
            data = llm.json(messages)
        except LLMError:
            break
        try:
            return Profile.model_validate({**data, "source": "llm"})
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            messages += [
                {"role": "assistant", "content": json.dumps(data)},
                {
                    "role": "user",
                    "content": f"That JSON failed validation ({problems}). Return the corrected JSON object only.",
                },
            ]
    return fallback_profile(text)

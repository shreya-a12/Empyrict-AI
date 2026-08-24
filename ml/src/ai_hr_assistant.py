import os
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(
    os.path.join(current_dir, "..", "..")
)

env_path = os.path.join(
    project_root,
    ".env"
)

load_dotenv(env_path)


# ============================================================
# LOAD GROQ API KEY
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. "
        "Please add it to your .env file."
    )


# ============================================================
# INITIALIZE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=groq_api_key
)


# ============================================================
# TEST GROQ CONNECTION
# ============================================================

def test_groq_connection():

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant for an HR "
                    "Employee Attrition Prediction System."
                )
            },
            {
                "role": "user",
                "content": (
                    "Say hello and confirm that the "
                    "AI HR Assistant is working."
                )
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# AI HR ANALYSIS
# ============================================================

def generate_hr_analysis(
    employee_data,
    prediction_result
):

    prompt = f"""
You are an AI HR Assistant for an Employee Attrition
Prediction and Analytics Platform.

Analyze the employee information and the machine learning
prediction provided below.

IMPORTANT:
- Do not change or question the ML prediction.
- Use the ML prediction only as decision-support information.
- Provide practical HR retention suggestions.
- Do not make unsupported claims about the employee.
- Keep the response professional and concise.

EMPLOYEE DATA:
{employee_data}

ML PREDICTION RESULT:
{prediction_result}

Provide the response in the following format:

EMPLOYEE PRIORITY:
Explain whether this employee requires Low, Medium, or High HR attention.

KEY RISK FACTORS:
List the employee-related factors that may contribute to attrition risk.

RECOMMENDED HR ACTIONS:
Provide 3 to 5 practical actions HR can take.

RETENTION STRATEGY:
Provide a short personalized strategy to help reduce the
employee's likelihood of leaving.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional AI HR decision "
                    "support assistant. Provide practical, "
                    "ethical, and evidence-based HR guidance."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GROQ AI HR ASSISTANT TEST")
    print("=" * 60)

    response = test_groq_connection()

    print("\nAI Response:\n")
    print(response)
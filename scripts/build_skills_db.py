"""
scripts/build_skills_db.py
===========================
Regenerates data/skills.json from the curated dictionary below. This is a
maintenance utility, not part of the runtime app  run it after editing the
SKILLS dict to add/remove a skill, rather than hand-editing the JSON output.

Usage:
    python scripts/build_skills_db.py
"""
import json
from pathlib import Path

# category -> { canonical name: [aliases] }
# Aliases only need to cover spellings that WON'T already be caught by the
# matcher's normalization pass (lowercasing + stripping spaces/hyphens/dots),
# e.g. "TensorFlow" already matches "tensor-flow" and "Tensor Flow" without an
# explicit alias — aliases here are for genuinely different tokens like
# "JS" -> "JavaScript" or "K8s" -> "Kubernetes".
SKILLS = {
    "Programming Languages": {
        "Python": [], "Java": [], "JavaScript": ["JS"], "TypeScript": ["TS"],
        "C++": ["CPP", "C Plus Plus"], "C#": ["C Sharp", "CSharp"], "C": [],
        "Ruby": [], "Go": ["Golang"], "Rust": [], "PHP": [], "Swift": [],
        "Kotlin": [], "Scala": [], "R": ["R Programming", "R Language"],
        "MATLAB": [], "Perl": [], "Dart": [], "Objective-C": ["Objective C"],
        "Elixir": [], "Haskell": [], "Julia": [], "Lua": [], "Groovy": [],
        "VB.NET": ["Visual Basic .NET"], "F#": ["F Sharp"], "Solidity": [],
        "Assembly": [],
    },
    "Web & Frontend": {
        "React": ["React.js", "ReactJS"], "Angular": ["AngularJS"],
        "Vue.js": ["Vue", "VueJS"], "Svelte": [], "Next.js": ["NextJS"],
        "jQuery": [], "HTML5": ["HTML"], "CSS3": ["CSS"], "Sass": ["SCSS"],
        "Tailwind CSS": ["TailwindCSS", "Tailwind"], "Bootstrap": [],
        "Redux": [], "Webpack": [], "Vite": [], "Gatsby": [], "Remix": [],
        "Alpine.js": [], "Material UI": ["MUI"], "Three.js": [], "D3.js": ["D3"],
    },
    "Backend Frameworks": {
        "Django": [], "Flask": [], "FastAPI": [], "Node.js": ["NodeJS", "Node"],
        "Express.js": ["Express", "ExpressJS"], "Spring Boot": ["Spring"],
        "ASP.NET": ["ASP .NET", "DotNet", ".NET"], "Ruby on Rails": ["Rails"],
        "Laravel": [], "NestJS": [], "REST API": ["RESTful API", "REST APIs"],
        "GraphQL": [], "gRPC": [], "WebSockets": [], "Symfony": [],
    },
    "AI / Machine Learning": {
        "Machine Learning": ["ML"], "Deep Learning": ["DL"], "TensorFlow": ["TF"],
        "PyTorch": [], "Scikit-learn": ["Sklearn", "Scikit Learn"], "Keras": [],
        "Natural Language Processing": ["NLP"], "Computer Vision": [],
        "LangChain": [], "LangGraph": [], "CrewAI": [], "AutoGen": [],
        "Haystack": [], "LlamaIndex": [], "DSPy": [],
        "Hugging Face Transformers": ["HuggingFace", "Transformers"],
        "OpenAI SDK": ["OpenAI API"], "Anthropic Claude SDK": ["Claude API"],
        "Google Gemini API": ["Gemini API"], "OpenCV": [], "XGBoost": [],
        "LightGBM": [], "Reinforcement Learning": [], "Generative AI": ["GenAI"],
        "Large Language Models": ["LLM", "LLMs"],
        "Retrieval Augmented Generation": ["RAG"], "Neural Networks": [],
        "Pandas": [], "NumPy": [], "MLOps": [], "Prompt Engineering": [],
        "Model Fine-Tuning": ["Fine-tuning"], "BERT": [], "GPT": [],
        "Vector Databases": [], "Pinecone": [], "Weaviate": [], "Chroma": [],
        "Time Series Forecasting": [], "Recommender Systems": [],
    },
    "Cloud & DevOps": {
        "AWS": ["Amazon Web Services"], "Azure": ["Microsoft Azure"],
        "Google Cloud Platform": ["GCP", "Google Cloud"], "Docker": [],
        "Kubernetes": ["K8s"], "Terraform": [], "Jenkins": [],
        "CI/CD": ["CI CD", "Continuous Integration"], "Ansible": [],
        "GitHub Actions": [], "GitLab CI": [], "CloudFormation": [], "Helm": [],
        "Prometheus": [], "Grafana": [], "Nginx": [], "ArgoCD": [], "Istio": [],
        "OpenShift": [], "Datadog": [], "New Relic": [], "Splunk": [],
    },
    "Databases": {
        "SQL": [], "PostgreSQL": ["Postgres"], "MySQL": [], "MongoDB": ["Mongo"],
        "Redis": [], "Cassandra": [], "Oracle Database": ["Oracle DB"],
        "SQLite": [], "DynamoDB": [], "Elasticsearch": [], "Firebase": [],
        "Snowflake": [], "Neo4j": [], "CockroachDB": [], "MariaDB": [],
        "InfluxDB": [], "Supabase": [], "Redshift": [],
    },
    "Mobile Development": {
        "React Native": [], "Flutter": [], "Android Development": ["Android"],
        "iOS Development": ["iOS"], "SwiftUI": [], "Xamarin": [], "Ionic": [],
        "Jetpack Compose": [],
    },
    "Testing & QA": {
        "Selenium": [], "Pytest": [], "JUnit": [], "Cypress": [], "Jest": [],
        "Postman": [], "TestNG": [], "Unit Testing": [], "Test Automation": [],
        "Playwright": [], "Appium": [], "Mocha": [], "JMeter": [],
    },
    "Operating Systems": {
        "Linux": [], "Windows Server": [], "Unix": [], "macOS": [], "Ubuntu": [],
        "Bash Scripting": ["Bash", "Shell Scripting"], "CentOS": [], "Debian": [],
        "PowerShell": [],
    },
    "Data Engineering": {
        "Apache Spark": ["Spark", "PySpark"], "Hadoop": [],
        "Apache Kafka": ["Kafka"], "Apache Airflow": ["Airflow"],
        "ETL": ["ETL Pipelines"], "Data Warehousing": [], "dbt": [],
        "Apache Beam": [], "BigQuery": [], "Apache NiFi": [], "Presto": ["Trino"],
        "Delta Lake": [],
    },
    "Tools & Collaboration": {
        "Git": [], "GitHub": [], "GitLab": [], "Bitbucket": [], "Jira": [],
        "Confluence": [], "VS Code": ["Visual Studio Code"], "Figma": [],
        "Notion": [], "Slack": [], "Asana": [], "Trello": [], "Miro": [],
        "Linear": [],
    },
    "Soft Skills": {
        "Leadership": [], "Communication": [], "Problem Solving": [],
        "Team Management": [], "Project Management": [], "Agile": ["Agile Methodology"],
        "Scrum": [], "Critical Thinking": [], "Time Management": [],
        "Collaboration": ["Teamwork"], "Mentoring": [], "Public Speaking": [],
        "Adaptability": [], "Conflict Resolution": [], "Decision Making": [],
        "Emotional Intelligence": [], "Negotiation": [], "Creativity": [],
    },
}


def build() -> dict:
    out = {"categories": {}}
    total = 0
    for category, skills in SKILLS.items():
        out["categories"][category] = [
            {"name": name, "aliases": aliases} for name, aliases in skills.items()
        ]
        total += len(skills)
    out["metadata"] = {"total_skills": total, "total_categories": len(SKILLS)}
    return out


if __name__ == "__main__":
    data = build()
    out_path = Path(__file__).resolve().parent.parent / "data" / "skills.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {data['metadata']['total_skills']} skills across "
          f"{data['metadata']['total_categories']} categories -> {out_path}")

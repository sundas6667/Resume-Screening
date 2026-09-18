"""
core/cv_generator.py
======================
Generates realistic sample resume PDFs for demo/testing — lets a user try
the whole pipeline without sourcing their own resumes. Pure content
generation via reportlab (the same library core/report_generator.py uses),
producing PDF bytes that flow through core/parser.py exactly like a real
upload would.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer

from utils.helpers import get_logger

logger = get_logger(__name__)


def _esc(value: object) -> str:
    return xml_escape(str(value))


@dataclass
class SampleExperience:
    title: str
    company: str
    years: str
    bullets: List[str] = field(default_factory=list)


@dataclass
class SampleProject:
    name: str
    description: str
    link: str


@dataclass
class SampleCandidateProfile:
    """Content template for one archetype of sample candidate."""
    key: str
    display_label: str  # shown on a "Generate Sample CV" UI button
    full_name: str
    email: str
    phone: str
    location: str
    summary: str
    skills: List[str]
    experience: List[SampleExperience]
    education: str
    projects: List[SampleProject]
    certifications: List[str] = field(default_factory=list)


SAMPLE_PROFILES: Dict[str, SampleCandidateProfile] = {
    "ai_ml_engineer": SampleCandidateProfile(
        key="ai_ml_engineer",
        display_label="AI/ML Engineer",
        full_name="Alex Chen",
        email="alex.chen@example.com",
        phone="+1 415-555-0142",
        location="San Francisco, CA",
        summary=(
            "Machine Learning Engineer with 6 years of experience building and deploying "
            "production ML systems for computer vision and NLP applications."
        ),
        skills=["Python", "TensorFlow", "PyTorch", "Scikit-learn", "NLP", "Computer Vision",
                "AWS", "Docker", "Kubernetes", "MLOps", "FastAPI", "SQL"],
        experience=[
            SampleExperience(
                title="Senior Machine Learning Engineer", company="Vertex AI Labs", years="2021 - Present",
                bullets=[
                    "Led a team of 4 engineers building NLP pipelines for document classification",
                    "Deployed models to production serving 5M+ requests/day on Kubernetes",
                    "Reduced model inference latency by 35% through quantization and optimization",
                ],
            ),
            SampleExperience(
                title="Machine Learning Engineer", company="DataForge Inc.", years="2018 - 2021",
                bullets=[
                    "Built computer vision models for defect detection in manufacturing",
                    "Implemented MLOps pipelines using AWS SageMaker and Docker",
                ],
            ),
        ],
        education="Master of Science in Computer Science, Carnegie Mellon University, 2018",
        projects=[
            SampleProject(
                name="Real-Time Object Detection System",
                description=(
                    "Built a YOLO-based detection system deployed on edge devices using PyTorch "
                    "and OpenCV, achieving 30fps on embedded hardware."
                ),
                link="github.com/alexchen/edge-detection",
            ),
        ],
        certifications=["AWS Certified Machine Learning - Specialty"],
    ),
    "full_stack_developer": SampleCandidateProfile(
        key="full_stack_developer",
        display_label="Full Stack Developer",
        full_name="Jordan Rivera",
        email="jordan.rivera@example.com",
        phone="+1 512-555-0198",
        location="Austin, TX",
        summary="Full stack developer with 5 years of experience building scalable web applications using React and Node.js.",
        skills=["JavaScript", "TypeScript", "React", "Node.js", "Express.js", "MongoDB",
                "PostgreSQL", "Docker", "Git", "REST API", "HTML5", "CSS3"],
        experience=[
            SampleExperience(
                title="Senior Full Stack Developer", company="Bright Path Software", years="2022 - Present",
                bullets=[
                    "Built and maintained React dashboards used by 80k+ monthly active users",
                    "Designed a Node.js microservices architecture handling 2M+ daily transactions",
                    "Mentored 3 junior developers on code review and testing best practices",
                ],
            ),
            SampleExperience(
                title="Full Stack Developer", company="Webline Studio", years="2019 - 2022",
                bullets=[
                    "Developed e-commerce platforms using the MERN stack",
                    "Migrated a legacy PHP application to a modern React/Node.js architecture",
                ],
            ),
        ],
        education="Bachelor of Science in Software Engineering, University of Texas at Austin, 2019",
        projects=[
            SampleProject(
                name="E-Commerce Platform",
                description=(
                    "Full stack e-commerce application built with React, Node.js, and MongoDB, "
                    "supporting 10,000+ product listings."
                ),
                link="github.com/jordanrivera/ecommerce-platform",
            ),
        ],
        certifications=[],
    ),
    "data_scientist": SampleCandidateProfile(
        key="data_scientist",
        display_label="Data Scientist",
        full_name="Priya Sharma",
        email="priya.sharma@example.com",
        phone="+1 617-555-0173",
        location="Boston, MA",
        summary="Data scientist with 4 years of experience delivering predictive analytics and business intelligence solutions.",
        skills=["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "Machine Learning",
                "Data Warehousing", "AWS", "Statistics"],
        experience=[
            SampleExperience(
                title="Data Scientist", company="Insight Analytics Co.", years="2021 - Present",
                bullets=[
                    "Built churn prediction models improving retention campaign ROI by 22%",
                    "Designed an A/B testing framework used across 5 product teams",
                    "Automated reporting pipelines, saving 15 hours/week of manual analysis",
                ],
            ),
            SampleExperience(
                title="Junior Data Analyst", company="Retail Metrics Group", years="2019 - 2021",
                bullets=[
                    "Built executive reporting dashboards",
                    "Performed statistical analysis on customer segmentation data",
                ],
            ),
        ],
        education="Master of Science in Data Science, Boston University, 2019",
        projects=[
            SampleProject(
                name="Customer Churn Prediction Model",
                description=(
                    "Built a gradient-boosted model predicting customer churn with 87% accuracy "
                    "using historical transaction data."
                ),
                link="github.com/priyasharma/churn-prediction",
            ),
        ],
        certifications=["Google Professional Data Engineer"],
    ),
    "devops_engineer": SampleCandidateProfile(
        key="devops_engineer",
        display_label="DevOps Engineer",
        full_name="Marcus Johnson",
        email="marcus.johnson@example.com",
        phone="+1 303-555-0164",
        location="Denver, CO",
        summary="DevOps engineer with 7 years of experience automating infrastructure and CI/CD pipelines at scale.",
        skills=["AWS", "Terraform", "Kubernetes", "Docker", "Jenkins", "CI/CD", "Ansible",
                "Python", "Bash Scripting", "Linux", "Prometheus", "Grafana"],
        experience=[
            SampleExperience(
                title="Senior DevOps Engineer", company="CloudScale Systems", years="2020 - Present",
                bullets=[
                    "Managed Kubernetes infrastructure supporting 200+ microservices",
                    "Built Terraform modules cutting environment provisioning from days to hours",
                    "Implemented Prometheus/Grafana observability across all production services",
                ],
            ),
            SampleExperience(
                title="Systems Engineer", company="NetOps Solutions", years="2017 - 2020",
                bullets=[
                    "Automated deployment pipelines using Jenkins and Ansible",
                    "Migrated on-premise infrastructure to AWS, cutting hosting costs by 30%",
                ],
            ),
        ],
        education="Bachelor of Science in Information Technology, Colorado State University, 2017",
        projects=[
            SampleProject(
                name="Infrastructure-as-Code Toolkit",
                description=(
                    "Open-source Terraform module collection for standardized AWS environment "
                    "provisioning, adopted by 12 internal teams."
                ),
                link="github.com/marcusjohnson/iac-toolkit",
            ),
        ],
        certifications=["AWS Certified DevOps Engineer - Professional", "Certified Kubernetes Administrator (CKA)"],
    ),
}


def list_available_profiles() -> List[Dict[str, str]]:
    """For a UI's "Generate Sample CV" buttons: [{"key": ..., "label": ...}, ...]"""
    return [{"key": p.key, "label": p.display_label} for p in SAMPLE_PROFILES.values()]


def generate_sample_resume_pdf(profile_key: str) -> bytes:
    """Render one sample candidate profile as PDF bytes, ready to feed
    straight into core.parser.ResumeParser exactly like a real upload.
    """
    if profile_key not in SAMPLE_PROFILES:
        available = ", ".join(SAMPLE_PROFILES.keys())
        raise ValueError(f"Unknown sample profile '{profile_key}'. Available: {available}")
    logger.info("Generating sample resume for profile '%s'", profile_key)
    return _render_pdf(SAMPLE_PROFILES[profile_key])


def _render_pdf(profile: SampleCandidateProfile) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Title"], fontSize=18, spaceAfter=2)
    contact_style = ParagraphStyle("Contact", parent=styles["Normal"], fontSize=9, textColor=colors.grey)
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=4,
        textColor=colors.HexColor("#2563EB"),
    )
    body_style = styles["Normal"]

    elements = [
        Paragraph(_esc(profile.full_name), name_style),
        Paragraph(_esc(f"{profile.email} | {profile.phone} | {profile.location}"), contact_style),
        Spacer(1, 0.15 * inch),
        Paragraph("SUMMARY", section_style),
        Paragraph(_esc(profile.summary), body_style),
        Paragraph("SKILLS", section_style),
        Paragraph(_esc(", ".join(profile.skills)), body_style),
        Paragraph("EXPERIENCE", section_style),
    ]
    for job in profile.experience:
        elements.append(Paragraph(f"<b>{_esc(job.title)}, {_esc(job.company)}</b> &nbsp;&nbsp; {_esc(job.years)}", body_style))
        elements.append(ListFlowable(
            [ListItem(Paragraph(_esc(b), body_style)) for b in job.bullets],
            bulletType="bullet", leftIndent=14,
        ))
        elements.append(Spacer(1, 0.08 * inch))

    elements.append(Paragraph("EDUCATION", section_style))
    elements.append(Paragraph(_esc(profile.education), body_style))

    elements.append(Paragraph("PROJECTS", section_style))
    for project in profile.projects:
        elements.append(Paragraph(f"<b>{_esc(project.name)}</b>", body_style))
        elements.append(Paragraph(_esc(project.description), body_style))
        elements.append(Paragraph(_esc(project.link), body_style))
        elements.append(Spacer(1, 0.08 * inch))

    if profile.certifications:
        elements.append(Paragraph("CERTIFICATIONS", section_style))
        for cert in profile.certifications:
            elements.append(Paragraph(_esc(cert), body_style))

    document.build(elements)
    return buffer.getvalue()

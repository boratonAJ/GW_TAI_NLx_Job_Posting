from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def add_box(ax, x, y, w, h, title, lines, face, edge="#0f172a"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=11, fontweight="bold", color="#0b1220")
    ax.text(x + 0.015, y + h - 0.065, "\n".join(lines), ha="left", va="top", fontsize=8.8, color="#0b1220")
    return patch


def connect(ax, a, b, text=""):
    x1 = a.get_x() + a.get_width()
    y1 = a.get_y() + a.get_height() / 2
    x2 = b.get_x()
    y2 = b.get_y() + b.get_height() / 2
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#334155")
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, text, ha="center", va="center", fontsize=8, color="#1f2937")


def down_connect(ax, top, bottom, text=""):
    x1 = top.get_x() + top.get_width() / 2
    y1 = top.get_y()
    x2 = bottom.get_x() + bottom.get_width() / 2
    y2 = bottom.get_y() + bottom.get_height()
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#334155")
    ax.add_patch(arrow)
    if text:
        ax.text((x1 + x2) / 2 + 0.02, (y1 + y2) / 2, text, ha="left", va="center", fontsize=8, color="#1f2937")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.97,
        "Colorado Workforce Intelligence: NLP-Based Skill Extraction & Matching Architecture",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )

    input_jobs = add_box(
        ax,
        0.03,
        0.64,
        0.19,
        0.24,
        "1) NLx Raw Inputs",
        [
            "• data/raw/colorado.csv",
            "• data/raw/colorado_processed.csv",
            "• Job metadata, requirements",
            "  MOC/CIP/ONET/NAICS fields",
        ],
        "#dbeafe",
    )
    preprocess = add_box(
        ax,
        0.27,
        0.64,
        0.2,
        0.24,
        "2) Preprocessing Layer",
        [
            "• Column normalization",
            "• Missing-value handling",
            "• requirements profile cache",
            "  (education/experience)",
        ],
        "#dcfce7",
    )
    nlp_extract = add_box(
        ax,
        0.52,
        0.64,
        0.2,
        0.24,
        "3) NLP Skill Extraction",
        [
            "• Text normalization",
            "• Skill catalog frequency filter",
            "• TF-IDF (1,2-grams)",
            "• Top-k skill mentions/job",
        ],
        "#fef3c7",
    )
    artifacts = add_box(
        ax,
        0.77,
        0.64,
        0.2,
        0.24,
        "4) Processed Artifacts",
        [
            "• nlp_skill_mentions.csv",
            "• nlp_skill_profiles.csv",
            "• nlp_requirements_profile.csv",
            "• analytics db/csv",
        ],
        "#ede9fe",
    )

    matcher = add_box(
        ax,
        0.18,
        0.3,
        0.22,
        0.23,
        "5) Matching Engine",
        [
            "• Cached TF-IDF index",
            "• Cosine similarity ranking",
            "• top_n recommendation output",
        ],
        "#fee2e2",
    )
    skill_gap = add_box(
        ax,
        0.44,
        0.3,
        0.22,
        0.23,
        "6) Explainability Layer",
        [
            "• Skill-gap analysis",
            "• Matched vs missing skills",
            "• Requirement source labels",
            "  (dataset / inferred)",
        ],
        "#cffafe",
    )
    ux = add_box(
        ax,
        0.7,
        0.3,
        0.25,
        0.23,
        "7) Streamlit UX Applications",
        [
            "• Job Seeker: semantic job matching",
            "• Student: skill demand + field explorer",
            "• Veteran: MOC direct + skill translation",
            "• Usage Insights: persistent analytics",
        ],
        "#e0e7ff",
    )

    connect(ax, input_jobs, preprocess, "schema alignment")
    connect(ax, preprocess, nlp_extract, "clean text + requirements")
    connect(ax, nlp_extract, artifacts, "persist structured outputs")

    down_connect(ax, preprocess, matcher, "jobs_clean")
    down_connect(ax, nlp_extract, matcher, "skill profiles")
    down_connect(ax, matcher, skill_gap, "ranked results")
    connect(ax, skill_gap, ux, "interpretable recommendations")
    down_connect(ax, artifacts, ux, "cached artifacts + logs")

    ax.text(
        0.03,
        0.08,
        "Core techniques: TF-IDF vectorization, cosine similarity, regex-based requirement inference, skill-gap token overlap, MOC semantic mapping.",
        fontsize=9.5,
        color="#1e293b",
    )
    ax.text(
        0.03,
        0.05,
        "Connectors shown represent data lineage from raw NLx inputs to processed artifacts, reusable matching index, and audience-specific application views.",
        fontsize=9.5,
        color="#1e293b",
    )

    png_path = out_dir / "NLP_SKILL_EXTRACTION_ARCHITECTURE.png"
    jpg_path = out_dir / "NLP_SKILL_EXTRACTION_ARCHITECTURE.jpg"
    pdf_path = out_dir / "NLP_SKILL_EXTRACTION_ARCHITECTURE.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(jpg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight", orientation="landscape")
    plt.close(fig)

    print(png_path)
    print(jpg_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
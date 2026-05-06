"""Generate LaTeX paper tables from aggregated OpenFake result CSVs.

The result CSVs are expected to have one row per source/generator with:
    model,n,accuracy,tpr,tnr,f1,roc_auc

Example:
    python NewOpen/OpenFake/scripts/make_paper_tables.py \
        --results-dir NewOpen/OpenFake/results \
        --splits-dir NewOpen/OpenFake/splits \
        --output NewOpen/OpenFake/results/paper_tables.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


OPENFAKE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Detector:
    key: str
    label: str
    path_template: str


DETECTORS = [
    Detector("of_v2", "\\textbf{\\textsc{OpenFake}}", "of_v2/{csv_name}"),
    Detector("cf_pretrained", "\\textsc{C-F}", "cf_pretrained/{csv_name}"),
    Detector("genimage_swinv2", "\\textsc{GenImage}", "genimage_swinv2/{csv_name}"),
    Detector("sofake_v2", "\\textsc{So-Fake}", "sofake_v2/{csv_name}"),
    Detector("semitruths_swinv2", "\\textsc{S.-Truths}", "semitruths_swinv2/{csv_name}"),
    Detector("drct", "DRCT", "baselines/drct/drct_convnext_base_in22k_{csv_name}"),
    Detector("ffpp", "FF++", "baselines/ff++/effnb4_{csv_name}"),
    Detector("clip_d10k", "CLIP D-10k+", "baselines/clip/clipdet_latent10k_plus_{csv_name}"),
]

OPENFAKE_TABLES = [
    (
        "of_test_indist_v2.csv",
        "openfake_indist_results",
        "\\textsc{OpenFake} In-Distribution",
        "Performance on the \\textsc{OpenFake} in-distribution split.",
    ),
    (
        "of_test_ood_models_v2.csv",
        "openfake_ood_results",
        "\\textsc{OpenFake} Out-of-Distribution",
        "Performance on the \\textsc{OpenFake} out-of-distribution generator split.",
    ),
]

CROSS_BENCHMARK_TESTS = [
    ("of_test_indist_v2.csv", "\\textsc{OpenFake} ID"),
    ("of_test_ood_models_v2.csv", "\\textsc{OpenFake} OOD"),
    ("genimage_test.csv", "\\textsc{GenImage}"),
    ("cf_eval.csv", "\\textsc{C-F}"),
    ("sofake_ood.csv", "\\textsc{So-Fake}"),
    ("semitruths_eval.csv", "\\textsc{S.-Truths}"),
    #("reddit.csv", "\\textsc{Reddit}"),
]

REAL_SOURCES = {"laion", "pexels", "docci", "imagenet"}

NAME_MAP = {
    "adm": "ADM",
    "biggan": "BigGAN",
    "cf_eval": "\\textsc{C-F}",
    "chroma": "Chroma",
    "clipdet_latent10k_plus": "CLIP D-10k+",
    "Corvi2023": "DMD Corvi'23",
    "bfl-flux.2-klein-9b": "Flux 2 Klein 9B",
    "dalle2": "DALL\\textperiodcentered E~2",
    "dalle3": "DALL\\textperiodcentered E~3",
    "dalle-2": "DALL\\textperiodcentered E~2",
    "dalle-3": "DALL\\textperiodcentered E~3",
    "dall-e-3": "DALL\\textperiodcentered E~3",
    "DFGAN": "DFGAN",
    "docci": "DOCCI",
    "drct_convnext_base_in22k": "DRCT",
    "effnb4": "FF++",
    "ernie-image": "Ernie Image",
    "ernie-image-turbo": "Ernie Image Turbo",
    "flux-1.0-dev": "Flux 1.0 Dev",
    "flux-1.0-schnell": "Flux 1.0 Schnell",
    "flux-1.1-pro": "Flux 1.1 Pro",
    "flux.1-dev": "Flux 1.0 Dev",
    "flux.1-schnell": "Flux 1.0 Schnell",
    "flux.2-dev": "Flux 2 Dev",
    "flux.2-klein-4b": "Flux 2 Klein 4B",
    "flux.2-klein-9b": "Flux 2 Klein 9B",
    "flux.2-klein-base-4b": "Flux 2 Klein Base 4B",
    "flux-dev": "Flux Dev",
    "flux-pro": "Flux Pro",
    "flux-realism": "Flux Realism",
    "genimage_test": "\\textsc{GenImage}",
    "glide": "GLIDE",
    "gpt-image-1": "GPT Image 1",
    "gpt-image-1.5": "GPT Image 1.5",
    "gpt-image-2": "GPT Image 2",
    "grok-2": "Grok 2",
    "grok-2-image-1212": "Grok 2",
    "hidream-i1-full": "HiDream-I1 Full",
    "ideogram-2.0": "Ideogram 2.0",
    "ideogram-3.0": "Ideogram 3.0",
    "illustrious": "Illustrious",
    "imagen-3.0": "Imagen 3.0",
    "imagen-3.0-002": "Imagen 3.0",
    "imagen-4.0": "Imagen 4.0",
    "imagenet": "ImageNet",
    "laion": "LAION",
    "aurora-20-1-25": "Aurora",
    "frames-23-1-25": "Frames",
    "lumina-17-2-25": "Lumina",
    "halfmoon-4-4-25": "Halfmoon",
    "lumina": "Lumina",
    "midjourney-5": "Midjourney 5",
    "midjourney-6": "Midjourney 6",
    "midjourney-7": "Midjourney 7",
    "nano-banana": "Nano Banana",
    "nano-banana-2": "Nano Banana 2",
    "nano-banana-pro": "Nano Banana Pro",
    "novaanimexl-ilv140": "Nova Anime XL IL v14.0",
    "of_test_indist_v2": "\\textsc{OpenFake} ID",
    "of_test_ood_models_v2": "\\textsc{OpenFake} OOD",
    "pexels": "Pexels",
    "recraft-v2": "Recraft v2",
    "recraft-v3": "Recraft v3",
    "sd-1.5": "SD 1.5",
    "sd-1.5-dreamshaper": "SD 1.5 DreamShaper",
    "sd-1.5-epicdream": "SD 1.5 EpicDream",
    "sd-2.1": "SD 2.1",
    "sd-3.5": "SD 3.5",
    "sd-turbo": "SD Turbo",
    "sd-xl": "SD XL",
    "sd-xl-v1.0-vae-fix": "SD XL VAE Fix",
    "sdxl": "SD XL",
    "sdxl-1.0": "SD XL",
    "sdxl-epic-realism": "SD XL Epic Realism",
    "sdxl-juggernaut": "SD XL Juggernaut",
    "sdxl-realvis-v5": "SD XL RealVis v5",
    "sdxl-touchofrealism": "SD XL Touch of Realism",
    "sdxl-turbo": "SD XL Turbo",
    "semitruths_eval": "\\textsc{S.-Truths}",
    "sora-2": "Sora 2",
    "sofake_ood": "\\textsc{So-Fake}",
    "stable_diffusion_v_1_4": "Stable Diffusion 1.4",
    "stable_diffusion_v_1_5": "Stable Diffusion 1.5",
    "stable-diffusion-v1-4": "Stable Diffusion 1.4",
    "stable-diffusion-v1-5": "Stable Diffusion 1.5",
    "stable-diffusion-xl-base-1.0": "Stable Diffusion XL",
    "studio-ghibli-wan2.1-t2v-14b-v1.0": "Wan Video 2.1",
    "veo-3": "Veo 3",
    "vqdm": "VQDM",
    "wan-video-2.2": "Wan Video 2.2",
    "wan-video-2.5": "Wan Video 2.5",
    "wukong": "Wukong",
    "z-image-turbo": "Z-Image Turbo",
}


@dataclass(frozen=True)
class ResultRow:
    model: str
    n: int
    accuracy: float | None
    tpr: float | None
    tnr: float | None
    f1: float | None
    roc_auc: float | None


@dataclass(frozen=True)
class DisplayRow:
    label: str
    sources: list[str]
    block: str
    is_real: bool = False


@dataclass(frozen=True)
class DatasetStat:
    model: str
    count: int
    release: str
    model_type: str
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=OPENFAKE_ROOT / "results",
        help="Directory containing detector result subdirectories.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=OPENFAKE_ROOT / "splits",
        help="Directory containing source split CSVs used for row ordering.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OPENFAKE_ROOT / "results" / "paper_tables.tex",
        help="Path for the generated all-tables LaTeX file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OPENFAKE_ROOT / "results" / "paper_tables",
        help="Directory for importable per-table LaTeX fragments.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal places to show.",
    )
    parser.add_argument(
        "--compression-csv",
        type=Path,
        default=OPENFAKE_ROOT / "results" / "compression" / "of_v2_ood.csv",
        help="Compression sweep result CSV.",
    )
    return parser.parse_args()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def nice_name(raw: str) -> str:
    if raw in NAME_MAP:
        return NAME_MAP[raw]
    normalized = raw.replace("_", "-")
    if normalized in NAME_MAP:
        return NAME_MAP[normalized]
    parts = [part for part in normalized.split("-") if part]
    label = latex_escape(" ".join(part.upper() if part in {"sd", "xl"} else part.capitalize() for part in parts))
    return re.sub(r"\bV(?=\d)", "v", label)


def is_video_source(raw: str) -> bool:
    name = raw.replace("_", "-").lower()
    return (
        name.startswith("veo")
        or name.startswith("sora")
        or name.startswith("wan-video")
        or name.startswith("studio-ghibli-wan2.1")
        or name.startswith("wan2.1")
        or name.startswith("grok-imagine")
    )


def format_label(label: str, sources: list[str]) -> str:
    if any(is_video_source(source) for source in sources):
        return f"\\textcolor{{blue}}{{{label}}}"
    return label


def merge_name(raw: str) -> str:
    """Map raw generator/source names to rows that should be averaged."""
    name = raw.replace("_", "-").lower()
    if raw in REAL_SOURCES:
        return nice_name(raw)
    if name in {"reddit"}:
        return "Reddit"
    if "sdxl" in name or "sd-xl" in name or "stable-diffusion-xl" in name:
        return "SD XL"
    if name.startswith("sd-1.5") or name == "stable-diffusion-v1-5":
        return "SD 1.5"
    if name in {"sd-2.1"}:
        return "SD 2.1"
    if name in {"sd-3.5"}:
        return "SD 3.5"
    if name in {"sd-turbo"}:
        return "SD Turbo"
    if name == "stable-diffusion-v1-4":
        return "SD 1.4"
    if "flux.2-klein" in name or "flux-2-klein" in name:
        return "Flux 2 Klein"
    if "flux.2-dev" in name or "flux-2-dev" in name:
        return "Flux 2 Dev"
    if "schnell" in name:
        return "Flux Schnell"
    if "pro" in name and "flux" in name:
        return "Flux Pro"
    if "flux" in name:
        return "Flux Dev"
    if name.startswith("animagine-xl"):
        return "Animagine XL"
    if name.startswith("anima"):
        return "Anima Preview"
    if "pony" in name:
        return "Pony models"
    return nice_name(raw)


def block_name(raw: str) -> str:
    """Map raw generator/source names to visual blocks separated by midrules."""
    name = raw.replace("_", "-").lower()
    if raw in REAL_SOURCES:
        return "Real"
    if (
        name.startswith("sd-")
        or "sdxl" in name
        or "sd-xl" in name
        or name.startswith("stable-diffusion")
    ):
        return "Stable Diffusion"
    if "flux" in name:
        return "Flux"
    if name.startswith("imagen") or name.startswith("nano-banana"):
        return "Google"
    if name.startswith("seedream"):
        return "Seedream"
    if name.startswith("animagine-xl"):
        return "Animagine XL"
    if "pony" in name:
        return "Pony"
    if name.startswith("grok"):
        return "Grok"
    if name.startswith("gpt") or name.startswith("dall"):
        return "OpenAI"
    if name.startswith("recraft") or name.startswith("ernie-image"):
        return "Recraft"
    if is_video_source(raw):
        return "Video"
    return "Other"


def fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def parse_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    return float(value)


def read_results(path: Path) -> dict[str, ResultRow]:
    if not path.exists():
        return {}
    rows: dict[str, ResultRow] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            rows[model] = ResultRow(
                model=model,
                n=int(float(row["n"])),
                accuracy=parse_float(row.get("accuracy", "")),
                tpr=parse_float(row.get("tpr", "")),
                tnr=parse_float(row.get("tnr", "")),
                f1=parse_float(row.get("f1", "")),
                roc_auc=parse_float(row.get("roc_auc", "")),
            )
    return rows


def detector_result_path(results_dir: Path, detector: Detector, csv_name: str) -> Path:
    return results_dir / detector.path_template.format(csv_name=csv_name)


def read_detector_results(results_dir: Path, detector: Detector, csv_name: str) -> dict[str, ResultRow]:
    return read_results(detector_result_path(results_dir, detector, csv_name))


def read_result_metadata(path: Path) -> dict:
    metadata_path = Path(str(path) + ".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as handle:
        return json.load(handle)


def read_overall_roc_auc(path: Path) -> float | None:
    metadata = read_result_metadata(path)
    value = metadata.get("roc_auc", metadata.get("roc_AUC", metadata.get("auc_roc")))
    if value is None:
        return None
    return float(value)


def read_compression_results(path: Path) -> list[ResultRow]:
    if not path.exists():
        return []
    rows: list[ResultRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ResultRow(
                    model=row["model"],
                    n=int(float(row["n"])),
                    accuracy=parse_float(row.get("accuracy", "")),
                    tpr=parse_float(row.get("tpr", "")),
                    tnr=parse_float(row.get("tnr", "")),
                    f1=parse_float(row.get("f1", "")),
                    roc_auc=parse_float(row.get("roc_auc", "")),
                )
            )
    return rows


def read_compression_by_quality(path: Path) -> dict[str, list[ResultRow]]:
    if not path.exists():
        return {}
    by_quality: dict[str, list[ResultRow]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            quality = row["quality"]
            by_quality.setdefault(quality, []).append(
                ResultRow(
                    model=row["model"],
                    n=int(float(row["n"])),
                    accuracy=parse_float(row.get("accuracy", "")),
                    tpr=parse_float(row.get("tpr", "")),
                    tnr=parse_float(row.get("tnr", "")),
                    f1=parse_float(row.get("f1", "")),
                    roc_auc=parse_float(row.get("roc_auc", "")),
                )
            )
    return by_quality


def read_dataset_stats(path: Path) -> dict[str, DatasetStat]:
    stats: dict[str, DatasetStat] = {}
    if not path.exists():
        return stats
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            if model not in stats:
                stats[model] = DatasetStat(
                    model=model,
                    count=0,
                    release=row.get("release_date", ""),
                    model_type=row.get("type", ""),
                    label=row.get("label", ""),
                )
            current = stats[model]
            stats[model] = DatasetStat(
                model=model,
                count=current.count + 1,
                release=current.release or row.get("release_date", ""),
                model_type=current.model_type or row.get("type", ""),
                label=current.label or row.get("label", ""),
            )
    return stats


def read_dataset_stats_many(paths: list[Path]) -> dict[str, DatasetStat]:
    combined: dict[str, DatasetStat] = {}
    for path in paths:
        for model, stat in read_dataset_stats(path).items():
            if model not in combined:
                combined[model] = stat
                continue
            current = combined[model]
            combined[model] = DatasetStat(
                model=model,
                count=current.count + stat.count,
                release=min(current.release, stat.release) if current.release and stat.release else current.release or stat.release,
                model_type=current.model_type or stat.model_type,
                label=current.label or stat.label,
            )
    return combined


def split_order(path: Path, fallback_rows: list[str]) -> list[str]:
    if not path.exists():
        return fallback_rows
    seen = set()
    ordered = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row.get("model", "")
            if model and model not in seen:
                seen.add(model)
                ordered.append(model)
    for model in fallback_rows:
        if model not in seen:
            ordered.append(model)
    return ordered


def metric_for_row(row: ResultRow | None, source: str) -> float | None:
    if row is None:
        return None
    if source in REAL_SOURCES:
        return row.tnr
    return row.tpr


def weighted_average(rows: list[ResultRow], attr: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = getattr(row, attr)
        if value is None:
            continue
        numerator += value * row.n
        denominator += row.n
    if denominator == 0:
        return None
    return numerator / denominator


def macro_average(rows: list[ResultRow], attr: str) -> float | None:
    values = [getattr(row, attr) for row in rows if getattr(row, attr) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def weighted_metric(rows: list[tuple[ResultRow, str]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for row, source in rows:
        value = metric_for_row(row, source)
        if value is None:
            continue
        numerator += value * row.n
        denominator += row.n
    if denominator == 0:
        return None
    return numerator / denominator


def grouped_sources(ordered_sources: list[str]) -> list[DisplayRow]:
    grouped: list[DisplayRow] = []
    index: dict[tuple[str, str], int] = {}
    first_seen: dict[str, int] = {}
    for source in ordered_sources:
        label = merge_name(source)
        block = block_name(source)
        first_seen.setdefault(block, len(first_seen))
        key = (block, label)
        if key not in index:
            index[key] = len(grouped)
            grouped.append(DisplayRow(label=label, sources=[], block=block))
        grouped[index[key]].sources.append(source)

    def sort_key(row: DisplayRow) -> tuple[int, str]:
        block_order = {
            "Stable Diffusion": 0,
            "Flux": 1,
            "Google": 2,
            "Seedream": 3,
            "Animagine XL": 4,
            "Pony": 5,
            "Grok": 6,
            "OpenAI": 7,
            "Recraft": 8,
            "Video": 9,
        }
        return (block_order.get(row.block, 100 + first_seen[row.block]), row.label.lower())

    return sorted(grouped, key=sort_key)


def block_counts(rows: list[DisplayRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.block] = counts.get(row.block, 0) + 1
    return counts


def needs_block_rule(previous_block: str | None, current_block: str, counts: dict[str, int]) -> bool:
    if previous_block is None or previous_block == current_block:
        return False
    if previous_block == "Video" or current_block == "Video":
        return False
    return counts.get(previous_block, 0) > 1 or counts.get(current_block, 0) > 1


def group_value(
    detector_rows: dict[str, ResultRow],
    sources: list[str],
) -> float | None:
    rows = [(detector_rows[source], source) for source in sources if source in detector_rows]
    return weighted_metric(rows)


def table_header(title: str, caption: str, label: str, columns: list[str]) -> list[str]:
    alignment = "l|" + "c" * len(columns)
    lines = [
        "\\begin{table}[h]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        f"\\textbf{{{title}}} & " + " & ".join(f"\\makecell{{\\small {name}}}" for name in columns) + " \\\\",
        "\\midrule",
    ]
    return lines


def openfake_header(caption: str, title: str, columns: list[str], show_title: bool = True) -> list[str]:
    lines = [
        "\\begin{table}[h]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l|l|" + "c" * len(columns) + "}",
        "\\toprule",
        "\\multicolumn{2}{l|}{} & " + " & ".join(f"\\makecell{{\\small {name}}}" for name in columns) + " \\\\",
        "\\midrule",
    ]
    if show_title:
        lines.extend([
            f"\\multicolumn{{{len(columns) + 2}}}{{c}}{{\\textbf{{{title}}}}} \\\\",
            "\\midrule",
        ])
    return lines


def table_footer(label: str) -> list[str]:
    return [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        f"\\label{{tab:{label}}}",
        "\\end{table}",
    ]


def simple_table_header(title: str, caption: str, columns: list[str]) -> list[str]:
    alignment = "l|" + "c" * len(columns)
    return [
        "\\begin{table}[h]",
        f"\\caption{{{caption}}}",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        f"\\textbf{{{title}}} & " + " & ".join(f"\\makecell{{\\small {name}}}" for name in columns) + " \\\\",
        "\\midrule",
    ]


def simple_table_footer(label: str) -> list[str]:
    return [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\label{{tab:{label}}}",
        "\\end{table}",
    ]


def tex_int(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def release_month(value: str) -> str:
    if not value:
        return "--"
    parts = value.strip().split("-")
    if len(parts) >= 2:
        return f"{parts[0]}--{parts[1]}"
    return latex_escape(value)


def variant_tag(model_type: str) -> str:
    kind = model_type.strip().lower()
    if kind in {"finetune", "fine-tune", "ft"}:
        return "FT"
    if kind == "lora":
        return "LoRA"
    if kind == "video":
        return "video"
    if kind == "image":
        return "image"
    return "base"


def stats_group(model: str) -> str:
    name = model.replace("_", "-").lower()
    if name in REAL_SOURCES:
        return "real"
    if name in {"sd-1.5", "sd-1.5-dreamshaper", "sd-1.5-epicdream", "stable-diffusion-v1-5"}:
        return "sd15"
    if (
        name.startswith("sdxl")
        or name.startswith("sd-xl")
        or name.startswith("stable-diffusion-xl")
        or name.startswith("one-obsession")
        or name.startswith("realvisxl")
    ):
        return "sdxl"
    if name.startswith("flux.1") or name.startswith("flux-1") or name in {
        "mystic",
        "flux-mvc5000",
        "flux-amateursnapshotphotos",
        "flux-realism",
        "openflux.1",
    }:
        return "flux1"
    if name.startswith("flux.2") or name.startswith("bfl-flux.2"):
        return "flux2"
    if name.startswith("animagine-xl"):
        return "animagine"
    if "pony" in name:
        return "pony"
    if name.startswith("seedream"):
        return "seedream"
    if name.startswith("nano-banana"):
        return "nano"
    if name.startswith("imagen"):
        return "imagen"
    if "wan" in name:
        return "wan"
    if name.startswith("amateur-photography"):
        return "amateur"
    return name


DATASET_GROUP_LABELS = {
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "flux1": "Flux 1.0 dev",
    "flux2": "Flux 2",
    "animagine": "Animagine XL",
    "pony": "Pony models",
    "seedream": "Seedream",
    "nano": "Nano Banana",
    "imagen": "Imagen",
    "wan": "Wan Video",
    "amateur": "Amateur Photography",
}


DATASET_GROUP_ORDER = [
    "sd15",
    "sd-2.1",
    "sdxl",
    "sd-3.5",
    "flux1",
    "flux2",
    "flux.1-schnell",
    "flux-1.1-pro",
    "dalle-3",
    "midjourney-6",
    "imagen",
    "gpt-image-1",
    "grok-2-image-1212",
    "ideogram-3.0",
    "seedream",
    "nano",
    "animagine",
    "pony",
    "wan",
]


def dataset_variant_name(stat: DatasetStat) -> str:
    label = nice_name(stat.model)
    tag = variant_tag(stat.model_type)
    return f"{label} ({tag})"


def make_dataset_stats_table(splits_dir: Path) -> str:
    stats = read_dataset_stats_many(
        [
            splits_dir / "of_train_v2.csv",
            splits_dir / "of_test_indist_v2.csv",
            splits_dir / "of_test_ood_models_v2.csv",
        ]
    )
    real_stats = [stat for stat in stats.values() if stat.label == "real"]
    fake_stats = [stat for stat in stats.values() if stat.label == "fake"]
    grouped: dict[str, list[DatasetStat]] = {}
    for stat in fake_stats:
        grouped.setdefault(stats_group(stat.model), []).append(stat)

    ordered_groups = [group for group in DATASET_GROUP_ORDER if group in grouped]
    ordered_groups.extend(sorted(group for group in grouped if group not in set(ordered_groups)))
    caption = "\\textsc{OpenFake} statistics. Image counts are exact. $^*$We release 3M real prompts, of which 800k additional prompts are provided beyond the balanced subset used on the HuggingFace Hub. LoRA variants (``LoRA'') and full finetunes (``FT'') are listed on separate, smaller rows directly below their base models."
    header = [
        "\\toprule",
        "\\textbf{Source} & \\textbf{Release (YYYY--MM)} & \\textbf{\\# Images} \\\\",
        "\\midrule",
    ]

    lines = [
        "\\small",
        "\\begin{longtable}{lrr}",
        f"\\caption{{{caption}}}\\label{{tab:dataset_stats}}\\\\",
        *header,
        "\\endfirsthead",
        "\\toprule",
        "\\multicolumn{3}{l}{\\textit{\\textsc{OpenFake} statistics (continued)}}\\\\",
        "\\midrule",
        *header[1:],
        "\\endhead",
        "\\midrule",
        "\\multicolumn{3}{r}{\\textit{Continued on next page}}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
        f"\\textit{{Real images}} & -- & {tex_int(sum(stat.count for stat in real_stats))} \\\\",
    ]
    for stat in sorted(real_stats, key=lambda stat: nice_name(stat.model).lower()):
        lines.append(
            f"\\quad {{\\scriptsize {nice_name(stat.model)}}} & "
            f"{{\\scriptsize {release_month(stat.release)}}} & "
            f"{{\\scriptsize {tex_int(stat.count)}}} \\\\"
        )
    lines.extend([
        "\\midrule",
    ])

    for group in ordered_groups:
        group_stats = sorted(grouped[group], key=lambda stat: nice_name(stat.model).lower())
        parent_label = DATASET_GROUP_LABELS.get(group, nice_name(group))
        total = sum(stat.count for stat in group_stats)
        release = min(release_month(stat.release) for stat in group_stats if stat.release)
        lines.append(f"{parent_label} & {release} & {tex_int(total)} \\\\")
        if len(group_stats) > 1:
            for stat in group_stats:
                variant = dataset_variant_name(stat)
                lines.append(
                    f"\\quad {{\\scriptsize {variant}}} & "
                    f"{{\\scriptsize {release_month(stat.release)}}} & "
                    f"{{\\scriptsize {tex_int(stat.count)}}} \\\\"
                )
        lines.append("\\midrule")

    total_synthetic = sum(stat.count for stat in fake_stats)
    lines.extend(
        [
            f"\\textbf{{Total synthetic}} & -- & \\textbf{{{tex_int(total_synthetic)}}} \\\\",
            "\\end{longtable}",
        ]
    )
    return "\n".join(lines)


def make_openfake_table(
    results_dir: Path,
    splits_dir: Path,
    csv_name: str,
    label: str,
    title: str,
    caption: str,
    digits: int,
) -> str:
    detector_results = {
        detector.key: read_detector_results(results_dir, detector, csv_name)
        for detector in DETECTORS
    }
    all_sources = sorted({source for rows in detector_results.values() for source in rows})
    ordered_sources = [
        source for source in split_order(splits_dir / csv_name, all_sources)
        if source in all_sources
    ]
    columns = [detector.label for detector in DETECTORS]
    lines = openfake_header(caption, title, columns)

    real_sources = [source for source in ordered_sources if source in REAL_SOURCES]
    fake_sources = [source for source in ordered_sources if source not in REAL_SOURCES]

    for source in real_sources:
        values = [
            fmt(metric_for_row(detector_results[detector.key].get(source), source), digits)
            for detector in DETECTORS
        ]
        lines.append(f"\\rowcolor{{green!20}} & {nice_name(source)} (TNR) & " + " & ".join(values) + " \\\\")

    if real_sources and fake_sources:
        lines.append("\\midrule")

    previous_block: str | None = None
    fake_rows = grouped_sources(fake_sources)
    counts = block_counts(fake_rows)
    for row in fake_rows:
        if needs_block_rule(previous_block, row.block, counts):
            lines.append("\\cmidrule(lr){2-" + str(len(columns) + 2) + "}")
        values = [
            fmt(group_value(detector_results[detector.key], row.sources), digits)
            for detector in DETECTORS
        ]
        lines.append(f"& {format_label(row.label, row.sources)} & " + " & ".join(values) + " \\\\")
        previous_block = row.block

    lines.append("\\midrule")
    avg_values = []
    for detector in DETECTORS:
        fake_rows = [row for source, row in detector_results[detector.key].items() if source not in REAL_SOURCES]
        avg_values.append(fmt(weighted_average(fake_rows, "tpr"), digits))
    lines.append("& Average TPR & " + " & ".join(avg_values) + " \\\\")

    overall_values = []
    for detector in DETECTORS:
        overall_values.append(fmt(weighted_average(list(detector_results[detector.key].values()), "accuracy"), digits))
    lines.append("& Overall Accuracy & " + " & ".join(overall_values) + " \\\\")
    lines.extend(table_footer(label))
    return "\n".join(lines)


def make_combined_openfake_table(results_dir: Path, splits_dir: Path, digits: int) -> str:
    columns = [detector.label for detector in DETECTORS]
    split_defs = [
        ("In-distribution", "of_test_indist_v2.csv"),
        ("Out-of-distribution", "of_test_ood_models_v2.csv"),
    ]
    lines = openfake_header(
        "Performance comparison on \\textsc{OpenFake} across detectors trained on different datasets. Finetuned (FT) and LoRA variants are grouped under their respective base generators. Video generators shown in \\textcolor{blue}{blue} are out-of-distribution for all detectors. SwinV2 trained on \\textsc{OpenFake} consistently outperforms others on unseen generators, while most alternative detectors exhibit high false positive rates (misclassification of real images).",
        "\\textsc{OpenFake}",
        columns,
        show_title=False,
    )
    for split_i, (split_label, csv_name) in enumerate(split_defs):
        detector_results = {
            detector.key: read_detector_results(results_dir, detector, csv_name)
            for detector in DETECTORS
        }
        all_sources = sorted({source for rows in detector_results.values() for source in rows})
        ordered_sources = [
            source for source in split_order(splits_dir / csv_name, all_sources)
            if source in all_sources
        ]
        real_sources = [source for source in ordered_sources if source in REAL_SOURCES]
        fake_groups = grouped_sources([source for source in ordered_sources if source not in REAL_SOURCES])
        body_rows = (
            [DisplayRow(label=nice_name(source), sources=[source], block="Real", is_real=True) for source in real_sources]
            + fake_groups
        )
        if split_i:
            lines.append("\\midrule\\midrule")
        row_count = len(body_rows) + 2
        prefix = (
            f"\\multirow{{{row_count}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\small\\textbf{{{split_label}}}}}}}"
        )
        previous_block: str | None = None
        counts = block_counts(body_rows)
        for row_i, row in enumerate(body_rows):
            if needs_block_rule(previous_block, row.block, counts):
                rule = "\\cmidrule(lr){2-" + str(len(columns) + 2) + "}"
                lines.append(rule)
            values = [
                fmt(group_value(detector_results[detector.key], row.sources), digits)
                for detector in DETECTORS
            ]
            label_text = f"{row.label} (TNR)" if row.is_real else format_label(row.label, row.sources)
            first_cell = prefix if row_i == 0 else ""
            row_prefix = "\\rowcolor{green!20} " if row.is_real else ""
            lines.append(f"{row_prefix}{first_cell} & {label_text} & " + " & ".join(values) + " \\\\")
            previous_block = row.block

        lines.append("\\midrule")
        avg_values = []
        overall_values = []
        for detector in DETECTORS:
            fake_rows = [row for source, row in detector_results[detector.key].items() if source not in REAL_SOURCES]
            avg_values.append(fmt(weighted_average(fake_rows, "tpr"), digits))
            overall_values.append(fmt(weighted_average(list(detector_results[detector.key].values()), "accuracy"), digits))
        lines.append("& Average TPR & " + " & ".join(avg_values) + " \\\\")
        lines.append("& Overall Accuracy & " + " & ".join(overall_values) + " \\\\")
    lines.extend(table_footer("openfake_combined_results"))
    return "\n".join(lines)


def make_reddit_table(results_dir: Path, digits: int) -> str:
    metrics = [
        ("Accuracy", "accuracy"),
        ("TPR", "tpr"),
        ("TNR", "tnr"),
        ("F1", "f1"),
        ("ROC AUC", "roc_auc"),
    ]
    columns = ["\\textsc{OpenFake} Full"] + [detector.label for detector in DETECTORS]
    lines = table_header(
        "\\textsc{Reddit}",
        "Performance on the \\textsc{Reddit} split across detectors trained on different datasets. \\textsc{OpenFake} Full is trained on the full \\textsc{OpenFake} dataset, including OOD models, which improves performance.",
        "reddit_results",
        columns,
    )
    for metric_name, attr in metrics:
        values = []
        full_path = results_dir / "of_v1" / "reddit.csv"
        full_row = read_results(full_path).get("reddit")
        if attr == "roc_auc":
            values.append(fmt(read_overall_roc_auc(full_path), digits))
        else:
            values.append(fmt(getattr(full_row, attr) if full_row else None, digits))
        for detector in DETECTORS:
            path = detector_result_path(results_dir, detector, "reddit.csv")
            row = read_results(path).get("reddit")
            if attr == "roc_auc":
                values.append(fmt(read_overall_roc_auc(path), digits))
            else:
                values.append(fmt(getattr(row, attr) if row else None, digits))
        lines.append(f"{metric_name} & " + " & ".join(values) + " \\\\")
    lines.extend(table_footer("reddit_results"))
    return "\n".join(lines)


def make_reddit_metric_table(
    results_dir: Path,
    digits: int,
    metric_name: str,
    attr: str,
    label: str,
    caption: str,
) -> str:
    columns = ["\\textsc{OpenFake} Full"] + [detector.label for detector in DETECTORS]
    lines = table_header("\\textsc{Reddit}", caption, label, columns)
    metric_values = []
    full_path = results_dir / "of_v1" / "reddit.csv"
    full_row = read_results(full_path).get("reddit")
    if attr == "roc_auc":
        metric_values.append(read_overall_roc_auc(full_path))
    else:
        metric_values.append(getattr(full_row, attr) if full_row else None)
    for detector in DETECTORS:
        path = detector_result_path(results_dir, detector, "reddit.csv")
        row = read_results(path).get("reddit")
        if attr == "roc_auc":
            metric_values.append(read_overall_roc_auc(path))
        else:
            metric_values.append(getattr(row, attr) if row else None)
    best_value = max((value for value in metric_values if value is not None), default=None)
    values = []
    for value in metric_values:
        formatted = fmt(value, digits)
        if value is not None and best_value is not None and math.isclose(value, best_value):
            formatted = f"\\textbf{{{formatted}}}"
        values.append(formatted)
    lines.append(f"{metric_name} & " + " & ".join(values) + " \\\\")
    lines.extend(table_footer(label))
    return "\n".join(lines)


def make_reddit_roc_auc_table(results_dir: Path, digits: int) -> str:
    return make_reddit_metric_table(
        results_dir,
        digits,
        "ROC AUC",
        "roc_auc",
        "reddit_roc_auc",
        "ROC AUC on the \\textsc{Reddit} split across detectors trained on different datasets. Compare columns to see which training source transfers best to in-the-wild Reddit imagery.",
    )


def make_cross_benchmark_metric_table(
    results_dir: Path,
    digits: int,
    attr: str,
    label: str,
    caption: str,
) -> str:
    columns = [name for _, name in CROSS_BENCHMARK_TESTS] + ["Average"]
    lines = table_header(
        "Train \\textbackslash \\space Test",
        caption,
        label,
        columns,
    )

    iid_matches = {
        ("of_v2", "of_test_indist_v2.csv"),
        ("genimage_swinv2", "genimage_test.csv"),
        ("cf_pretrained", "cf_eval.csv"),
        ("sofake_v2", "sofake_ood.csv"),
        ("semitruths_swinv2", "semitruths_eval.csv"),
    }

    detector_values: dict[str, list[float | None]] = {}
    for detector in DETECTORS:
        values = []
        for csv_name, _ in CROSS_BENCHMARK_TESTS:
            if attr == "roc_auc":
                values.append(read_overall_roc_auc(detector_result_path(results_dir, detector, csv_name)))
            else:
                rows = read_detector_results(results_dir, detector, csv_name)
                values.append(weighted_average(list(rows.values()), attr))
        detector_values[detector.key] = values

    non_iid_best_by_column: dict[str, float] = {}
    for column_idx, (csv_name, _) in enumerate(CROSS_BENCHMARK_TESTS):
        candidates = [
            values[column_idx]
            for detector, values in detector_values.items()
            if (detector, csv_name) not in iid_matches and values[column_idx] is not None
        ]
        if candidates:
            non_iid_best_by_column[csv_name] = max(candidates)

    averages: dict[str, float] = {}
    for detector, values in detector_values.items():
        present_values = [value for value in values if value is not None]
        if present_values:
            averages[detector] = sum(present_values) / len(present_values)
    best_average = max(averages.values()) if averages else None

    def wrap(value: str, command: str) -> str:
        return f"\\{command}{{{value}}}"

    def is_close(left: float | None, right: float | None) -> bool:
        return (
            left is not None
            and right is not None
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )

    for detector in DETECTORS:
        formatted_values = []
        values = detector_values[detector.key]
        for column_idx, (csv_name, _) in enumerate(CROSS_BENCHMARK_TESTS):
            value = values[column_idx]
            formatted = fmt(value, digits)
            if (detector.key, csv_name) in iid_matches:
                formatted = wrap(formatted, "textit")
            elif is_close(value, non_iid_best_by_column.get(csv_name)):
                formatted = wrap(formatted, "textbf")
            formatted_values.append(formatted)

        average = averages.get(detector.key)
        formatted_average = fmt(average, digits)
        if is_close(average, best_average):
            formatted_average = wrap(formatted_average, "textbf")
        formatted_values.append(formatted_average)

        lines.append(f"{detector.label} & " + " & ".join(formatted_values) + " \\\\")
    lines.extend(table_footer(label))
    return "\n".join(lines)


def make_cross_benchmark_table(results_dir: Path, digits: int) -> str:
    return make_cross_benchmark_metric_table(
        results_dir,
        digits,
        "accuracy",
        "cross_benchmark_accuracy",
        "Cross-benchmark weighted accuracy. Italics mark matched train--test benchmarks, bold values mark the strongest cross-benchmark transfer for each test set, and the final column summarizes average accuracy across all test sets.",
    )


def make_cross_benchmark_roc_auc_table(results_dir: Path, digits: int) -> str:
    return make_cross_benchmark_metric_table(
        results_dir,
        digits,
        "roc_auc",
        "cross_benchmark_roc_auc",
        "Cross-benchmark ROC AUC computed from all image-level prediction scores. Italics mark matched train--test benchmarks, bold values mark the strongest cross-benchmark transfer for each test set, and the final column summarizes average ROC AUC across all test sets.",
    )


def quality_sort_key(quality: str) -> int:
    if quality == "none":
        return 100
    return int(quality)


def quality_label(quality: str) -> str:
    return "None" if quality == "none" else quality


def compression_summary(path: Path) -> list[dict[str, float | str | None]]:
    by_quality = read_compression_by_quality(path)
    rows = []
    for quality in sorted(by_quality, key=quality_sort_key, reverse=True):
        quality_rows = by_quality[quality]
        fake_rows = [row for row in quality_rows if row.model not in REAL_SOURCES]
        real_rows = [row for row in quality_rows if row.model in REAL_SOURCES]
        rows.append(
            {
                "quality": quality,
                "fake_acc": weighted_average(fake_rows, "accuracy"),
                "real_tnr": weighted_average(real_rows, "tnr"),
                "overall_acc": weighted_average(quality_rows, "accuracy"),
                "fake_f1": weighted_average(fake_rows, "f1"),
            }
        )
    return rows


def make_compression_table(path: Path, digits: int) -> str:
    rows = compression_summary(path)
    lines = simple_table_header(
        "JPEG Quality",
        "Robustness of the \\textbf{\\textsc{OpenFake}} detector on the \\textsc{OpenFake} OOD split under JPEG compression.",
        ["Real TNR", "OOD Acc.", "Overall Acc.", "OOD F1"],
    )
    for row in rows:
        lines.append(
            f"{quality_label(str(row['quality']))} & "
            + " & ".join(
                [
                    fmt(row["real_tnr"], digits),  # type: ignore[arg-type]
                    fmt(row["fake_acc"], digits),  # type: ignore[arg-type]
                    fmt(row["overall_acc"], digits),  # type: ignore[arg-type]
                    fmt(row["fake_f1"], digits),  # type: ignore[arg-type]
                ]
            )
            + " \\\\"
        )
    lines.extend(simple_table_footer("compression_ood_results"))
    return "\n".join(lines)


def make_compression_figure(path: Path, figure_path: Path) -> None:
    rows = compression_summary(path)
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    labels = [quality_label(str(row["quality"])) for row in rows]
    values = [float(row["fake_acc"]) for row in rows if row["fake_acc"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(labels, values, marker="o", linewidth=1.8, markersize=4.5)
    ax.set_xlabel("JPEG quality")
    ax.set_ylabel("Average OOD accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def make_compression_figure_fragment(figure_path: Path) -> str:
    latex_path = Path(os.path.relpath(figure_path, Path.cwd())).as_posix()
    return "\n".join(
        [
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.55\\textwidth]{{{latex_path}}}",
            "\\label{fig:compression_ood_accuracy}",
            "\\end{figure}",
        ]
    )


PLOT_LABELS = {
    "of_v2": "OpenFake",
    "cf_pretrained": "C-F",
    "genimage_swinv2": "GenImage",
    "sofake_v2": "So-Fake",
    "semitruths_swinv2": "S.-Truths",
    "drct": "DRCT",
    "ffpp": "FF++",
    "clip_d10k": "CLIP D-10k+",
}


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def detector_balanced_source_accuracy(
    results_dir: Path,
    detector: Detector,
    csv_name: str,
) -> tuple[float | None, float]:
    real_values = []
    fake_values = []
    for source, row in read_detector_results(results_dir, detector, csv_name).items():
        value = metric_for_row(row, source)
        if value is None:
            continue
        if source in REAL_SOURCES:
            real_values.append(value)
        else:
            fake_values.append(value)
    all_values = real_values + fake_values
    se = standard_error(all_values)
    if real_values and fake_values:
        score = 0.5 * (sum(real_values) / len(real_values)) + 0.5 * (sum(fake_values) / len(fake_values))
        return score, se
    values = real_values or fake_values
    if not values:
        return None, 0.0
    return sum(values) / len(values), se


def openfake_split_plot_rows(results_dir: Path) -> list[tuple[str, float | None, float | None, float, float]]:
    rows = []
    for detector in DETECTORS:
        indist, indist_se = detector_balanced_source_accuracy(results_dir, detector, "of_test_indist_v2.csv")
        ood, ood_se = detector_balanced_source_accuracy(results_dir, detector, "of_test_ood_models_v2.csv")
        if indist is not None or ood is not None:
            rows.append((PLOT_LABELS[detector.key], indist, ood, indist_se, ood_se))
    rows.sort(key=lambda item: max(v for v in item[1:3] if v is not None), reverse=True)
    return rows


def make_openfake_split_figure(results_dir: Path, figure_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = openfake_split_plot_rows(results_dir)
    labels = [row[0] for row in rows]
    indist_values = [np.nan if row[1] is None else row[1] for row in rows]
    ood_values = [np.nan if row[2] is None else row[2] for row in rows]
    indist_errors = [0.0 if row[1] is None else row[3] for row in rows]
    ood_errors = [0.0 if row[2] is None else row[4] for row in rows]
    y = np.arange(len(labels))
    height = 0.36

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    error_style = {"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8}
    ax.barh(
        y - height / 2,
        indist_values,
        height,
        xerr=indist_errors,
        label="In-distribution",
        color="#b8b8b8",
        error_kw=error_style,
    )
    ax.barh(
        y + height / 2,
        ood_values,
        height,
        xerr=ood_errors,
        label="Out-of-distribution",
        color="#a00000",
        error_kw=error_style,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Macro-F1")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def make_openfake_split_vertical_figure(results_dir: Path, figure_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = openfake_split_plot_rows(results_dir)
    labels = [row[0] for row in rows]
    indist_values = [np.nan if row[1] is None else row[1] for row in rows]
    ood_values = [np.nan if row[2] is None else row[2] for row in rows]
    indist_errors = [0.0 if row[1] is None else row[3] for row in rows]
    ood_errors = [0.0 if row[2] is None else row[4] for row in rows]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    error_style = {"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8}
    ax.bar(
        x - width / 2,
        indist_values,
        width,
        yerr=indist_errors,
        label="In-distribution",
        color="#b8b8b8",
        error_kw=error_style,
    )
    ax.bar(
        x + width / 2,
        ood_values,
        width,
        yerr=ood_errors,
        label="Out-of-distribution",
        color="#a00000",
        error_kw=error_style,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, loc="upper right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def make_openfake_split_figure_fragment(figure_path: Path) -> str:
    latex_path = Path(os.path.relpath(figure_path, Path.cwd())).as_posix()
    return "\n".join(
        [
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=0.72\\textwidth]{{{latex_path}}}",
            "\\label{fig:openfake_split_macro_f1}",
            "\\end{figure}",
        ]
    )


def make_openfake_split_vertical_figure_fragment(figure_path: Path) -> str:
    latex_path = Path(os.path.relpath(figure_path, Path.cwd())).as_posix()
    return "\n".join(
        [
            "\\begin{figure}[h]",
            "\\centering",
            f"\\includegraphics[width=\\textwidth]{{{latex_path}}}",
            "\\label{fig:openfake_split_balanced_accuracy_vertical}",
            "\\end{figure}",
        ]
    )


def main() -> None:
    args = parse_args()
    tables: list[tuple[str, str]] = []
    tables.append(("dataset_stats.tex", make_dataset_stats_table(args.splits_dir)))
    tables.append(
        (
            "openfake_combined_results.tex",
            make_combined_openfake_table(args.results_dir, args.splits_dir, args.digits),
        )
    )
    for csv_name, label, title, caption in OPENFAKE_TABLES:
        tables.append(
            (
                f"{label}.tex",
                make_openfake_table(
                    args.results_dir,
                    args.splits_dir,
                    csv_name,
                    label,
                    title,
                    caption,
                    args.digits,
                ),
            )
        )
    tables.extend(
        [
            ("reddit_results.tex", make_reddit_table(args.results_dir, args.digits)),
            ("reddit_roc_auc.tex", make_reddit_roc_auc_table(args.results_dir, args.digits)),
            ("cross_benchmark_accuracy.tex", make_cross_benchmark_table(args.results_dir, args.digits)),
            ("cross_benchmark_roc_auc.tex", make_cross_benchmark_roc_auc_table(args.results_dir, args.digits)),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    openfake_split_figure_path = args.output_dir / "openfake_split_macro_f1.pdf"
    make_openfake_split_figure(args.results_dir, openfake_split_figure_path)
    openfake_split_vertical_figure_path = args.output_dir / "openfake_split_balanced_accuracy_vertical.pdf"
    make_openfake_split_vertical_figure(args.results_dir, openfake_split_vertical_figure_path)
    compression_figure_path = args.output_dir / "compression_ood_accuracy.pdf"
    make_compression_figure(args.compression_csv, compression_figure_path)
    tables.extend(
        [
            ("openfake_split_macro_f1_figure.tex", make_openfake_split_figure_fragment(openfake_split_figure_path)),
            (
                "openfake_split_balanced_accuracy_vertical_figure.tex",
                make_openfake_split_vertical_figure_fragment(openfake_split_vertical_figure_path),
            ),
            ("compression_ood_results.tex", make_compression_table(args.compression_csv, args.digits)),
            ("compression_ood_accuracy_figure.tex", make_compression_figure_fragment(compression_figure_path)),
        ]
    )
    for filename, table in tables:
        (args.output_dir / filename).write_text(table + "\n")
    all_inputs = [
        f"\\input{{{Path(os.path.relpath(args.output_dir / filename, Path.cwd())).as_posix()}}}"
        for filename, _ in tables
    ]
    (args.output_dir / "all_tables.tex").write_text("\n".join(all_inputs) + "\n")
    chunks = [table for _, table in tables]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(chunks) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote importable fragments to {args.output_dir}")


if __name__ == "__main__":
    main()

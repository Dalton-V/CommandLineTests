"""Find unlisted calibration pairs and propose labels using their measured colors."""

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from measure_capture_card_colors import (
    EXPLANATION_NAME,
    FLATNESS_WARNING_STDDEV,
    IMAGE_EXTENSIONS,
    LAYOUTS,
    MANIFEST_NAME,
    REFERENCE_LABEL,
    SIMILAR_LABEL_RMS,
    Calibration,
    default_calibration_dir,
    load_manifest,
    sample_calibration,
)


SAMPLE_FRACTION = 0.4  # Match the default in measure_capture_card_colors.py.
MINIMUM_MATCH_MARGIN = 1.0  # RMS levels between the best and second-best matches.
SCREENSHOT_LABELS = {REFERENCE_LABEL, "Console-Screenshot"}


def discover_pairs(folder, manifest):
    listed = {Path(pair[page]).as_posix() for pairs in manifest.values()
              for pair in pairs for page in ("up", "down")}
    calibrations = []
    for console in LAYOUTS:
        console_dir = folder / console
        if not console_dir.is_dir():
            continue
        pairs = {}
        for path in sorted(console_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if path.relative_to(folder).as_posix() in listed:
                continue
            stem, _, page = path.stem.rpartition("-")
            if not stem or page not in ("Up", "Down"):
                raise ValueError(f"{path.name}: expected matching -Up and -Down filenames.")
            pair = pairs.setdefault(stem, {})
            if page in pair:
                raise ValueError(f"{console}/{stem}: more than one {page} image.")
            pair[page] = path
        for stem, pair in pairs.items():
            for page in ("Up", "Down"):
                if page not in pair:
                    raise ValueError(f"{console}/{stem}: missing an unlisted -{page} image.")
            calibrations.append(Calibration(console, pair["Up"], pair["Down"]))
    return calibrations


def sample_new_pair(calibration):
    for path in (calibration.up_path, calibration.down_path):
        with Image.open(path) as image:
            if abs(image.width / image.height - 16 / 9) > 0.01:
                raise ValueError(f"{path.name}: expected a full 16:9 frame, got {image.width}x{image.height}.")

    sample_calibration(calibration, SAMPLE_FRACTION, None)
    colors = calibration.colors
    if not np.isfinite(colors).all() or not np.isfinite(calibration.stddevs).all():
        raise ValueError(f"{calibration.name}: could not sample all 60 color patches.")
    if np.linalg.matrix_rank(colors - colors.mean(axis=0)) < 3:
        raise ValueError(f"{calibration.name}: sampled patches lack the expected range of colors.")


def propose_labels(folder, manifest, calibrations):
    references, distortions = load_manifest(folder, folder / MANIFEST_NAME, folder / EXPLANATION_NAME)
    if not references:
        raise ValueError("The manifest must contain at least one reference pair.")
    profiles = {REFERENCE_LABEL: references}
    profiles.update((distortion.label, distortion.calibrations) for distortion in distortions)
    averages = {}
    for label, pairs in profiles.items():
        for pair in pairs:
            sample_calibration(pair, SAMPLE_FRACTION, None)
        averages[label] = np.mean([pair.colors for pair in pairs], axis=0)

    for calibration in calibrations:
        sample_new_pair(calibration)
        matches = sorted((float(np.sqrt(np.mean((calibration.colors - colors) ** 2))), label)
                         for label, colors in averages.items())
        best_rms, best_label = matches[0]
        flat_patches = np.all(calibration.stddevs <= FLATNESS_WARNING_STDDEV)
        clear_match = (best_label not in SCREENSHOT_LABELS and best_rms <= SIMILAR_LABEL_RMS
                       and matches[1][0] - best_rms >= MINIMUM_MATCH_MARGIN and flat_patches)
        label = best_label
        if not clear_match:
            base_label = calibration.name.replace("/", "-", 1)
            label = base_label
            suffix = 2
            while label in manifest:
                label = f"{base_label}-{suffix}"
                suffix += 1

        manifest.setdefault(label, []).append({
            "up": calibration.up_path.relative_to(folder).as_posix(),
            "down": calibration.down_path.relative_to(folder).as_posix(),
        })
        yield calibration.name, label, matches[:3], clear_match, flat_patches


def write_manifest(path, manifest):
    profiles = []
    for label, pairs in manifest.items():
        entries = ",\n".join(f"    {json.dumps(pair)}" for pair in pairs)
        profiles.append(f"  {json.dumps(label)}: [\n{entries}\n  ]")
    path.write_text("{\n" + ",\n".join(profiles) + "\n}\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, nargs="?", default=default_calibration_dir())
    parser.add_argument("--dry-run", action="store_true", help="validate and preview without writing the manifest")
    parser.add_argument("--summary", type=Path, help="write a Markdown report of the proposed labels")
    args = parser.parse_args()

    manifest_path = args.folder / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        calibrations = discover_pairs(args.folder, manifest)
        proposals = list(propose_labels(args.folder, manifest, calibrations)) if calibrations else []
        if proposals and not args.dry_run:
            write_manifest(manifest_path, manifest)
    except (OSError, ValueError, KeyError, Image.DecompressionBombError) as error:
        parser.exit(1, f"Calibration error: {error}\n")

    summary = ["### Proposed calibration labels", ""]
    needs_review = False
    if not proposals:
        summary.append("All calibration pairs are already listed in the manifest.")
    for name, label, matches, clear_match, flat_patches in proposals:
        candidates = ", ".join(f"<code>{html.escape(candidate)}</code> ({rms:.2f} RMS)"
                               for rms, candidate in matches)
        decision = "Existing label" if clear_match else "Provisional label for review"
        summary.append(f"- <code>{html.escape(name)}</code> → <code>{html.escape(label)}</code>. "
                       f"{decision}. Closest profiles: {candidates}.")
        if not clear_match:
            needs_review = True
        if not flat_patches:
            summary.append("  Some sampled patches are uneven. Check the screenshot framing and swatch report.")
    report = "\n".join(summary) + "\n"
    if args.summary:
        args.summary.write_text(report, encoding="utf-8")
    else:
        print(report, end="")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            print(f"needs_review={str(needs_review).lower()}", file=output)


if __name__ == "__main__":
    main()

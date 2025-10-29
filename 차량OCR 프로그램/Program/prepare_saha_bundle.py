import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = [
    "app.py",
    "my_models.py",
    "weights_config.json",
    "가중치_연결_가이드.md",
]
DEFAULT_WEIGHTS = {
    "plate_detector": "plate_detector.pt",
    "ocr": "ocr_model.pt",
}


def load_weight_config(config_path: Path) -> Dict[str, str]:
    data = DEFAULT_WEIGHTS.copy()
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key in DEFAULT_WEIGHTS:
                    value = raw.get(key)
                    if isinstance(value, str) and value.strip():
                        data[key] = value.strip()
        except json.JSONDecodeError as exc:
            raise SystemExit(
                f"❌ weights_config.json JSON 파싱 오류: {exc}\n"
                "파일 내용을 다시 확인해주세요."
            ) from exc
    return data


def resolve_path(rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def collect_files(include_training: bool) -> List[Path]:
    files: List[Path] = []
    for name in DEFAULT_FILES:
        candidate = BASE_DIR / name
        if candidate.exists():
            files.append(candidate)
        else:
            raise SystemExit(f"❌ 필수 파일 누락: {candidate}")

    config = load_weight_config(BASE_DIR / "weights_config.json")
    for key, rel_path in config.items():
        weight_path = resolve_path(rel_path)
        if not weight_path.exists():
            raise SystemExit(
                f"❌ '{key}' 가중치 파일을 찾을 수 없습니다: {weight_path}\n"
                "weights_config.json 경로나 파일 존재 여부를 확인하세요."
            )
        files.append(weight_path)

    if include_training:
        for extra in ("OCR", "Obj Detection"):
            extra_path = BASE_DIR / extra
            if extra_path.exists():
                files.append(extra_path)
    return files


def build_bundle(output: Path, include_training: bool) -> Path:
    files = collect_files(include_training)
    with tempfile_directory() as temp_dir:
        staging = Path(temp_dir)
        for src in files:
            dest = staging / src.name
            if src.is_dir():
                shutil.copytree(src, dest)
            else:
                shutil.copy2(src, dest)
        archive_path = shutil.make_archive(
            base_name=str(output.with_suffix("")),
            format="zip",
            root_dir=staging,
        )
    return Path(archive_path)


class tempfile_directory:
    def __enter__(self):
        self._dir = tempfile.mkdtemp()
        return self._dir

    def __exit__(self, exc_type, exc, tb):
        shutil.rmtree(self._dir, ignore_errors=True)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="사하구청 전달용 패키지를 생성합니다.",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="학습용 소스(OCR/, Obj Detection/)도 함께 압축합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="생성될 ZIP 파일 경로 (기본: Program/dist/saha_bundle_yyyymmdd_HHMMSS.zip)",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = BASE_DIR / "dist"
    default_output_dir.mkdir(exist_ok=True)
    output = args.output if args.output else default_output_dir / f"saha_bundle_{timestamp}.zip"

    archive_path = build_bundle(output, include_training=args.include_training)
    print(f"✅ 패키지 생성 완료: {archive_path}")


if __name__ == "__main__":
    main(sys.argv[1:])

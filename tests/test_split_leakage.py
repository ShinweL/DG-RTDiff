import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPLITS = PROJECT.parent / "data" / "fixed_composite_dataset_v2_gain18_lower" / "source_splits.json"


def test_source_splits_are_disjoint() -> None:
    data = json.loads(SPLITS.read_text(encoding="utf-8"))
    for key in ("m_source_ids", "r2_background_files"):
        train = set(data[key]["train"])
        val = set(data[key]["val"])
        test = set(data[key]["test"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)


def test_r1_was_not_used_for_synthesis() -> None:
    data = json.loads(SPLITS.read_text(encoding="utf-8"))
    assert data["r1_used"] is False

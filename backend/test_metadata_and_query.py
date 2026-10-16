import json

from PIL import Image

from recompute_metadata import recompute
from main import _deterministic_search_phrase


def test_recompute_metadata_writes_histogram(tmp_path):
    products_path = tmp_path / "clean_products.json"
    images_dir = tmp_path / "images" / "sofa"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (80, 80), (25, 25, 25)).save(images_dir / "1.jpg")
    products_path.write_text(json.dumps([{
        "id": 1,
        "image_path": "sofa/1.jpg",
        "baseColour": "black",
    }]), encoding="utf-8")

    updated, missing = recompute(products_path, tmp_path / "images")
    product = json.loads(products_path.read_text(encoding="utf-8"))[0]
    assert updated == 1
    assert missing == 0
    assert product["color_hist"]
    assert "black" in product["color_hist"]


def test_deterministic_query_phrase_keeps_attributes():
    assert _deterministic_search_phrase("search for a black leather sofa") == "black leather sofa"

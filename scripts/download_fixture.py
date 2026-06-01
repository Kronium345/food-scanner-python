"""Download a small pizza test fixture from Wikimedia Commons."""

from pathlib import Path

import urllib.request

FIXTURE_URL = "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=320&q=80"
OUTPUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pizza.jpg"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {FIXTURE_URL} -> {OUTPUT}")
    req = urllib.request.Request(FIXTURE_URL, headers={"User-Agent": "oq-food-vision-fixture/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        OUTPUT.write_bytes(resp.read())
    print(f"Saved {OUTPUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()

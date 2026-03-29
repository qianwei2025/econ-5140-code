#!/usr/bin/env python3
"""Rebuild `data/smoking_prop99.csv` from the tidysynth (CRAN) smoking panel.

Source: tidysynth R package — `data/smoking.rda` (same variables as Abadie et al. 2010).

Usage (from repo root):
  pip install pyreadr requests
  python scripts/build_smoking_prop99_csv.py

Requires network access to download the tidysynth source tarball from CRAN.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path

import requests

CRAN_TIDYSYNTH = "https://cran.r-project.org/src/contrib/tidysynth_0.2.1.tar.gz"
RDA_PATH = "tidysynth/data/smoking.rda"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "smoking_prop99.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyreadr  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install pyreadr: pip install pyreadr") from e

    r = requests.get(CRAN_TIDYSYNTH, timeout=120)
    r.raise_for_status()
    tf = tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz")
    member = tf.getmember(RDA_PATH)
    f = tf.extractfile(member)
    if f is None:
        raise RuntimeError(f"Missing {RDA_PATH} in tarball")
    data = f.read()
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        res = pyreadr.read_r(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    df = list(res.values())[0].copy()
    df = df.rename(columns={"state": "State", "year": "Year", "cigsale": "PacksPerCapita"})
    df["Year"] = df["Year"].astype(int)
    treat_year = 1989
    df["treated"] = ((df["State"] == "California") & (df["Year"] >= treat_year)).astype(int)
    cols = ["State", "Year", "PacksPerCapita", "lnincome", "beer", "age15to24", "retprice", "treated"]
    df = df[cols]
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

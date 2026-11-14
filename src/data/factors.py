"""Daily risk-free rate and factor returns from the Ken French data library.

Used for two things only (protocol.md sections 4 and 5):
  - RF, the daily return of a 1-month Treasury bill, as the risk-free rate in Sharpe and Sortino
  - Mom, the daily US momentum factor, for the factor regression

The files are public CSVs inside zip archives. Values are in percent and are
converted to fractions here. Nothing is filled in: if a download or a parse
fails, SourceError is raised and the caller reports that the number is missing.

Rows after `end` are dropped before anything is returned, so holdout dates
never leave this module unless the caller asks for them.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.data.sources import SourceError

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "factors": "F-F_Research_Data_Factors_daily_CSV.zip",      # Mkt-RF, SMB, HML, RF
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",           # Mom
}
SOURCE_NAME = "Ken French data library (daily)"


def parse_french_csv(text: str) -> pd.DataFrame:
    """Read one French daily CSV: keep the lines that start with an 8-digit date."""
    header, rows = None, []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) > 1 and cells[0] == "" and header is None:
            header = cells[1:]                                  # the line naming the columns
        elif len(cells) > 1 and len(cells[0]) == 8 and cells[0].isdigit():
            rows.append(cells)
    if header is None or not rows:
        raise SourceError("could not find a header and dated rows in the French CSV")
    table = pd.DataFrame(rows, columns=["date"] + header[: len(rows[0]) - 1])
    table.index = pd.to_datetime(table.pop("date"), format="%Y%m%d")
    table.index.name = "session"
    table = table.astype(float)
    # The library marks missing values as -99.99 or -999. Treat them as missing, never as returns.
    table = table.mask(table <= -99.0)
    return table / 100.0                                        # percent -> fraction


def _download(file_name: str) -> str:
    """Fetch one zip from the library and return the text of the CSV inside it."""
    try:
        request = urllib.request.Request(BASE_URL + file_name, headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        return archive.read(archive.namelist()[0]).decode("latin-1")
    except Exception as error:
        raise SourceError(f"could not fetch {file_name}: {type(error).__name__}: {error}") from error


def load_factors(end, cache_dir: Path | None = None, start=None) -> pd.DataFrame:
    """Return daily `rf`, `mkt_rf` and `mom` as fractions, up to and including `end`.

    If `cache_dir` is given, the parsed table is saved there and reused next time.
    """
    cache_file = Path(cache_dir) / "french_daily.parquet" if cache_dir else None
    if cache_file is not None and cache_file.exists():
        table = pd.read_parquet(cache_file)
    else:
        factors = parse_french_csv(_download(FILES["factors"]))
        momentum = parse_french_csv(_download(FILES["momentum"]))
        table = pd.DataFrame({"rf": factors["RF"], "mkt_rf": factors["Mkt-RF"]}).join(
            momentum.iloc[:, 0].rename("mom"), how="left")
        table["fetched_at"] = pd.Timestamp.now(tz="UTC")
        table["source"] = SOURCE_NAME
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(cache_file)
    table = table[table.index <= pd.Timestamp(end)]
    if start is not None:
        table = table[table.index >= pd.Timestamp(start)]
    return table

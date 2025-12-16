from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

DEFAULT_SRF_XLSX_URL = (
    "https://sentiwiki.copernicus.eu/__attachments/1692737/"
    "COPE-GSEG-EOPG-TN-15-0007%20-%20Sentinel-2%20Spectral%20Response%20Functions%202022%20-%203.2.xlsx"
)

S2_BANDS_13 = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B10","B11","B12"]

def pick_sheet_name(xl: pd.ExcelFile, platform: str = "S2A") -> str:
    platform = platform.upper()
    candidates = [s for s in xl.sheet_names if ("Spectral Responses" in s and platform in s)]
    if not candidates:
        raise ValueError(f"No sheet containing 'Spectral Responses' and '{platform}' found. Sheets: {xl.sheet_names}")
    return candidates[0]

def load_s2_srf_from_xlsx(
    xlsx_url: str = DEFAULT_SRF_XLSX_URL,
    platform: str = "S2A",
    bands: Optional[List[str]] = None,
    wavelength_col: str = "SR_WL",
    col_prefix: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Returns dict: band -> (lambda_nm, response) with response > 0 and finite.
    """
    bands = bands or S2_BANDS_13
    platform = platform.upper()
    if col_prefix is None:
        col_prefix = f"{platform}_SR_AV_"

    xl = pd.ExcelFile(xlsx_url)
    sheet = pick_sheet_name(xl, platform=platform)
    df = xl.parse(sheet)

    wavelength_nm = pd.to_numeric(df[wavelength_col], errors="coerce").to_numpy()

    srf_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for b in bands:
        col = f"{col_prefix}{b}"
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in sheet '{sheet}'.")
        resp = pd.to_numeric(df[col], errors="coerce").to_numpy()
        m = np.isfinite(wavelength_nm) & np.isfinite(resp) & (resp > 0)
        lam = wavelength_nm[m].astype(float)
        rsp = resp[m].astype(float)
        srf_dict[b] = (lam, rsp)

    return srf_dict

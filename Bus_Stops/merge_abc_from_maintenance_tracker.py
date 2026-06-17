"""
Merge GoDurham A/B/C cleaning categories from the maintenance tracker workbook
into a bus stop inventory CSV.

Usage:
    python merge_abc_from_maintenance_tracker.py inventory.csv "May 26-30  Schedule ABC Bus stops' Maintenance Tracker Final 2025 (1).xlsx"

Outputs:
    inventory_with_abc_categories.csv
    abc_categories_clean.csv
    abc_merge_report.txt

Notes:
    - Uses only Python standard libraries.
    - Reads the workbook's Lookup sheet directly from the .xlsx file.
    - Pulls the MASTER list columns:
        Bus Stop ID = column C
        Bus Stop Location = column D
        Type = column E
    - Merges categories into the inventory by stop ID.
"""

from __future__ import annotations

import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Dict, List, Tuple, Optional

NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

STOP_ID_COLUMNS = [
    "Stop Code",
    "stop_id",
    "Stop ID",
    "Bus Stop ID",
    "STOP_ID",
    "stop_code",
    "StopCode",
]

CATEGORY_COLUMN = "Current Cleaning Category"


def col_letters_to_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref.upper()).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx  # 1-based


def normalize_stop_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text == "":
        return ""
    # Excel may store IDs as 6291.0. Convert clean integer-looking values.
    try:
        num = float(text.replace(",", ""))
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return text


def read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("main:si", NS):
        parts = []
        # A shared string may be split across rich-text runs.
        for t in si.findall(".//main:t", NS):
            parts.append(t.text or "")
        strings.append("".join(parts))
    return strings


def find_sheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    rel_map = {}
    for rel in rels_root:
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if rid:
            rel_map[rid] = target

    for sheet in wb_root.findall("main:sheets/main:sheet", NS):
        name = sheet.attrib.get("name", "")
        if name.strip().lower() == sheet_name.strip().lower():
            rid = sheet.attrib.get("{%s}id" % NS["rel"])
            target = rel_map.get(rid, "")
            if not target:
                break
            if target.startswith("/"):
                return target.lstrip("/")
            if target.startswith("worksheets/"):
                return "xl/" + target
            return "xl/" + target

    raise ValueError(f"Could not find sheet named {sheet_name!r} in workbook.")


def cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        parts = []
        for t in cell.findall(".//main:t", NS):
            parts.append(t.text or "")
        return "".join(parts).strip()

    v = cell.find("main:v", NS)
    if v is None or v.text is None:
        return ""

    raw = v.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)].strip()
        except (ValueError, IndexError):
            return raw.strip()

    return raw.strip()


def read_lookup_rows_from_xlsx(xlsx_path: str) -> List[Dict[str, str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet_path = find_sheet_path(zf, "Lookup")
        root = ET.fromstring(zf.read(sheet_path))

    abc_rows: List[Dict[str, str]] = []
    seen: Dict[str, str] = {}

    for row in root.findall(".//main:sheetData/main:row", NS):
        row_num = int(row.attrib.get("r", "0"))
        if row_num < 4:  # headers are above this
            continue

        values_by_col: Dict[int, str] = {}
        for c in row.findall("main:c", NS):
            ref = c.attrib.get("r", "")
            if not ref:
                continue
            values_by_col[col_letters_to_index(ref)] = cell_value(c, shared_strings)

        # MASTER list columns: C = Bus Stop ID, D = Location, E = Type
        stop_id = normalize_stop_id(values_by_col.get(3, ""))
        stop_name = values_by_col.get(4, "").strip()
        category = values_by_col.get(5, "").strip().upper()

        if stop_id and category in {"A", "B", "C"} and stop_id not in seen:
            seen[stop_id] = category
            abc_rows.append({
                "stop_id": stop_id,
                CATEGORY_COLUMN: category,
                "abc_stop_name": stop_name,
            })

    return abc_rows


def find_stop_id_column(fieldnames: List[str]) -> Optional[str]:
    lower_to_original = {f.strip().lower(): f for f in fieldnames}
    for candidate in STOP_ID_COLUMNS:
        found = lower_to_original.get(candidate.lower())
        if found:
            return found
    return None


def read_inventory(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_categories(inventory_path: str, tracker_xlsx_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    inventory_rows, inventory_fields = read_inventory(inventory_path)
    stop_id_col = find_stop_id_column(inventory_fields)
    if not stop_id_col:
        raise ValueError(
            "Could not find a stop ID column in the inventory. Expected one of: "
            + ", ".join(STOP_ID_COLUMNS)
        )

    abc_rows = read_lookup_rows_from_xlsx(tracker_xlsx_path)
    abc_by_id = {r["stop_id"]: r for r in abc_rows}

    # Remove any old exact category column so the output has one clear source of truth.
    output_fields = [f for f in inventory_fields if f != CATEGORY_COLUMN]
    if CATEGORY_COLUMN not in output_fields:
        output_fields.append(CATEGORY_COLUMN)
    if "abc_stop_name" not in output_fields:
        output_fields.append("abc_stop_name")

    matched = 0
    missing_inventory_ids: List[str] = []

    merged_rows: List[Dict[str, str]] = []
    for row in inventory_rows:
        new_row = dict(row)
        sid = normalize_stop_id(row.get(stop_id_col, ""))
        abc = abc_by_id.get(sid)
        if abc:
            matched += 1
            new_row[CATEGORY_COLUMN] = abc[CATEGORY_COLUMN]
            new_row["abc_stop_name"] = abc["abc_stop_name"]
        else:
            new_row[CATEGORY_COLUMN] = ""
            new_row["abc_stop_name"] = ""
            if sid:
                missing_inventory_ids.append(sid)
        merged_rows.append(new_row)

    inventory_ids = {normalize_stop_id(row.get(stop_id_col, "")) for row in inventory_rows}
    abc_not_in_inventory = [r for r in abc_rows if r["stop_id"] not in inventory_ids]

    write_csv(os.path.join(output_dir, "inventory_with_abc_categories.csv"), merged_rows, output_fields)
    write_csv(os.path.join(output_dir, "abc_categories_clean.csv"), abc_rows, ["stop_id", CATEGORY_COLUMN, "abc_stop_name"])
    write_csv(
        os.path.join(output_dir, "abc_not_in_inventory.csv"),
        abc_not_in_inventory,
        ["stop_id", CATEGORY_COLUMN, "abc_stop_name"],
    )

    counts = Counter(r[CATEGORY_COLUMN] for r in abc_rows)
    report = [
        "A/B/C merge report",
        "====================",
        f"Inventory file: {inventory_path}",
        f"Tracker workbook: {tracker_xlsx_path}",
        f"Inventory stop ID column used: {stop_id_col}",
        "",
        f"Inventory rows: {len(inventory_rows)}",
        f"A/B/C rows extracted from Lookup sheet: {len(abc_rows)}",
        f"Matched inventory rows: {matched}",
        f"Inventory rows missing A/B/C category: {len(inventory_rows) - matched}",
        f"A/B/C tracker rows not found in inventory: {len(abc_not_in_inventory)}",
        "",
        "A/B/C counts from tracker:",
        f"  A: {counts.get('A', 0)}",
        f"  B: {counts.get('B', 0)}",
        f"  C: {counts.get('C', 0)}",
        "",
        "Next step:",
        "  python task_2_2_cleaning_priority.py inventory_with_abc_categories.csv",
    ]
    with open(os.path.join(output_dir, "abc_merge_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Done.")
    print(f"Output folder: {output_dir}")
    print(f"Extracted A/B/C rows: {len(abc_rows)}")
    print(f"Matched inventory rows: {matched} of {len(inventory_rows)}")
    print(f"A/B/C counts: A={counts.get('A',0)}, B={counts.get('B',0)}, C={counts.get('C',0)}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python merge_abc_from_maintenance_tracker.py inventory.csv maintenance_tracker.xlsx [output_folder]")
        sys.exit(1)

    inventory_path = sys.argv[1]
    tracker_xlsx_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) >= 4 else "abc_merge_outputs"
    merge_categories(inventory_path, tracker_xlsx_path, output_dir)


if __name__ == "__main__":
    main()

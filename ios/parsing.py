from __future__ import annotations

from typing import List, Sequence, Tuple
import csv


def read_xy_csv(path: str, delimiter: str = None) -> List[Tuple[float, float]]:
    """
    Reads 2-column text file (CSV/TSV/space-separated) with x,y.
    - delimiter=None -> autodetect common separators (comma, semicolon, tab, space).
    - Ignores empty lines and lines starting with '#'.
    """
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        raise ValueError("No data in file: {}".format(path))

    if delimiter is None:
        sample = lines[0]
        if "\t" in sample:
            delimiter = "\t"
        elif ";" in sample and "," in sample:
            delimiter = ";"
        elif ";" in sample:
            delimiter = ";"
        elif "," in sample:
            delimiter = ","
        else:
            delimiter = None

    xy: List[Tuple[float, float]] = []
    if delimiter is None:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            x = float(parts[0])
            y = float(parts[1])
            xy.append((x, y))
    else:
        reader = csv.reader(lines, delimiter=delimiter)
        for row in reader:
            if len(row) < 2:
                continue
            x = float(row[0].strip())
            y = float(row[1].strip())
            xy.append((x, y))

    if len(xy) < 2:
        raise ValueError("Need at least 2 points")
    return xy


def sort_and_collapse_duplicates(xy: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Табличный источник допускает не монотонный x.
    Для интерполяции делаем:
      1) сортировку по x
      2) схлопывание одинаковых x (среднее y)
    """
    pts = sorted((float(x), float(y)) for x, y in xy)
    out: List[Tuple[float, float]] = []
    i = 0
    n = len(pts)
    while i < n:
        x0 = pts[i][0]
        s = pts[i][1]
        c = 1
        i += 1
        while i < n and pts[i][0] == x0:
            s += pts[i][1]
            c += 1
            i += 1
        out.append((x0, s / c))
    if len(out) < 2:
        raise ValueError("After collapsing duplicates, need at least 2 unique x")
    return out

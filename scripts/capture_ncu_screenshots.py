#!/usr/bin/env python3
"""Capture genuine Nsight Compute GUI-window screenshots on interactive Windows.

Launch the installed ncu-ui against each native report, enumerate the visible
window, use Windows PrintWindow to capture that application window only, reject
blank captures, and write PNGs plus a manifest. If no usable interactive desktop
exists, record skips rather than synthesizing screenshots. Standard library only.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

if sys.platform != "win32":
    print("NCU_GUI_SNAPSHOTS=SKIPPED; Windows desktop API is unavailable.")
    raise SystemExit(0)

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
BOOL, HWND, HDC = wintypes.BOOL, wintypes.HWND, wintypes.HDC
HBITMAP, HGDIOBJ, LPARAM = wintypes.HBITMAP, wintypes.HGDIOBJ, wintypes.LPARAM


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class RGBQUAD(ctypes.Structure):
    _fields_ = [("rgbBlue", ctypes.c_ubyte), ("rgbGreen", ctypes.c_ubyte),
                ("rgbRed", ctypes.c_ubyte), ("rgbReserved", ctypes.c_ubyte)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]


WNDENUMPROC = ctypes.WINFUNCTYPE(BOOL, HWND, LPARAM)
user32.EnumWindows.argtypes = [WNDENUMPROC, LPARAM]
user32.EnumWindows.restype = BOOL
user32.IsWindowVisible.argtypes = [HWND]
user32.GetWindowTextLengthW.argtypes = [HWND]
user32.GetWindowTextW.argtypes = [HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowThreadProcessId.argtypes = [HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowRect.argtypes = [HWND, ctypes.POINTER(RECT)]
user32.GetWindowRect.restype = BOOL
user32.ShowWindow.argtypes = [HWND, ctypes.c_int]
user32.SetForegroundWindow.argtypes = [HWND]
user32.GetDC.argtypes = [HWND]
user32.GetDC.restype = HDC
user32.ReleaseDC.argtypes = [HWND, HDC]
user32.ReleaseDC.restype = ctypes.c_int
user32.PrintWindow.argtypes = [HWND, HDC, wintypes.UINT]
user32.PrintWindow.restype = BOOL
gdi32.CreateCompatibleDC.argtypes = [HDC]
gdi32.CreateCompatibleDC.restype = HDC
gdi32.DeleteDC.argtypes = [HDC]
gdi32.CreateDIBSection.argtypes = [
    HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = HBITMAP
gdi32.SelectObject.argtypes = [HDC, HGDIOBJ]
gdi32.SelectObject.restype = HGDIOBJ
gdi32.DeleteObject.argtypes = [HGDIOBJ]
BI_RGB, DIB_RGB_COLORS, PW_RENDERFULLCONTENT, SW_MAXIMIZE = 0, 0, 2, 3


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    import struct
    crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _bgra_to_png(raw: bytes, width: int, height: int) -> bytes:
    import struct
    row_bytes = width * 4
    scanlines = bytearray((row_bytes + 1) * height)
    dest = 0
    for y in range(height):
        scanlines[dest] = 0
        dest += 1
        row = raw[y * row_bytes:(y + 1) * row_bytes]
        for p in range(0, row_bytes, 4):
            scanlines[dest] = row[p + 2]
            scanlines[dest + 1] = row[p + 1]
            scanlines[dest + 2] = row[p]
            scanlines[dest + 3] = 255
            dest += 4
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
            + _png_chunk(b"IEND", b""))


def _colour_count(raw: bytes, width: int, height: int) -> int:
    colours, row_bytes = set(), width * 4
    sx, sy = max(1, width // 24), max(1, height // 24)
    for y in range(0, height, sy):
        for x in range(0, width, sx):
            i = y * row_bytes + x * 4
            colours.add((raw[i], raw[i + 1], raw[i + 2]))
    return len(colours)


def _visible_windows():
    found = []

    @WNDENUMPROC
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title_buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buf, length + 1)
        pid = wintypes.DWORD(0)
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        found.append({"hwnd": hwnd, "pid": pid.value, "title": title_buf.value})
        return True

    user32.EnumWindows(callback, 0)
    return found


def _ui_path(ncu_exe: Path):
    """Find ncu-ui.exe across supported Nsight Compute install layouts."""
    candidates = [ncu_exe.resolve().parent / "ncu-ui.exe"]
    pf = os.environ.get("ProgramFiles")
    if pf:
        root = Path(pf) / "NVIDIA Corporation"
        if root.is_dir():
            for install in sorted(root.glob("Nsight Compute*")):
                candidates.extend((
                    install / "ncu-ui.exe",
                    install / "host" / "windows-desktop-win7-x64" / "ncu-ui.exe",
                    install / "target" / "windows-desktop-win7-x64" / "ncu-ui.exe",
                ))
                if install.is_dir():
                    try:
                        candidates.extend(install.rglob("ncu-ui.exe"))
                    except OSError:
                        pass
    seen = set()
    for item in candidates:
        key = str(item).casefold()
        if key in seen:
            continue
        seen.add(key)
        if item.is_file():
            return item
    return None


def _wait_window(pid, stem, timeout):
    end = time.time() + timeout
    while time.time() < end:
        windows = _visible_windows()
        exact = [w for w in windows if w["pid"] == pid]
        if exact:
            return exact[0]
        matching = [w for w in windows if
                    ("nsight compute" in w["title"].lower() or "ncu" in w["title"].lower())]
        stem_matches = [w for w in matching if stem.lower() in w["title"].lower()]
        if stem_matches:
            return stem_matches[0]
        if matching:
            return matching[0]
        time.sleep(0.5)
    return None


def _capture(hwnd, destination):
    rect = RECT()
    if not user32.GetWindowRect(HWND(hwnd), ctypes.byref(rect)):
        raise RuntimeError("GetWindowRect failed")
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width < 700 or height < 450:
        raise RuntimeError(f"Nsight window too small: {width}x{height}")
    screen_dc = user32.GetDC(HWND(0))
    mem_dc = gdi32.CreateCompatibleDC(screen_dc)
    if not mem_dc:
        user32.ReleaseDC(HWND(0), screen_dc)
        raise RuntimeError("CreateCompatibleDC failed")
    info = BITMAPINFO()
    info.bmiHeader = BITMAPINFOHEADER(
        ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32,
        BI_RGB, width * height * 4, 0, 0, 0, 0
    )
    bits = ctypes.c_void_p()
    bitmap = gdi32.CreateDIBSection(
        mem_dc, ctypes.byref(info), DIB_RGB_COLORS, ctypes.byref(bits), None, 0
    )
    if not bitmap or not bits.value:
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(HWND(0), screen_dc)
        raise RuntimeError("CreateDIBSection failed")
    old_bitmap = gdi32.SelectObject(mem_dc, bitmap)
    try:
        if not user32.PrintWindow(HWND(hwnd), mem_dc, PW_RENDERFULLCONTENT):
            raise RuntimeError("PrintWindow failed; screenshot rejected")
        raw = ctypes.string_at(bits.value, width * height * 4)
        distinct = _colour_count(raw, width, height)
        if distinct < 20:
            raise RuntimeError(f"Image appears blank ({distinct} sampled colours)")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_bgra_to_png(raw, width, height))
        return width, height, distinct
    finally:
        gdi32.SelectObject(mem_dc, old_bitmap)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(HWND(0), screen_dc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiler-screenshots"))
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ui = _ui_path(args.ncu_exe)
    manifest = {
        "ncu_ui": str(ui) if ui else None,
        "capture_method": "Windows PrintWindow capture of only the real Nsight Compute application window",
        "captures": [], "skipped": [], "status": "pending",
    }
    if ui is None:
        manifest["status"] = "skipped: ncu-ui.exe not found in the Nsight Compute installation"
        manifest["skipped"].extend({
            "report": str(report),
            "reason": "ncu-ui.exe not found in the Nsight Compute installation",
        } for report in args.reports)
    else:
        for report in args.reports:
            if not report.is_file():
                manifest["skipped"].append({"report": str(report), "reason": "native report absent"})
                continue
            try:
                process = subprocess.Popen(
                    [str(ui), str(report.resolve())], cwd=str(report.resolve().parent),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                window = _wait_window(process.pid, report.stem, args.wait_seconds)
                if not window:
                    window = _wait_window(None, report.stem, 5)
                if not window:
                    manifest["skipped"].append({
                        "report": str(report),
                        "reason": "no visible Nsight Compute window; interactive desktop may be unavailable",
                    })
                    continue
                user32.ShowWindow(HWND(window["hwnd"]), SW_MAXIMIZE)
                user32.SetForegroundWindow(HWND(window["hwnd"]))
                time.sleep(2)
                dest = output_dir / (report.stem.replace(".", "_") + ".png")
                width, height, colours = _capture(window["hwnd"], dest)
                manifest["captures"].append({
                    "report": str(report), "window_title": window["title"],
                    "screenshot": str(dest), "width": width, "height": height,
                    "sampled_distinct_colours": colours,
                })
                print(f"NCU_SNAPSHOT_CAPTURED={dest} {width}x{height}")
            except Exception as exc:
                manifest["skipped"].append({"report": str(report), "reason": f"{type(exc).__name__}: {exc}"})
                print(f"NCU_SNAPSHOT_SKIPPED={report}: {type(exc).__name__}: {exc}")
    if manifest["captures"]:
        manifest["status"] = "captured"
    elif manifest["status"] == "pending":
        manifest["status"] = "no screenshots captured; see skipped reasons"
    (output_dir / "capture-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md = ["# Nsight Compute GUI screenshot manifest", "", f"Status: {manifest['status']}", ""]
    md.extend(
        f"- {x['screenshot']} — {x['window_title']}, {x['width']}x{x['height']}; report {x['report']}."
        for x in manifest["captures"]
    )
    if manifest["skipped"]:
        md += ["", "## Skipped / unavailable", ""]
        md.extend(f"- {x['report']} — {x['reason']}." for x in manifest["skipped"])
    (output_dir / "capture-manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"NCU_GUI_SNAPSHOT_COUNT={len(manifest['captures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

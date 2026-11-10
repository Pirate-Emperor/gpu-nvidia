from __future__ import annotations
import argparse, ctypes, hashlib, importlib.util, subprocess, time, json
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ncu",type=Path,required=True)
    ap.add_argument("--helper",type=Path,required=True)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location("ncu_cap",a.helper)
    cap=importlib.util.module_from_spec(spec);spec.loader.exec_module(cap)
    ui=cap._ui_path(a.ncu)
    if not ui: raise RuntimeError("ncu-ui.exe not found")
    proc=subprocess.Popen([str(ui),str(a.report.resolve())],cwd=str(a.report.parent),
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    win=cap._wait_window(proc.pid,a.report.stem,60) or cap._wait_window(None,a.report.stem,10)
    if not win: raise RuntimeError("Nsight report window did not open")
    hwnd=win["hwnd"]
    cap.user32.ShowWindow(cap.HWND(hwnd),cap.SW_MAXIMIZE)
    cap.user32.SetForegroundWindow(cap.HWND(hwnd));time.sleep(1.5)
    cap._click_relative(hwnd,230,212,verify_y=225);time.sleep(.8)
    captures=[]
    def snap(name):
        path=a.out/(name+".png")
        w,h,col=cap._capture(hwnd,path)
        if (w,h)!=(1936,1048):raise RuntimeError(f"Unexpected screenshot dimensions {w}x{h}")
        captures.append({"name":name,"file":path.name,"sha256":sha(path),"width":w,"height":h,"colours":col})
    snap("01-source-initial")
    # Right-view selector: open actual combo, choose PTX from the list, verify visually via screenshot.
    cap._click_at(hwnd,278,215,pause=.5)
    snap("02-right-view-menu")
    cap._type_key(0x28);cap._type_key(0x0D);time.sleep(.8)
    snap("03-right-view-selected")
    rect=cap.RECT()
    if not cap.user32.GetWindowRect(cap.HWND(hwnd),ctypes.byref(rect)):raise RuntimeError("GetWindowRect failed")
    w,h=rect.right-rect.left,rect.bottom-rect.top
    probes=[]
    # The right metrics table is in the central-right data pane, not the far-right toolbar.
    # The probe tests three in-table positions, each with a baseline and after-scroll capture.
    for idx,frac in enumerate((.72,.75,.78),1):
        x=rect.left+int(w*frac);y=rect.top+int(h*.58)
        if not cap.user32.SetCursorPos(x,y):raise RuntimeError("SetCursorPos failed")
        time.sleep(.15)
        target=cap.user32.WindowFromPoint(cap.POINT(x,y))
        klass=ctypes.create_unicode_buffer(256);cap.user32.GetClassNameW(target,klass,256)
        before=a.out/f"04-before-{idx}.png"
        after=a.out/f"05-after-{idx}.png"
        cap._capture(hwnd,before)
        cap.user32.mouse_event(cap.MOUSEEVENTF_HWHEEL,0,0,ctypes.c_uint32((120*12)&0xffffffff).value,None)
        time.sleep(.75)
        cap._capture(hwnd,after)
        bsha,asha=sha(before),sha(after)
        probes.append({"index":idx,"fraction":frac,"x":x,"y":y,"child_class":klass.value,
                       "before":before.name,"after":after.name,"before_sha256":bsha,
                       "after_sha256":asha,"changed":bsha!=asha})
        # Restore using the same position before testing the next candidate.
        cap.user32.mouse_event(cap.MOUSEEVENTF_HWHEEL,0,0,ctypes.c_uint32((-120*12)&0xffffffff).value,None)
        time.sleep(.5)
    manifest={"report":str(a.report),"window_title":win.get("title"),
              "captures":captures,"scroll_probes":probes}
    (a.out/"probe-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    if not any(x["changed"] for x in probes):
        raise RuntimeError("No candidate location changed the Nsight window")
if __name__=="__main__":main()

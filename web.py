"""Web server (in-memory frame serving)."""

import http.server
import json
import threading
import time

import cv2

from config import FRAME_INTERVAL
from store import EventStore

# ── HTML ─────────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Камера</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#0a0a0a;color:#e0e0e0;
  font-family:-apple-system,BlinkMacSystemFont,sans-serif}
body{display:grid;grid-template-rows:auto 1fr;
  padding:10px;padding-top:max(10px,env(safe-area-inset-top));gap:10px;overflow:hidden}
header{display:flex;align-items:center;gap:8px}
#dot{width:8px;height:8px;border-radius:50%;background:#555;transition:background .3s;flex-shrink:0}
#dot.live{background:#30d158}
header span{font-size:.75rem;font-weight:600;color:#555;letter-spacing:.08em;text-transform:uppercase}
#rbtn{margin-left:auto;background:none;border:1px solid #333;color:#888;
  border-radius:4px;padding:2px 8px;cursor:pointer;font-size:.8rem}
#rbtn:active{opacity:.5}
#grid{display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);
  gap:8px;min-height:0}
.cell{background:#1c1c1e;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.cell img{width:100%;flex:1;object-fit:contain;display:block;min-height:0;background:#000}
.info{padding:5px 7px;flex-shrink:0;background:#1c1c1e}
.names{font-size:.75rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.names.k{color:#30d158}.names.u{color:#ff453a}.names.empty{color:#333}
.meta{font-size:.65rem;color:#555;margin-top:1px;display:flex;gap:6px}
.count{color:#888}
#live-label{font-size:.65rem;color:#555;padding:4px 7px;flex-shrink:0;text-align:center}
</style></head><body>
<header>
  <span id="dot"></span><span>Камера</span>
  <button id="rbtn" onclick="fetch('/reload').then(r=>r.json()).then(d=>{this.textContent='OK';setTimeout(()=>this.textContent='↻',2000)}).catch(()=>this.textContent='✗')">↻</button>
</header>
<div id="grid">
  <div class="cell" id="cell-live">
    <img id="cam" alt="">
    <div id="live-label">прямой эфир</div>
  </div>
</div>
<script>
const grid=document.getElementById('grid'),dot=document.getElementById('dot');
const TOTAL=16;
let lastTs=0,cells=[];
for(let i=0;i<TOTAL-1;i++){
  const c=document.createElement('div');
  c.className='cell';
  c.innerHTML='<img src="" alt="" style="visibility:hidden"><div class="info"><div class="names empty">—</div><div class="meta"><span class="count"></span></div></div>';
  grid.appendChild(c);
  cells.push(c);
}
function fmtDate(ts){
  const d=new Date(ts*1000);
  const dd=String(d.getDate()).padStart(2,'0');
  const mm=String(d.getMonth()+1).padStart(2,'0');
  const hh=String(d.getHours()).padStart(2,'0');
  const mi=String(d.getMinutes()).padStart(2,'0');
  const ss=String(d.getSeconds()).padStart(2,'0');
  return dd+'.'+mm+' '+hh+':'+mi+':'+ss;
}
function rebuild(evs){
  const list=evs.slice().reverse();
  cells.forEach((c,i)=>{
    const ev=list[i];
    const img=c.querySelector('img');
    const nm=c.querySelector('.names');
    const meta=c.querySelector('.meta');
    if(!ev){
      img.style.visibility='hidden';img.src='';
      nm.className='names empty';nm.textContent='—';
      meta.innerHTML='';
      return;
    }
    img.style.visibility='';
    if(img.dataset.ts!==String(ev.ts)){
      img.dataset.ts=ev.ts;
      img.src='snap/'+ev.img+'.jpg?t='+ev.ts;
    }
    const kn=ev.names.filter(n=>n!=='Незнакомец');
    const un=ev.names.filter(n=>n==='Незнакомец').length;
    const total=ev.names.length;
    const hasKnown=kn.length>0;
    let label=kn.join(', ');
    if(un>0)label+=(label?'\u00a0+\u00a0':'')+(un===1?'незн.':un+'\u00a0незн.');
    nm.className='names '+(hasKnown?'k':'u');
    nm.textContent=label||'Незнакомец';
    meta.innerHTML='<span>'+fmtDate(ev.ts)+'</span><span class="count">'+total+'\u00a0чел.</span>';
  });
}
function poll(){
  fetch('detections.json?t='+Date.now()).then(r=>r.json()).then(evs=>{
    dot.className='live';
    const newTs=evs.length?evs[evs.length-1].ts:0;
    if(newTs!==lastTs){lastTs=newTs;rebuild(evs);}
  }).catch(()=>dot.className='');
}
poll();setInterval(poll,1000);
const cam=document.getElementById('cam');
setInterval(()=>{cam.src='frame.jpg?t='+Date.now();},334);
cam.src='frame.jpg?t=0';
</script></body></html>
"""


# ── Web handler ──────────────────────────────────────────────────────────────
class _WebHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        web = self.server.web
        if path in ("/", "/index.html"):
            self._send(_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/frame.jpg":
            buf = web.get_frame_jpeg()
            if buf:
                self._send(buf, "image/jpeg")
            else:
                self.send_error(503)
        elif path == "/detections.json":
            data = json.dumps(
                web.event_store.recent(), ensure_ascii=False
            ).encode()
            self._send(data, "application/json")
        elif path.startswith("/snap/") and path.endswith(".jpg"):
            try:
                eid = int(path[6:-4])
            except ValueError:
                self.send_error(404)
                return
            jpeg = web.event_store.get_snapshot(eid)
            if jpeg:
                self._send(jpeg, "image/jpeg")
            else:
                self.send_error(404)
        elif path == "/health":
            self._send(
                json.dumps(web.get_health()).encode(), "application/json"
            )
        elif path == "/reload":
            web.reload_requested.set()
            self._send(b'{"ok":true}', "application/json")
        elif path == "/clear":
            web.event_store.clear()
            self._send(b'{"ok":true}', "application/json")
        elif path == "/debug/aligned.jpg":
            buf = web._aligned_jpeg
            if buf:
                self._send(buf, "image/jpeg")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


class WebServer:
    def __init__(self, port: int, event_store: EventStore):
        self.event_store = event_store
        self.reload_requested = threading.Event()
        self._frame_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()
        self._live_frame = None
        self._live_lock = threading.Lock()
        self._freeze_frame = None
        self._freeze_lock = threading.Lock()
        self._use_freeze = False
        self._aligned_jpeg: bytes | None = None
        self._start_time = time.time()
        self._last_detect_ts = 0.0

        httpd = http.server.ThreadingHTTPServer(("", port), _WebHandler)
        httpd.web = self
        self._httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._frame_writer, daemon=True).start()
        print(f"[*] Веб: http://0.0.0.0:{port}")

    def update_source(self, frame, has_known, has_stranger, stranger_conf):
        snap = frame.copy()
        if has_known or not has_stranger:
            self._use_freeze = False
        elif has_stranger:
            with self._freeze_lock:
                self._freeze_frame = snap
            if stranger_conf:
                self._use_freeze = True
        with self._live_lock:
            self._live_frame = snap

    def update_aligned(self, jpeg_bytes: bytes):
        self._aligned_jpeg = jpeg_bytes

    def notify_detection(self):
        self._last_detect_ts = time.time()

    def get_frame_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._frame_jpeg

    def get_health(self) -> dict:
        return {
            "uptime_s": round(time.time() - self._start_time),
            "last_detection_ts": self._last_detect_ts,
            "frame_jpeg_bytes": len(self._frame_jpeg) if self._frame_jpeg else 0,
        }

    def shutdown(self):
        self._httpd.shutdown()

    def _frame_writer(self):
        while True:
            time.sleep(FRAME_INTERVAL)
            if self._use_freeze:
                with self._freeze_lock:
                    frame = self._freeze_frame
            else:
                with self._live_lock:
                    frame = self._live_frame
            if frame is None:
                continue
            try:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    with self._frame_lock:
                        self._frame_jpeg = buf.tobytes()
            except Exception:
                pass

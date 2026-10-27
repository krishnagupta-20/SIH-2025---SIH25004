from __future__ import annotations

"""Local image-based cattle and buffalo breed recognition app.

Run: python app.py
Then open: http://127.0.0.1:8000
Install dependencies first: python -m pip install -r requirements.txt
"""
import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
import uvicorn

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("BREED_MODEL", ROOT / "model.onnx"))
LABELS_PATH = Path(os.getenv("BREED_LABELS", ROOT / "breeds.json"))
MAX_UPLOAD = 12 * 1024 * 1024
app = FastAPI(title="Gau Pehchan | Breed Recognition", docs_url="/api/docs")
_model = None
_metadata = None

PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f4f5ef"><title>Gau Pehchan · Breed recognition</title>
<style>
:root{--green:#315c40;--dark:#294a35;--ink:#1d2a21;--muted:#748074;--line:#e2e6dc;--paper:#f4f5ef}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px Arial,sans-serif}.wrap{max-width:1120px;margin:auto;padding:24px}.brand{font-size:20px;font-weight:bold;color:var(--green);padding:13px 2px;border-bottom:1px solid var(--line)}.brand small{float:right;color:var(--muted);font-size:12px;font-weight:normal}.hero{margin:25px 0;background:var(--dark);color:white;padding:27px 30px;border-radius:18px}.hero small{color:#d5e2cb;letter-spacing:1px;font-weight:bold}.hero h1{font-size:25px;margin:9px 0}.hero p{color:#e0e9dc;line-height:1.6;max-width:720px;margin:0}.status{margin-top:17px;border-radius:9px;background:#ffffff19;padding:10px 12px;font-size:12px}.status.warn{background:#9a632d}.grid{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;align-items:start}.card{background:#fff;padding:22px;border:1px solid var(--line);border-radius:18px}.card h2{font-size:16px;margin:0 0 7px}.sub{color:var(--muted);font-size:12px;margin-bottom:17px}.drop{height:250px;border:2px dashed #cbd5c6;border-radius:14px;background:#fafbf8;display:grid;place-items:center;text-align:center;overflow:hidden;transition:.15s}.drop.over{border-color:var(--green);background:#eff4e9}.drop img{max-width:100%;max-height:246px;object-fit:contain}.pick{display:inline-block;margin-top:13px;padding:9px 14px;background:#fff;border:1px solid var(--line);border-radius:9px;color:var(--green);font-weight:bold;cursor:pointer}.drop input{display:none}.row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:15px}label{font-size:10px;font-weight:bold;color:#667266;letter-spacing:.4px}input[type=text],select{padding:10px;border:1px solid var(--line);border-radius:8px;width:100%;margin-top:6px;color:var(--ink);background:#fff}.button{border:0;background:var(--green);color:white;padding:12px;border-radius:9px;font-weight:bold;width:100%;margin-top:15px;cursor:pointer}.button:disabled{opacity:.5;cursor:not-allowed}.hint{text-align:center;margin:9px 0 0}.empty{height:300px;display:grid;place-content:center;text-align:center;color:#71806e}.empty b{color:#263b2d}.result{display:none}.result.on{display:block}.name{font-size:24px;font-weight:bold;margin:9px 0}.meter{height:6px;background:#edf0e9;margin:9px 0 18px;border-radius:6px;overflow:hidden}.meter i{display:block;height:100%;width:0;background:#6d9860;transition:width .4s}.match{display:flex;justify-content:space-between;padding:11px;border-top:1px solid #edf0e9;font-size:12px}.note{background:#f5f6f1;padding:12px;border-radius:9px;color:#737d70;font-size:11px;line-height:1.6;margin-top:12px}.records{margin-top:22px;color:#617060;font-size:12px}.record{display:inline-block;background:white;border:1px solid var(--line);border-radius:9px;padding:10px;margin:8px 7px 0 0;color:var(--ink)}.record button{width:auto;margin:0 0 0 8px;padding:3px 6px;background:none;color:#788276}.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:#243d2c;color:white;padding:11px 17px;border-radius:10px;display:none;max-width:90vw;text-align:center}@media(max-width:720px){.wrap{padding:14px}.grid{grid-template-columns:1fr}.hero{padding:21px}.hero h1{font-size:21px}.empty{height:175px}.brand small{font-size:10px}.drop{height:220px}}
+</style></head><body><main class="wrap"><header class="brand">🐄 gau पहचान <small>Livestock intelligence · India</small></header><section class="hero"><small>FIELD-READY BREED ESTIMATION</small><h1>Identify a cattle or buffalo breed</h1><p>Upload a clear photo for a visual estimate across indigenous and crossbred livestock. Review the shortlist with a local expert.</p><div id="status" class="status">Checking local model…</div></section><div class="grid"><section class="card"><h2>Animal photo</h2><div class="sub">Choose a sharp, well-lit face or side view. JPG, PNG or WEBP · up to 12 MB.</div><div class="drop" id="drop"><div id="placeholder">📷<br><br>Drop a photo here<br><label class="pick" for="photo">Choose photo or use camera</label></div><img id="preview" alt="Selected animal photo" hidden></div><input id="photo" type="file" accept="image/jpeg,image/png,image/webp" capture="environment" hidden><div class="row"><div><label for="species">ANIMAL TYPE</label><select id="species"><option value="any">Cow or buffalo</option><option value="cattle">Cattle</option><option value="buffalo">Buffalo</option></select></div><div><label for="tag">ANIMAL ID · OPTIONAL</label><input id="tag" type="text" maxlength="40" placeholder="Village 04 · 127"></div></div><button id="go" class="button" disabled>✧ Analyze photo</button><div class="sub hint">Your photo is processed by this local app.</div></section><section class="card"><h2>Breed estimate</h2><div id="hint" class="sub">Your result appears here.</div><div id="empty" class="empty"><div style="font-size:34px">⌕</div><b>Ready when you are</b><p>Add an animal photo to see its closest breed matches.</p></div><div id="result" class="result"><small>CLOSEST VISUAL MATCH</small><div id="name" class="name"></div><div id="kind" class="sub"></div><div>Model confidence <b id="conf" style="float:right"></b></div><div class="meter"><i id="fill"></i></div><div id="matches"></div><div class="note">Visual estimates may vary with age, pose, lighting and crossbreeding. Confirm with local records and a trained expert. A photo cannot prove pedigree or calculate COI.</div><button id="save" class="button" style="background:#fff;color:#315c40;border:1px solid var(--line)">＋ Save animal record on this device</button></div></section></div><section class="records"><b>Recently saved</b><div id="records"></div></section></main><div id="toast" class="toast"></div>
+<script>
+const $=s=>document.querySelector(s);let file=null,last=null;function toast(t){let n=$('#toast');n.textContent=t;n.style.display='block';setTimeout(()=>n.style.display='none',2800)}
+fetch('/api/health').then(r=>r.json()).then(j=>{let n=$('#status');if(j.model_ready){n.textContent='✓ Local breed model is ready';n.className='status'}else{n.textContent='Model not installed yet. Add a trained model.onnx and breeds.json beside app.py to enable predictions.';n.className='status warn'}}).catch(()=>{$('#status').textContent='Local model status could not be checked.'});
+function setFile(f){if(!f)return;if(!['image/jpeg','image/png','image/webp'].includes(f.type)){toast('Choose a JPG, PNG or WEBP image.');return}if(f.size>12*1024*1024){toast('Photo must be 12 MB or smaller.');return}file=f;$('#preview').src=URL.createObjectURL(f);$('#preview').hidden=false;$('#placeholder').hidden=true;$('#go').disabled=false}$('#photo').onchange=e=>setFile(e.target.files[0]);let drop=$('#drop');for(let ev of ['dragenter','dragover'])drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('over')});for(let ev of ['dragleave','drop'])drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('over')});drop.addEventListener('drop',e=>setFile(e.dataTransfer.files[0]));
+$('#go').onclick=async()=>{if(!file)return;$('#go').disabled=true;$('#go').textContent='Analyzing photo…';try{let d=new FormData();d.append('image',file);d.append('species',$('#species').value);let r=await fetch('/api/predict',{method:'POST',body:d}),j=await r.json();if(!r.ok)throw Error(j.detail||'Could not analyze photo');last=j;$('#empty').hidden=true;$('#result').className='result on';$('#hint').textContent='Review the top visual matches.';$('#name').textContent=j.top.name;$('#kind').textContent=(j.top.species==='buffalo'?'Water buffalo':'Cattle')+(j.top.crossbred?' · Crossbred':' · Indigenous breed');$('#conf').textContent=Math.round(j.top.confidence*100)+'%';$('#fill').style.width=Math.round(j.top.confidence*100)+'%';$('#matches').innerHTML=j.matches.slice(1).map(x=>'<div class="match"><span>'+escapeHtml(x.name)+' · '+(x.species==='buffalo'?'Buffalo':'Cattle')+'</span><b>'+Math.round(x.confidence*100)+'%</b></div>').join('')}catch(e){toast(e.message==='Failed to fetch'?'Could not reach the local model service.':e.message);$('#hint').textContent='Model unavailable or photo could not be analyzed.'}finally{$('#go').disabled=false;$('#go').textContent='✧ Analyze photo'}};
+function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function records(){let a=[];try{a=JSON.parse(localStorage.getItem('breedRecords')||'[]')}catch{}$('#records').innerHTML=a.slice(0,8).map((x,i)=>'<span class="record"><b>'+escapeHtml(x.name)+'</b> · '+escapeHtml(x.date)+'<button data-i="'+i+'" aria-label="Delete record">×</button></span>').join('')||'<p>No saved records yet.</p>';document.querySelectorAll('.record button').forEach(b=>b.onclick=()=>{let a=JSON.parse(localStorage.getItem('breedRecords')||'[]');a.splice(Number(b.dataset.i),1);localStorage.setItem('breedRecords',JSON.stringify(a));records()})}records();$('#save').onclick=()=>{if(!last)return;let a=JSON.parse(localStorage.getItem('breedRecords')||'[]');a.unshift({name:last.top.name+($('#tag').value?' · '+$('#tag').value:''),date:new Date().toLocaleDateString('en-IN')});localStorage.setItem('breedRecords',JSON.stringify(a));records();toast('Record saved in this browser.')};
+</script></body></html>'''
+
+def get_model():
+    global _model, _metadata
+    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
+        raise HTTPException(503, "Breed model is not installed. Add model.onnx and breeds.json beside app.py.")
+    if _model is None:
+        try:
+            _metadata = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
+            _model = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
+        except Exception as exc:
+            raise HTTPException(503, f"Could not load the breed model: {exc}") from exc
+    return _model, _metadata
+
+def preprocess(image: Image.Image, metadata: dict) -> np.ndarray:
+    width, height = metadata.get("input_size", [224, 224])
+    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (int(width), int(height)), method=Image.Resampling.BILINEAR)
+    array = np.asarray(image, dtype=np.float32) / 255.0
+    mean = np.asarray(metadata.get("mean", [.485, .456, .406]), dtype=np.float32)
+    std = np.asarray(metadata.get("std", [.229, .224, .225]), dtype=np.float32)
+    array = (array - mean) / std
+    if metadata.get("input_layout", "NCHW") == "NHWC":
+        return array[None, ...].astype(np.float32)
+    return array.transpose(2, 0, 1)[None, ...].astype(np.float32)
+
+@app.get("/", response_class=HTMLResponse)
+def home():
+    return PAGE
+
+@app.get("/api/health")
+def health():
+    return {"status": "ready" if MODEL_PATH.is_file() and LABELS_PATH.is_file() else "model_missing", "model_ready": MODEL_PATH.is_file() and LABELS_PATH.is_file()}
+
+@app.post("/api/predict")
+async def predict(image: Annotated[UploadFile, File()], species: Annotated[str, Form()] = "any"):
+    if species not in {"any", "cattle", "buffalo"}:
+        raise HTTPException(400, "Choose cattle, buffalo, or either animal type.")
+    raw = await image.read(MAX_UPLOAD + 1)
+    if len(raw) > MAX_UPLOAD:
+        raise HTTPException(413, "Photo must be 12 MB or smaller.")
+    try:
+        photo = Image.open(BytesIO(raw)); photo.load()
+    except (UnidentifiedImageError, OSError):
+        raise HTTPException(400, "Could not read photo. Use a JPG, PNG, or WEBP image.")
+    model, metadata = get_model()
+    labels = metadata.get("labels", [])
+    if not labels:
+        raise HTTPException(503, "The breed label list is empty.")
+    try:
+        input_name = metadata.get("input_name") or model.get_inputs()[0].name
+        output_name = metadata.get("output_name") or model.get_outputs()[0].name
+        scores = np.asarray(model.run([output_name], {input_name: preprocess(photo, metadata)})[0]).reshape(-1)
+    except Exception as exc:
+        raise HTTPException(500, f"Model inference failed: {exc}") from exc
+    if len(scores) != len(labels):
+        raise HTTPException(500, "Model output count does not match the breed label list.")
+    if metadata.get("output_kind", "logits") == "logits":
+        scores = np.exp(scores - np.max(scores)); scores /= scores.sum()
+    rows = [{"name": label["name"], "species": label["species"], "confidence": float(score), "crossbred": bool(label.get("crossbred", False))}
+            for label, score in zip(labels, scores) if species == "any" or label["species"] == species]
+    rows.sort(key=lambda item: item["confidence"], reverse=True)
+    if not rows:
+        raise HTTPException(400, "No breed labels match that animal type.")
+    return {"top": rows[0], "matches": rows[:3], "note": "Visual estimate; expert verification recommended."}
+
+if __name__ == "__main__":
+    parser = argparse.ArgumentParser(description="Start the local Gau Pehchan breed recognition app.")
+    parser.add_argument("--host", default="127.0.0.1", help="Use 127.0.0.1 for this computer only, or 0.0.0.0 for your local network.")
+    parser.add_argument("--port", default=8000, type=int)
+    args = parser.parse_args()
+    print(f"Gau Pehchan is starting at http://127.0.0.1:{args.port}")
+    uvicorn.run(app, host=args.host, port=args.port)

# QA4 serving smoke + robustness probes (in-process TestClient, real API path)
# Outputs -> /tmp/qa4/smoke_out.json  (repo stays byte-neutral)
import json, time
from fastapi.testclient import TestClient
from ryko_lm.serving.api import app  # noqa: E402

out = {"model": None, "smoke": [], "robust": [], "errors": []}
tc_cm = TestClient(app)
tc = tc_cm.__enter__()

r = tc.get("/health"); out["health"] = r.json()
r = tc.get("/model")
out["model"] = r.json() if r.status_code == 200 else {"status": r.status_code, "body": r.text[:200]}

smoke_prompts = [
    ("hadith", "ما هو الحديث الذي ورد في فضل الصدق؟"),
    ("completion", "أكمل: الكلمة في اللغة إما اسم أو فعل أو حرف."),
    ("fiqh", "ما معنى كلمة الفقه في المصطلح الشرعي؟"),
    ("arkan", "اذكر أحد أركان الإسلام الخمسة."),
    ("seerah", "من هو الخليفة الذي جمع المصحف في عهده؟"),
    ("ood-contemp", "ما هي عاصمة اليابان وما عدد سكانها؟"),
    ("ood-tech", "اشرح لي كيف يعمل الحاسوب خطوة بخطوة."),
]
for name, p in smoke_prompts:
    t0 = time.perf_counter()
    try:
        resp = tc.post("/generate", json={"prompt": p, "max_new_tokens": 48, "temperature": 0.8, "seed": 315})
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            txt = resp.json().get("text") or resp.json().get("output") or str(resp.json())
            out["smoke"].append({"name": name, "ok": True, "latency_s": round(dt, 2), "chars": len(txt), "text200": txt[:200]})
        else:
            out["smoke"].append({"name": name, "ok": False, "status": resp.status_code, "latency_s": round(dt, 2), "body": resp.text[:150]})
    except Exception as e:
        out["errors"].append({"name": name, "exc": repr(e)[:150], "latency_s": round(time.perf_counter() - t0, 2)})

robust_cases = [
    ("empty", ""),
    ("arabic-10000", "نصٌّ عربي طويل. " * 625),  # ~10000 chars
    ("english", "What is the capital of France? Answer briefly."),
    ("sql", "SELECT * FROM users WHERE id=1; DROP TABLE users; --"),
    ("mixed-rtl-ltr", "هذا نص عربي mixed مع English و123 وأرقام ٤٥٦ في سطر واحد."),
]
for name, p in robust_cases:
    t0 = time.perf_counter()
    try:
        resp = tc.post("/generate", json={"prompt": p, "max_new_tokens": 24, "seed": 315})
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            txt = resp.json().get("text") or str(resp.json())
            out["robust"].append({"name": name, "status": 200, "latency_s": round(dt, 2), "text200": txt[:200]})
        else:
            out["robust"].append({"name": name, "status": resp.status_code, "latency_s": round(dt, 2), "body": resp.text[:150]})
    except Exception as e:
        out["errors"].append({"name": name, "exc": repr(e)[:150]})

try:
    resp = tc.post("/generate", json={"prompt": "ما هو الحديث الذي ورد في فضل الصدق؟", "max_new_tokens": 8, "seed": 1})
    out["gen_keys"] = sorted(resp.json().keys()) if resp.status_code == 200 else resp.status_code
except Exception as e:
    out["errors"].append({"gen_keys": repr(e)[:100]})

json.dump(out, open("/tmp/qa4/smoke_out.json", "w"), ensure_ascii=False, indent=1)
tc_cm.__exit__(None, None, None)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])

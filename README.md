<div align="center">

# Ryko

### منظومة هندسية لتركيب القطع — NOVA-ONE

**Piece-Composition Engineering Platform — compose pieces you own into one governed system.**

[![CI](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml/badge.svg)](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-600%2B%20passing-brightgreen)
![Phase](https://img.shields.io/badge/phase-P0%20foundations-orange)
![License](https://img.shields.io/badge/license-proprietary-red)
[![Docs](https://img.shields.io/badge/docs-PLAN%20%7C%20ARCHITECTURE-informational)](docs/PLAN.md)

</div>

> **القرار المعماري الحاكم:** لا نموذج جاهزاً نضبطه، ولا نموذجاً نبنيه من الصفر. نأخذ **قطعاً** — أوزان أساس، Adapters، أدوات، بيانات، مدققات — ونركّبها في منظومة واحدة نملكها، خلف سجل قطع محكوم بعقود واجهة وتراخيص وتقييم Holdout ومراجعة مستقلة.
>
> **القاعدة الذهبية:** لا تجعل النموذج الذي كتب الكود هو الحكم الوحيد على صحته.

---

## ما هذا المستودع؟

Ryko تنفيذ هندسي لخطة **NOVA-ONE** (12–16 شهراً، 6 بوابات انتقال): منظومة تُنفّذ مهام هندسة برمجيات عبر حلقة تحقق لا تُعطَّل:

```
قطعة (أ) تكتب → اختبارات → إصلاح ≤3 → قطعة (ب) مستقلة تراجع → قبول/رفض بأدلة
```

كل قرار مسجَّل في Evidence Log قابل لإعادة التشغيل، وكل قطعة في Piece Registry بعقد واجهة وترخيص ونتيجة Holdout، وكل أمر في سجل تدقيق بسلسلة تجزئة غير قابلة للتعديل.

## الحالة الحالية

| المرحلة | الوسم | الحالة | الأدلة |
|---|---|---|---|
| P0 — البنية والوثائق | `v0.1.0` | منجز | CI أخضر + أرشيف حي 529 اختباراً |
| P0 — سجل القطع + التدقيق | `v0.2.0` | منجز | tamper-detection + حوكمة اعتماد |
| P0 — حلقة التحقق MVP | `v0.3.0` | منجز | بوابة 0: 3 مهام بأدلة كاملة |
| P1 — الحلقة على نموذج حقيقي (20 مهمة، ≥60%) | — | قادم | — |

الخطة الكاملة: [docs/PLAN.md](docs/PLAN.md) · المعمارية: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## البدء السريع

```bash
git clone https://github.com/tibatechapp-ship-it/Ryko.git && cd Ryko
pip install -r requirements.txt

make test            # اختبارات المنصة
make medical-tests   # اختبارات أرشيف MEDICAL-AI (529)
make registry-validate registry-list   # سجل القطع
make demo-loop       # بوابة 0: 3 مهام بحلقة التحقق الكاملة
```

تشغيل حلقة مهام عبر vLLM (أو أي نقطة نهاية OpenAI-Compatible) — بلا أي تغيير كود:

```bash
export RYKO_LLM_BASE_URL="http://localhost:8000/v1"
export RYKO_LLM_API_KEY="none"
export RYKO_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
python3 -m ryko.runner run --tasks tasks/demo --registry registry/registry.yaml
```

## بنية المستودع

```
Ryko/
├── ryko/                    # المنصة: registry, audit, tasks, providers, runner, evidence
├── registry/                # سجل القطع — قلب الحوكمة
├── tasks/                   # مواصفات المهام + fixtures
├── corpus/style/            # Style & Standards Corpus — أول أصل مملوك
├── evidence/                # أدلة البوابات (قابلة لإعادة التشغيل)
├── docs/                    # PLAN + ARCHITECTURE + ADRs
├── archive/medical-ai-v0/   # أول workload حقيقي — بواباته تعمل في CI
├── tests/                   # اختبارات المنصة
└── .github/                 # CI + قوالب PR/Issues
```

## الحوكمة في سطر واحد

| القاعدة | الأداة التي تفرضها |
|---|---|
| لا قطعة بلا عقد واجهة وترخيص وHoldout | `ryko.registry` — الاعتماد يرفض تلقائياً `compositional_delta < +0.10` |
| المراجع مستقل عن المنفذ | الحلقة ترفض `family` متطابقاً قبل أي توليد |
| لا فشل صامت | `ryko.audit` سلسلة تجزئة + كشف تلاعب |
| لا قبول بلا أدلة | `ryko.evidence` + Evidence Gate |
| لا مفاتيح في المستودع | مزودات تقرأ البيئة فقط، فحص gitleaks في CI |

## المساهمة

اقرأ [CONTRIBUTING.md](CONTRIBUTING.md) — تلتزم فيها بقواعد الانضباط الخمس و[Style Corpus](corpus/style/README.md). الأمان: [SECURITY.md](SECURITY.md). سلوك المجتمع: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## الترخيص

**ملكية خاصة — جميع الحقوق محفوظة © 2026 Vixona Technologies (tibatechapp-ship-it).** الاستخدام أو إعادة التوزيع تتطلب إذناً كتابياً.

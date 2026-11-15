<div align="center">

# Ryko

### نموذج لغوي عربي توليدي — خط كامل: كوربوس ← رمزنة ← تدريب ← خدمة

**منظومة عربية متكاملة وقابلة لإعادة الإنتاج لبناء نموذج لغة من الصفر، مبنية على
مكتبة [`tokenizers`](https://github.com/huggingface/tokenizers) (بـRust) وPyTorch —
رمزنة Byte-Level BPE، وRoPE، وRMSNorm، وSwiGLU، وGQA، مع خادم استدلال ببث مباشر
وواجهة تجربة عربية.**

[![CI](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml/badge.svg)](https://github.com/tibatechapp-ship-it/Ryko/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Params](https://img.shields.io/badge/params-11.5M%20%7C%2052M%20%7C%20100M-informational)
![Vocab](https://img.shields.io/badge/vocab-32%2C768%20Byte--BPE-orange)

English version: [README.md](README.md)

</div>

---

## ما هذا المستودع؟

Ryko **خط إنتاج كامل وقابل لإعادة التشغيل لبناء نموذج لغة عربي من الصفر** —
وليس ضبطاً دقيقاً لنموذج جاهز:

| المرحلة | الوحدة | الوظيفة |
|---|---|---|
| 1. الكوربوس | `ryko_lm.data.collect_corpus`، `collect_openiti`، `collect_shamela` | فصحى معاصرة (ويكيبيديا) + تراث: **تشكيلة، OpenITI 2023.1.8 (mARkdown ← نص)، الشاملة v4 (إثبات مصدر لكل صفحة)** |
| 2. التنظيف | `ryko_lm.data.clean_corpus`، `augment_corpus` | توحيد NFKC، إزالة التطويل والتشكيل، تجريد mARkdown، تصفية الأسطر، إزالة تكرار عالمية بالبثّ، توسيع تدريجي تدريب/تحقق |
| 3. الرمزنة | `ryko_lm.tokenizer` | تدريب **Byte-Level BPE بحجم 32,768** مع موحّد عربي — لا يمكن ظهور `[unk]` إطلاقاً |
| 4. الثنائية | `ryko_lm.data.binarize` | الكوربوس ← مصفوفات `uint16` مسطحة عبر memmap، ترميز دفعي في Rust |
| 5. النموذج | `ryko_lm.model.ryko_model` | محوّل فك تشفير: RoPE وRMSNorm وSwiGLU وGQA وفك تشفير بكاش |
| 6. التدريب | `ryko_lm.training.train` | AdamW + جدول كوزوني + تسخين، نقاط حفظ قابلة للاستئناف، سجل مقاييس JSONL |
| 7. التقييم | `ryko_lm.eval.evaluate` | ارتباك حتمي (نوافذ متتالية) + عينات توليد موحدة البذور |
| 8. الخدمة | `ryko_lm.serving.api` | FastAPI: `/generate` وبث SSE في `/generate/stream` وساحة تجربة عربية RTL |
| 9. الضمان | `ryko_lm.provenance`، `ryko_lm.serving.guard` | إيصالات sha256 قابلة للتحقق دون اتصال + حارس توليد قبل التسليم (نمط ZYKO) |

النموذج المرجعي **Ryko-Tiny (11.5 مليون معلم)** دُرّب فعلياً من البداية إلى
النهاية على معالج مركزي ضمن ميزانية CI لهذا المستودع، وتوفر تكوينان أكبر
(`small` و`base`) جاهزان للتدريب على GPU.

## النتائج (تشغيل مرجعي — Ryko-Tiny v1.2 على كوربوس التراث)

| المقياس | القيمة |
|---|---|
| المعاملات | **11,536,640** (منها 3.15M خارج التضمين) |
| المعمارية | 4 طبقات · d=256 · 8 رؤوس / 4 رؤوس KV (GQA) · SwiGLU 768 · RoPE θ=10⁴ · سياق 256 |
| المفردات | 32,768 رمزاً بنمط Byte-Level BPE، دُرّبت في 10.4 ثانية |
| خصوبة الرمزنة | **1.325 توكن/كلمة** · 4.23 حرف/توكن (نص تحقق عربي) |
| كوربوس التدريب (v3) | **246.1 مليون توكن** (تدريب) + 1.36 مليون (تحقق) — ويكيبيديا + تشكيلة + **OpenITI 2023.1.8** (شظايا 0–1، 1,782 كتاباً) + **الشاملة v4** (144 كتاباً من 41 تصنيفاً بإثبات صفحة)؛ حُذفت 1.43 مليون فقرة مكررة بين الطبعات |
| فحص Chinchilla | 11.5M معلم × 20 ≈ 230M توكن — الكوربوس الآن **يتجاوز** النسبة المثلى لـ Ryko-Tiny |
| التدريب | 5,349 خطوة · دفعة 16×256 · **21.9 مليون توكن مرّت على النموذج** (قابل للاستئناف؛ الجدولة تستهدف 20 ألف خطوة / 82M توكن) |
| خسارة التحقق (المجموعة كاملة، نفس البروتوكول) | قبل **6.224 (ارتباك 504.7)** ← بعد **5.078 (ارتباك 160.4)** — **تحسن 3.15×** |
| الاستدلال | ~5 مللي ثانية/توكن (معالج مركزي، كاش KV) · بث مباشر SSE |

**جودة التوليد (v1.2، نطاق التراث، حرارة 0.85):**

| الطلب | عينة (مختصرة) |
|---|---|
| `قال رسول الله صلى الله عليه وسلم` | «...أخبرنا محمد بن هارون ثنا سعيد عن يحيى بن أبي كثير عن عكرمة عن ابن عباس...» — النموذج أصدر **بنية سلسلة إسناد** لم يرها في v1.1 |
| `فأجاب أهل المذهب بأن` | «...في البيع أو غيره ولا يعتق فيه على بيعها وإنما ي...» — سجل نقاش فقهي |

المنحنيات وملفات التقييم والعينات الخام في [docs/results/](docs/results/).

> **ملاحظة هندسية صريحة:** بحجم 11.5 مليون معلم و21.9 مليون توكن تدريب، ينتج
> النموذج عربية بأسلوب تراثي متماسك موضعياً (أنماط الإسناد والعبارات الفقهية)
> لكنه لا يزال **غير** موثوق دلالياً — الارتباك ≈ 160 تحسن 3.15× عن v1.1 لكنه
> بعيد عن الجودة الإنتاجية. المدونة البالغة 246M توكن والمدرّب القابل
> للاستئناف يجعلان الجودة تتحسن بإعادة تشغيل نفس الأمر (المرحلة 3 في
> [docs/ROADMAP.md](docs/ROADMAP.md)) أو الانتقال إلى `ryko-small` (52M).
> التفاصيل في [docs/TRAINING.md](docs/TRAINING.md).

الأصول المرفوعة داخل الشجرة: `artifacts/tokenizer/ryko-tokenizer.json`
و`artifacts/training/final.pt` (نصف دقة) ومنحنى التدريب والعينات في
`docs/results/`.

## البدء السريع

```bash
git clone https://github.com/tibatechapp-ship-it/Ryko.git && cd Ryko
pip install -r requirements.txt          # torch (نسخ CPU تكفي) + tokenizers + fastapi

# 1) استخدم النموذج المدرب مباشرة دون تدريب
python -m ryko_lm.training.evaluate --checkpoint artifacts/training/final.pt \
    --prompts "اللغة العربية" "كان يا ما كان"

# 2) أطلق ساحة التجربة العربية
make serve                               # http://localhost:8000

# 3) أعد إنتاج الخط الكامل من الصفر
make collect && make clean-corpus && make tokenizer && make binarize && make train
```

من داخل بايثون:

```python
import torch
from ryko_lm.model.ryko_model import RykoLM
from ryko_lm.tokenizer.loader import RykoTokenizer

model = RykoLM.load("artifacts/training/final.pt")
tok   = RykoTokenizer.from_dir("artifacts/tokenizer")

ids  = tok.encode("اللغة العربية", add_bos=True)
out  = model.generate(torch.tensor([ids]), max_new_tokens=48, temperature=0.8)
print(tok.decode(out[0].tolist()))
```

## المعمارية باختصار

```
إدخال بايتات ──► Byte-Level BPE (32,768، موحّد عربي)
        ──► 4 × [ RMSNorm ─► انتباه GQA (RoPE، SDPA) ─► +متبقٍ
                  RMSNorm ─► SwiGLU(768)                  ─► +متبقٍ ]
        ──► RMSNorm ─► رأس إخراج مربوط ──► توزيع الرمز التالي
```

لماذا هذه الخيارات للعربية — Byte-Level BPE بدل WordPiece/Unigram، ومفردات 32
ألف، وعدم توحيد الحروف بقوة — تحليله الكامل في
[docs/ARABIC_TOKENIZATION.md](docs/ARABIC_TOKENIZATION.md).  وخصوبة 1.325
توكن/كلمة (مثل: `فسأكتبها ← فس|أ|كتب|ها` و`فاستخدمناها ← ف|است|خد|منا|ها`)
تكشف أن الـBPE تعلّم بنية الإلصاقات العربية بدل تقطيع الكلمات.

## بنية المستودع

```
Ryko/
├── src/ryko_lm/            # الحزمة (بيانات، رمزنة، نموذج، تدريب، خدمة)
├── configs/models/         # ryko-tiny / ryko-small / ryko-base
├── data/sample/            # عينة كوربوس مرفوعة لاختبارات CI
├── artifacts/              # الرمّاز المدرب + نقطة النهاية fp16 (مرفوعة)
├── docs/                   # ARCHITECTURE وARABIC_TOKENIZATION وTRAINING وresults/
├── tests/                  # 30 اختبار pytest (رمزنة، نموذج، بيانات، خدمة)
├── archive/                # إصدارات المنصة السابقة (محفوظة للتاريخ)
└── .github/workflows/      # CI: اختبارات وحدة + خط كامل من البداية للنهاية
```

## تكوينات الحجم

| التكوين | المعاملات | طبقات | d_model | رؤوس/KV | ffn | السياق | العتاد المستهدف |
|---|---|---|---|---|---|---|---|
| `ryko-tiny` | 11.5M | 4 | 256 | 8 / 4 | 768 | 256 | معالج مركزي (دُرّب في CI) |
| `ryko-small` | 52.2M | 12 | 512 | 8 / 4 | 1,408 | 512 | GPU واحد |
| `ryko-base` | 100.7M | 12 | 768 | 12 / 4 | 2,048 | 1,024 | 8× GPU (torchrun) |

## الخدمة والاستدلال

```bash
make serve                                                # http://localhost:8000
curl -s -X POST localhost:8000/generate -H 'Content-Type: application/json' \
     -d '{"prompt": "اللغة العربية", "max_new_tokens": 40, "seed": 7}'
```

نقاط النهاية: `GET /` (ساحة تجربة عربية RTL) · `GET /health` · `GET /model` ·
`POST /generate` · `POST /generate/stream` (بث SSE).  كل مفاتيح التوليد
متاحة: الحرارة، top-k، top-p، عقوبة التكرار، البذرة.

## خطة الطريق

- [ ] مزيد من توكنات التدريب (الأمثل تشينشيلا لـ11.5M ≈ 230 مليون توكن)
- [ ] تغطية اللهجات: إضافة كوربوسات وسائل التواصل (LABR وMSAC) بجانب الفصحى
- [ ] تشغيل مرجعي على GPU لتكوين `ryko-small` بمقاييس منشورة
- [ ] تصدير متوافق مع `transformers` (safetensors)
- [ ] مرحلة ضبط توجيهي (SFT) على كوربوس عربي صغير

## المساهمة

راجع [CONTRIBUTING.md](CONTRIBUTING.md).  سياسة الأمان:
[SECURITY.md](SECURITY.md).  مدونة السلوك: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## الرخصة

[Apache-2.0](LICENSE) — © 2026 مساهمو Ryko (tibatechapp-ship-it).

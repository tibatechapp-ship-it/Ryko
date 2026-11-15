# سجل التغييرات — Ryko

كل التغييرات المهمة تُوثق هنا. الصيغة مبنية على [Keep a Changelog](https://keepachangelog.com/ar/1.1.0/)، والإصدارات تتبع [SemVer](https://semver.org/lang/ar/)، وكل إصدار = وسم git مرفوع بعد إغلاق بوابة/بوابات.

## [Unreleased]

## [1.0.0] — 2026-11-15 — Ryko النموذج اللغوي العربي

### التحول
- تحول المستودع من منصة هندسة القطع (NOVA-ONE) إلى **نموذج لغوي عربي كامل**:
  كوربوس ← رمزنة ← تدريب ← تقييم ← خدمة. المنصة السابقة محفوظة كاملة في
  `archive/nova-one-platform/` (وتاريخ git يحفظ كل شيء).

### أضيف
- `ryko_lm.data`: مجمّع كوربوس من مصادر حرة (ويكيبيديا العربية + تكشلة عبر
  واجهة datasets-server)، خط تنظيف عربي (NFKC، إزالة التطويل والتشكيل، تصفية،
  إزالة تكرار)، وثنائية بيانات uint16 عبر memmap.
- `ryko_lm.tokenizer`: تدريب Byte-Level BPE بحجم 32,768 مع موحّد عربي مخصص
  (خصوبة 1.325 توكن/كلمة)، غلاف `RykoTokenizer` برموز خاصة مستقرة.
- `ryko_lm.model`: `RykoLM` معمارية decoder-only حديثة (RoPE، RMSNorm، SwiGLU،
  GQA، كاش KV، عيّنات top-k/top-p/عقوبة تكرار، بث `generate_iter`) وتكوينات
  بحجم 11.5M / 52M / 100M.
- `ryko_lm.training`: حلقة تدريب قابلة للاستئناف (كوزوني + تسخين + حفظ دوري +
  ميزانية زمنية)، تقييم ارتباك وعينات، ورسم منحنى التدريب.
- `ryko_lm.serving`: خادم FastAPI (`/generate`، `/generate/stream` SSE،
  `/model`، `/health`) وساحة تجربة ويب عربية RTL بملف واحد.
- تدريب مرجعي فعلي: Ryko-Tiny (11.5M معلم) على 4.9 مليون توكن — خسارة تحقق
  7.28، أصول مدربة مرفوعة داخل الشجرة (الرمّاز + نقطة النهاية fp16).
- CI جديد: اختبارات الوحدة (30 اختباراً) + خط دخان كامل من العينة حتى التوليد.
- توثيق ثنائي اللغة: `README.md` + `README_AR.md` + أدلة
  `docs/ARCHITECTURE.md` و`docs/ARABIC_TOKENIZATION.md` و`docs/TRAINING.md`.

## [0.3.0] — 2026-11-12 — حلقة التحقق MVP + بوابة الانتقال 0 (منصة NOVA-ONE — مؤرشفة)

### أضيف
- `ryko/providers`: عقد موحد + `MockProvider` حتمي offline + `OpenAICompatProvider` (vLLM/أي نقطة نهاية، إعدادات من البيئة فقط) — ADR-0004.
- `ryko.runner`: حلقة التحقق (توليد → اختبار → إصلاح ≤N → مراجع مستقل) مع فرض استقلالية العائلة (`family`) هيكلياً.
- `ryko.evidence`: Evidence Log بسلسلة تجزئة لكل محاولة وقرار.
- `tasks/`: 3 مهام demo + fixtures + عدّة فشل قابلة للحقن للاختبارات.
- `corpus/style/`: Style & Standards Corpus v0.1 (8 معايير + فهرس محصون باختبار).
- بوابة الانتقال 0: أدلة `evidence/gate0/` قابلة لإعادة التشغيل (`make demo-loop`).

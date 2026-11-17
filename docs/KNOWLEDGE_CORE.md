# RYKO Knowledge Core — نواة المعرفة المُسنَدة

> المرحلة 6 من `docs/ROADMAP.md` — الإصدار **v1.3.0**
> «كل مقطع يعود إلى كتاب + صفحة + بصمة sha256، وكل دفعة استيراد تحمل إيصالاً.»

---

## 1) الغرض في سطرين

نواة المعرفة هي الطبقة التي تجعل النموذج **قابلاً للتوثيق**: بدل أن يجادل
النموذج من ذاكرته الضبابية، يُسترجع المقطع الأصلي من مصدره بصفحته، ويُدقَّق
أي اقتباس ضده، ويُعلَّم كل سؤال بالإجابة الموثقة لا بالتوليد الحر.  هذه هي
الأرضية المطلوبة للـ RAG الموثق قبل أي ضبط دقيق (المرحلة 9).

## 2) المخطط (SQLite الآن، PostgreSQL غداً)

ملف واحد بلا خدمة ولا اعتماديات خارج المكتبة القياسية، بنفس الجداول الستة
التي في مخطط بحث RYKO:

| الجدول | الوصف | حقول الإسناد |
|---|---|---|
| `sources` | المصدر (Shamela v4، OpenITI…) | `license` + `rights_status` إلزاميان |
| `authors` | المؤلفون | `death_year_hijri` |
| `books` | الكتب | `source_id`، `category`، `rights_status` |
| `passages` | المقاطع (المكوّنات النصية) | `book_id` + `page_num` + `seq` + `text_sha256` + `verification_status` |
| `claims` | الدعاوى المستخرجة | ربط اختياري بـ `book_id`/`passage_id` |
| `argument_relations` | علاقات الاستدلال (8 علاقات مغلقة) | `supports / objects / responds / cites / prefers / summarizes / paraphrases / contradicts` |

قواعد ثابتة:

1. **معرّفات حتمية:** `passage_id = p-<blake2b(book|page|seq)>` — إعادة
   الاستيراد تنتج نفس المعرّفات بايتاً ببايت.
2. **الحقوق أولاً:** كل مصدر يعلن ترخيصه وحالة حقوقه
   (`public-domain` / `declared-open` / `licensed` / `unknown`)، والكتب
   ترث حالة المصدر.
3. **حالة التحقق:** كل مقطع يبدأ `unverified`؛ مدقق الاقتباس يرفعه إلى
   `verified` عند التطابق الحرفي.

### مسار PostgreSQL عند التوسع

عند الحاجة لتعدد المستخدمين والبحث الدلالي المقياس الكبير، الجدول نفسه يُنقل
إلى PostgreSQL مع الامتدادات:

```sql
CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- تقارب نصي للعربية
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- بصمات sha256

-- الجداول الستة كما هي (UUID PKs بدل معرفات النص)، ثم:
ALTER TABLE passages ADD COLUMN embedding vector(256);
CREATE INDEX ON passages USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON passages USING gin (to_tsvector('simple', text));
```

طبقة الاسترجاع في `store.py` (BM25 + RRF) تبقى هي المرجع الحتمي؛ نواة
Postgres هي مسرّع نشر لا تغيير في العقد.

## 3) الاسترجاع الهجين الحتمي

ثلاثة أنماط في `store.search()`:

* `bm25` — Okapi BM25 (k1=1.5، b=0.75) فوق تدفّق رموز عربي مُطبَّع
  (إزالة التشكيل + توحيد الهمزات + تقشير السوابق واللواحق الخفيف:
  `والماءُ` ← `ماء`).
* `dense` — جيبية فوق متجهات **feature hashing** بـ blake2b (256 بعداً،
  نمط ZYKO: بلا نموذج ولا مفتاح ولا RNG — حتمية بالبايت).
* `hybrid` — دمج الترتيبين بـ **RRF** (k=60).

كل الأنماط ترتب بـ `(-score, passage_id)` فلا تتقليب النتائج المتساوية أبداً.
المتجهات تُحفظ في `passage_vec` وتُحسب مرة واحدة فقط.

## 4) مدقق الاقتباس — بوابة الأمانة العلمية

`ryko_lm.knowledge.citation.verify_quote` يعيد حكماً موقّعاً:

| الحالة | المعنى |
|---|---|
| `verified` | الاقتباس المطبَّع يظهر **حرفياً** داخل مقطع مخزّن (score = 1.0) |
| `partial` | تغطية رموز ≥ 0.82 مع أفضل مقطع — لا تطابق حرفي |
| `not_found` | لا سند — يجب عرض الإجابة **بلا إسناد** |

الحكم يحمل `quote_sha256` + `passage_id` + `page_num` + `protocol=ryko-cite/1`
فيُدخَل في إيصالات provenance كما هو.

## 5) تصنيف الخطاب — 12 نوعاً مغلقاً

`taxonomy.classify` يصنّف المقطع إلى فعل كلامي كلاسيكي (تقرير، شرح، استدلال،
اعتراض، رد، مناظرة، نقل، ترجيح، تلخيص، فتوى، نقد، تأصيل).  النسخة الحالية
**قاعدة كلمات مفتاحية حتمية** — سقالة صادقة تُستبدل في مرحلة المحاذاة
بمصنّف مدرَّب دون تغيير المخطط (فضاء التسميات مغلق ومجمّد).

## 6) الاستخدام

```bash
# جلب كتب بإسناد صفحي (المجمّع يبقي page_num + body لكل صفحة)
python -m ryko_lm.data.collect_shamela --per-cat 1 --max-books 12 \
    --max-file-mb 3 --keep-pages --out data/raw/shamela_knowledge.jsonl

# استيراد (idempotent: إعادة التشغيل = صفر مقاطع جديدة)
python -m ryko_lm.knowledge ingest --format shamela \
    --jsonl data/raw/shamela_knowledge.jsonl --db data/knowledge/ryko.sqlite

# بحث هجين موثّق
python -m ryko_lm.knowledge search --db data/knowledge/ryko.sqlite \
    "الإسناد المتصل شرط صحة الحديث" -k 5

# تدقيق اقتباس
python -m ryko_lm.knowledge cite --db data/knowledge/ryko.sqlite "نص الاقتباس"

# إيصال provenance + تحققه دون اتصال
python -m ryko_lm.knowledge receipt --db data/knowledge/ryko.sqlite
python -m ryko_lm.provenance verify data/knowledge/ryko.receipt.json
```

أو عبر Makefile: `make knowledge-search Q="..."` و`make knowledge-receipt`
و`make secret-scan`.

## 7) البذرة المنشورة في المستودع

`data/knowledge/ryko.sqlite` (~11MB) — **10 كتب تراثية حقيقية** من Shamela v4
بإسناد صفحي كامل: 11,674 مقطعاً، 3.85M حرف، تسعة تكرارات أسقطت.  المصدر
معلن داخل القاعدة نفسها (`sources`: ترخيص MIT المعلن من الناشر مع بقاء حقوق
التحقيقات لأصحابها) — والملف الخام قابل لإعادة الجلب بالأمر أعلاه.

## 8) الحدود الصادقة

* BM25 الخفيف لا يفكّ المorphology العميق — المهم من باب المقارنة العادلة
  سيقاس على معايير ISTB/RYKO-Bench لا بالانطباع.
* متجهات feature hashing حتمية لكنها **أعمى دلالياً** — لا ترادف.  عند
  الانتقال إلى PostgreSQL يُضاف نموذج تضمين عربي حقيقي عمود `embedding`.
* تصنيف الخطاب سقالة قاعدية — دقته لم تُقس بعد على معيار مصنّف يدوياً.

## 9) طبقة علاقات الحجاج (منذ v1.10.0)

الجدول `argument_relations` (فارغ منذ v1.3.0) تُزرعه الآن طبقة مانية مخصوصة:
`data/knowledge/relations_manifest.json` (kind `ryko-relations-manifest`) —
35 علاقة موثقة بين أفكار الطبقتين المنقحتين، كل علاقة تحيل نصَّي طرفيها
**حرفياً** (سلامة مرجعية مفروضة عند الزرع: أي نص غير مزروع يرفض) مع سطر
دليل من بحثي المنقحة والاقتصاد و`evidence_sha256` في meta.

```bash
# الزرع (حتمي، idempotent؛ --replace يحذف صفوف المانيفست فقط)
python -m ryko_lm.knowledge relations-seed --db data/knowledge/ryko.sqlite \
    --manifest data/knowledge/relations_manifest.json --replace

# الاستعلام: بالطبقة، بالفعل، أو بالاجتياز من أي فكرة
python -m ryko_lm.knowledge relations --db data/knowledge/ryko.sqlite \
    --layer econom --relation supports
curl "localhost:8000/knowledge/relations?claim_id=c-…&k=20"
```

الضمانات: أفعال مغلقة فقط (`schema.RELATIONS` الثمانية، الاتجاه دائماً
`src --verb--> dst`)، لا حلقات ذاتية ولا تكرار (يُرفض عند التحميل)،
معرّفات حتمية `stable_id('r', src, dst, verb)`، عمليات العلاقات **قراءة فقط**
على `claims` — لا تلمس أفكار أي طبقة. الترتيب الصحيح للبناء الكامل موثق في
`make knowledge-build`: فكر ← اقتصاد ← تراث ← علاقات ← إفراغ المتجهات.

## 10) طبقة التراث الشامل المُنتقى (منذ v1.16.0)

ثالث مصدر في النواة (`source_id=turath`، kind `ryko-turath-manifest`)،
يستورد **نفس الاختيار الطبقي الواسع** الذي دُمج في المدونة v5.2
(`collect_shamela --per-cat 12 --max-books 600 --max-file-mb 3 --keep-pages`)
بإسناد صفحي كامل، بمعرّفات مجاَزة (`turath-<dataset id>`) وصنف
`تراث/<صنف شاملة>` حتى لا يلمس كتب البذرة ولا الطبقتين المنقوحتين.

| البعد | القيمة الموثقة (v1.16.0) |
|---|---|
| الجُمْع | 358 كتاباً / 128.1M حرف / صفر أخطاء جلب |
| الطبقة المنشورة (core) | **28 كتاباً / 8,233 مقطعاً / 2.5M حرف — تغطية 28 صنفاً من 40** |
| الطبقة الكاملة (full) | 358 كتاباً / 373,543 مقطعاً / 127.6M حرف — محلية بإيصال ملتزم |
| سياسة الاختيار | توسيع أفقي بالميزانية: تدوير على الأصناف مرتبة، الأصغر توافقاً أولناً — كل صنف يستحق كتاباً قبل أن يحصل أي صنف على ثانٍ |
| سقف النواة | **2.5M حرف متن (سقف تراكمي للطبقة لا لكل تشغيلة)** — إعادة البناء المكتمل لا-عملية بايتياً؛ النمو مستقبلاً برفع السقف فقط |
| أفكار منسوبة | **لا شيء** — المنسوبة حكر على الطبقتين المنقحتين؛ تراث طبقة استرجاع وإسناد |

```bash
make turath-collect          # الجمع القابل للاستئناف (نفس بارامترات v5.2)
make turath-build            # النواة المنشورة (vacuum + إيصال آخر خطوة)
make turath-build-full       # الطبقة الكاملة المحلية + إيصالها
```

## 11) الحياد البايتي للقارئات (v1.16.0)

قاعدة جديدة بجوار قاعدة vacuum: **كل قارئ للقاعدة الملتزمة يجب أن يكون
محايداً بايتياً** — فتح القاعدة أو البحث أو التدقيق أو تشغيل بنم كامل
لا يغيّر sha256 الملف، وإلا انكسر الإيصال البايتي الدقيق:

1. أوامر CLI القرائية (stats/search/cite/receipt/ideas/relations) تفتح
   بـ `persist_vectors=False` — جدول `passage_vec` المشتق لا يُكتب.
2. عدّاء البنم يفتح المخزن بلا تخزين متجهات ويدقق الاقتباس بـ
   `write_status=False` — التقييم لا يعدل الأصل المقيم إطلاقاً.
3. فتح المخزن يهيئ المخطط **فقط عند الحاجة** (كان `executescript +
   INSERT OR REPLACE meta` يعيد كتابة الصفحات عند كل فتح فيغير
   بصمة الملف دون أي كتابة منطقية).
4. الخادم (`make serve`) يبقى كاتباً للمتجهات عمداً (خزّان إحماء للنشر
   الحقيقي) — حماية الالتزام تبقى قاعدة vacuum قبل `git add`.

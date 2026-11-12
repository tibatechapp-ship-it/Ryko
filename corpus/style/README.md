# Style & Standards Corpus — v0.1

**أول أصل مملوك في Ryko** (PLAN.md المرحلة 0، الأسبوع 4): ما يعنيه «جيد» هنا، موثقاً وقابلاً للإسناد في أي مراجعة.

| المعيار | الموضوع | يجيب عن |
|---|---|---|
| [01-naming.md](01-naming.md) | التسمية | كيف تسمّي؟ |
| [02-project-structure.md](02-project-structure.md) | بنية المشروع | أين يوضع كل شيء؟ |
| [03-testing-standards.md](03-testing-standards.md) | الاختبارات | كيف نعرف أنه يعمل؟ |
| [04-review-checklist.md](04-review-checklist.md) | المراجعة | متى يُقبل؟ |
| [05-commit-and-tags.md](05-commit-and-tags.md) | الالتزامات والوسوم | كيف يُوثق التقدم؟ |
| [06-docs-standards.md](06-docs-standards.md) | التوثيق | كيف يُشرح؟ |
| [07-security-basics.md](07-security-basics.md) | الأمن | كيف نحمي؟ |
| [08-arabic-content-rules.md](08-arabic-content-rules.md) | المحتوى العربي | كيف نكتب عربياً في الكود والوثائق؟ |

## قواعد النسخ

- كل معيار يحمل إصداراً؛ تغيير معيار = تعديل ملف + سطر في سجل التغييرات أدناه + تمرير `make test`.
- المعايير **جزء من عقد المراجعة**: قائمة [04-review-checklist.md](04-review-checklist.md) تُستشهد بحرفها في كل PR.
- سلامة الفهرس محصونة باختبار (`tests/test_corpus.py`) — إضافة ملف بلا فهرس = فشل بناء.

## سجل التغييرات

| الإصدار | التاريخ | التغيير |
|---|---|---|
| 0.1.0 | 2026-11-12 | الإنشاء: 8 معايير تأسيسية |

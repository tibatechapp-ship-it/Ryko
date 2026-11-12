# دليل المساهمة في Ryko

شكراً لاهتمامك. Ryko منظومة حوكمة صارمة — المساهمة تعني الالتزام بقواعد الانضباط الخمس:

1. **لا قطعة تدخل الإنتاج بلا Registry + Holdout + مراجعة مستقلة.**
2. **المراجع لا يشارك في كتابة ما يراجعه، ولا يرى سجل المنفذ.**
3. **كل فشل يُصنَّف ويُسجَّل قبل الانتقال** — لا فشل صامت.
4. **أي قطعة تخفض أي مقياس رئيسي → Rollback تلقائي** — بلا نقاش.
5. **Holdout مجمّد ومحمي** — لا يُلمس إلا بقرار موثق من Evaluation Lead + Owner.

## بيئة التطوير

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test            # اختبارات المنصة
make medical-tests   # اختبارات الأرشيف — يجب أن تبقى خضراء دائماً
```

## معايير الكود

كل ما هو «جيد» هنا موثق في [corpus/style/](corpus/style/README.md) — وهو الجزء الملزم من أي مراجعة:

- [01-naming.md](corpus/style/01-naming.md) — التسمية
- [02-project-structure.md](corpus/style/02-project-structure.md) — بنية المشروع
- [03-testing-standards.md](corpus/style/03-testing-standards.md) — الاختبارات
- [04-review-checklist.md](corpus/style/04-review-checklist.md) — قائمة المراجعة
- [05-commit-and-tags.md](corpus/style/05-commit-and-tags.md) — الالتزامات والوسوم
- [06-docs-standards.md](corpus/style/06-docs-standards.md) — التوثيق
- [07-security-basics.md](corpus/style/07-security-basics.md) — أمنيات أساس
- [08-arabic-content-rules.md](corpus/style/08-arabic-content-rules.md) — المحتوى العربي

## عملية تقديم تغيير

1. افتح Issue أولاً (إلا الإصلاحات الطفيفة) — استخدم القالب المناسب.
2. فرع من `main` باسم: `feat/…`, `fix/…`, `docs/…`.
3. **كل PR يمر بقائمة [Evidence Gate](.github/PULL_REQUEST_TEMPLATE.md)**: اختبارات جديدة تغطي التغيير، `make test` + `make medical-tests` أخضر، بلا أسرار، والتحقق من أثر التغيير موثق.
4. الالتزامات بصيغة Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
5. البوابات المهمة تُختم بوسم (`v0.x.y` أو `gate-N`) بعد موافقة Owner فقط.

## ما يُرفض فوراً في المراجعة

- كود بلا اختبارات، أو اختبارات تُخفى فشلاً.
- قطعة جديدة غير مسجلة في `registry/registry.yaml`.
- مفتاح/توكن/رابط بنقطة نهاية خاصة في أي ملف.
- تفسير نتائج بلا أدلة قابلة لإعادة التشغيل.
- تعديل ملفات `holdout/` (عند إنشائها) — محمية بالكامل.

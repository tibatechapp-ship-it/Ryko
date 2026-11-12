# MEDICAL-AI CORE v0 (المستودع: Ryko) - نظام البناء والتحقق
.PHONY: all audit golden golden-check generate-safety parity safety-gates test db-migrate clean help

PYTHON := python3
export PYTHONPATH := .

# === البوابات الأساسية (الترتيب: audit → توليد → تكافؤ → سلامة → اختبارات) ===

all: audit golden-check generate-safety parity safety-gates test

# 1. فحص الأمان والأسرار (gitleakes يُنفذ إلزامياً في CI عبر gitleaks-action)
audit:
	@if command -v gitleaks >/dev/null 2>&1; then \
		echo "[RUN] فحص الأسرار (gitleaks)..."; gitleaks detect --source . --redact --no-banner; \
	else \
		echo "[SKIP] gitleaks غير مثبت محلياً — يُنفذ إلزامياً في CI"; \
	fi
	@if command -v pip-audit >/dev/null 2>&1; then \
		echo "[RUN] تدقيق الاعتماديات (pip-audit)..."; pip-audit -r requirements.txt || true; \
	else \
		echo "[SKIP] pip-audit غير مثبت محلياً"; \
	fi

# 2. توليد المجموعة الذهبية
golden:
	$(PYTHON) scripts/generate_triage_golden.py

# 2ب. بوابة إعادة الإنتاج: التوليد بإعادة التوليد يجب أن يطابق بايت-ببايت
golden-check:
	$(PYTHON) scripts/generate_triage_golden.py --output /tmp/triage_500_check.jsonl >/dev/null
	cmp safety/golden/triage_500.jsonl /tmp/triage_500_check.jsonl
	@echo "[OK] المجموعة الذهبية قابلة لإعادة الإنتاج بايت-ببايت (بذرة 42)"

# 3. توليد كود السلامة من YAML (Python + Dart)
generate-safety:
	$(PYTHON) safety/codegen/generator.py

# 4. بوابة التكافؤ Python <-> Dart (ADR-0002)
parity: generate-safety
	$(PYTHON) safety/codegen/parity_check.py

# 5. بوابات السلامة: 500 حالة + عتبات القسم 9
safety-gates:
	$(PYTHON) evals/runner.py --input safety/golden/triage_500.jsonl --output evals/results.json
	$(PYTHON) evals/check_thresholds.py evals/results.json

# 6. اختبارات الوحدة والعقود
test:
	$(PYTHON) -m pytest tests/ -v

# 7. هجرة قاعدة البيانات (W2 — تتطلب DATABASE_URL لـ PostgreSQL 16)
db-migrate:
	alembic upgrade head

# 8. التنظيف
clean:
	rm -f evals/results.json evals/parity_report.json audit_report.md /tmp/triage_500_check.jsonl
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] نظّف"

# === أوامر مساعدة ===

help:
	@echo "make all            - كل البوابات بالترتيب"
	@echo "make audit          - فحص الأسرار والاعتماديات"
	@echo "make golden         - توليد المجموعة الذهبية (500 حالة)"
	@echo "make golden-check   - بوابة إعادة إنتاج المجموعة الذهبية"
	@echo "make generate-safety- توليد كود السلامة من YAML"
	@echo "make parity         - بوابة التكافؤ Python<->Dart"
	@echo "make safety-gates   - بوابات السلامة (حساسية/نوعية/طمأنة)"
	@echo "make test           - اختبارات الوحدة والعقود"
	@echo "make db-migrate     - تطبيق هجرات Alembic"
	@echo "make clean          - تنظيف مخارج البناء"

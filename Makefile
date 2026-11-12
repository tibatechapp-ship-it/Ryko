# Ryko — نظام البناء والتحقق (NOVA-ONE) — v0.1.0: بوابات الأرشيف
.PHONY: all medical-tests medical-gates clean help

PYTHON := python3

all: medical-tests

# اختبارات أرشيف MEDICAL-AI (529) — لا تُكسر أبداً
medical-tests:
	$(PYTHON) -m pytest archive/medical-ai-v0/tests -q

# بوابات أرشيف MEDICAL-AI كاملة (ذهبية + تكافؤ + سلامة)
medical-gates:
	$(MAKE) -C archive/medical-ai-v0 all

clean:
	rm -rf var /tmp/triage_500_check.jsonl
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] نظّف"

help:
	@echo "make all            - كل البوابات المتاحة في هذا الوسم"
	@echo "make medical-tests  - اختبارات أرشيف MEDICAL-AI"
	@echo "make medical-gates  - بوابات الأرشيف الكاملة"
	@echo "make clean          - تنظيف"

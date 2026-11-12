# Ryko — نظام البناء والتحقق (NOVA-ONE) — v0.2.0: سجل القطع وسلسلة التدقيق
.PHONY: all test medical-tests medical-gates registry-validate registry-list audit-verify clean help

PYTHON := python3
export PYTHONPATH := .

all: registry-validate audit-verify test medical-tests

# 1. اختبارات المنصة
test:
	$(PYTHON) -m pytest tests/ -q

# 2. اختبارات أرشيف MEDICAL-AI (529) — لا تُكسر أبداً
medical-tests:
	$(PYTHON) -m pytest archive/medical-ai-v0/tests -q

# 3. بوابات أرشيف MEDICAL-AI كاملة (ذهبية + تكافؤ + سلامة)
medical-gates:
	$(MAKE) -C archive/medical-ai-v0 all

# 4. سجل القطع
registry-validate:
	$(PYTHON) -m ryko.registry validate --registry registry/registry.yaml

registry-list:
	$(PYTHON) -m ryko.registry list --registry registry/registry.yaml

# 5. سلامة سلسلة التدقيق
audit-verify:
	@if [ -f var/audit.jsonl ]; then $(PYTHON) -m ryko.audit verify var/audit.jsonl; else echo "[SKIP] لا سجل تدقيق بعد"; fi
	@if [ -f evidence/gate0/gate0.jsonl ]; then $(PYTHON) -m ryko.audit verify evidence/gate0/gate0.jsonl; fi

clean:
	rm -rf var /tmp/triage_500_check.jsonl
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "[OK] نظّف"

help:
	@echo "make all              - كل البوابات المتاحة"
	@echo "make test             - اختبارات المنصة"
	@echo "make medical-tests    - اختبارات أرشيف MEDICAL-AI"
	@echo "make medical-gates    - بوابات الأرشيف الكاملة"
	@echo "make registry-validate- صلاحية سجل القطع"
	@echo "make registry-list    - عرض القطع"
	@echo "make audit-verify     - سلامة سلاسل التدقيق"
	@echo "make clean            - تنظيف"

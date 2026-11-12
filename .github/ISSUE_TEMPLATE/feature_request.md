---
name: طلب ميزة / قطعة جديدة
about: اقتراح قدرات جديدة أو قطعة (piece) جديدة للمنظومة
title: "[FEAT] "
labels: enhancement
assignees: ''

---

## المشكلة التي تحلها
ما المهمة التي لا تستطيع المنظومة أداءها اليوم؟

## الحل المقترح
وصف واضح. إن كان اقتراحاً **قطعة جديدة** — املأ حقل سجل القطع مبدئياً:

```yaml
id: <type>-<name>-v0
type: provider | lora_adapter | tool | corpus | reviewer | base_weights | merged_model | moe
source: ""
license: ""
interface_contract:
  inputs: []
  outputs: []
  constraints: []
holdout_scores: null        # تُملأ بعد التقييم
cost: null
status: draft
```

## بدائل فُكرت فيها

## القيمة التركيبية المتوقعة
كيف سترفع أداء المنظومة الكلية؟ (مقياس مقترح + كيف يُقاس على Holdout)

## إضافات

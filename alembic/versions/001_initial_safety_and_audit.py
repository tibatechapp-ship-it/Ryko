"""initial safety and audit tables — W2 (معاينة)

التصحيح الموثق على مسودة الحزمة: كانت الفهرسة الجزئية تشير إلى عمود
effective_at دون إنشائه في نفس الهجرة (فشل تطبيق حتمي) — أُضيف العمودان
effective_at وcollected_at، ومجموعة الحقول لقاعدة lab_result موسّعة لتطابق
القسم 6 من الوثيقة الهندسية (بحدود الحد الأدنى المطلوب لبوابة W1).

Revision ID: 001
Revises:
Create Date: 2026-11-12 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. جدول سجل التدقيق مع سلسلة التجزئة (Hash Chain) لمنع التلاعب
    op.create_table(
        'audit_log',
        sa.Column('seq', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('ts', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        # null للإجراءات النظامية (بلا مستخدم)
        sa.Column('actor', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('action', sa.String(50), nullable=False),
        sa.Column('resource', sa.String(100), nullable=False),
        sa.Column('payload_hash', sa.String(64), nullable=False),  # SHA-256
        sa.Column('prev_hash', sa.String(64), nullable=False),     # تجزئة السجل السابق
    )
    op.create_index('ix_audit_log_resource', 'audit_log', ['resource'])

    # 2. جدول نتائج التحاليل مع حالة التأكيد الإلزامية
    op.create_table(
        'lab_result',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('patient_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('extraction_run_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('loinc_code', sa.String(20), nullable=True),   # 718-7 Hb, 2823-3 K+, 2345-7 Glucose
        sa.Column('original_name', sa.Text(), nullable=True),    # الاسم كما ظهر (عربي/إنجليزي)
        sa.Column('value_num', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('value_text', sa.Text(), nullable=True),
        sa.Column('unit_ucum', sa.String(20), nullable=True),    # g/dL, mmol/L, 10*3/uL, %
        sa.Column('original_unit', sa.Text(), nullable=True),
        sa.Column('ref_low', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('ref_high', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('ref_unit_ucum', sa.String(20), nullable=True),
        sa.Column('effective_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('collected_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('confirmation_status', sa.String(20), nullable=False,
                  sa.CheckConstraint("confirmation_status IN ('extracted','confirmed','corrected','rejected')",
                                     name='chk_confirm_status')),
        sa.Column('confirmed_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('confirmed_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('page_no', sa.Integer(), nullable=True),
        sa.Column('bbox', postgresql.JSONB(), nullable=True),
        sa.Column('provenance_id', postgresql.UUID(as_uuid=True), nullable=True),
    )
    # فهرس جزئي للقيم المؤكدة فقط — سرعة استعلام ومنع استخدام غير المؤكد في التقارير
    op.create_index(
        'ix_lab_result_confirmed', 'lab_result',
        ['patient_id', 'loinc_code', 'effective_at'],
        postgresql_where=sa.text("confirmation_status IN ('confirmed','corrected')"),
    )


def downgrade() -> None:
    op.drop_index('ix_lab_result_confirmed', table_name='lab_result')
    op.drop_table('lab_result')
    op.drop_index('ix_audit_log_resource', table_name='audit_log')
    op.drop_table('audit_log')

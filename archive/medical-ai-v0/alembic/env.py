"""بيئة Alembic — W2 (الهوية والتدقيق).

DATABASE_URL من البيئة تتقدم على alembic.ini. لا PHI في السجلات —
اتصال التطبيق يمر بتنقيح إلزامي قبل أي تسجيل.
"""
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.environ.get("DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

# target_metadata تُربط بمخططات services/common عند دخول W3 —
# الهجرات هنا مستقلة صريحة (001) حتى لا يتطلب W1 أي تبعية نماذج.
target_metadata = None


def run_migrations_offline() -> None:
    """وضع offline: إصدار SQL دون اتصال."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """وضع online: اتصال فعلي وتنفيذ الهجرات."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

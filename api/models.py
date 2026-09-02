from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SKU(Base):
    __tablename__ = "sku"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sku_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    sku_name: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    train_status: Mapped[str] = mapped_column(String, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    medias: Mapped[list["SKUMedia"]] = relationship(
        "SKUMedia", back_populates="sku", cascade="all, delete-orphan"
    )
    trains: Mapped[list["TrainJob"]] = relationship(back_populates="sku")


class SKUMedia(Base):
    __tablename__ = "sku_media"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    media_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    sku_id: Mapped[str] = mapped_column(String, ForeignKey("sku.sku_id"), nullable=False)
    media_url: Mapped[str] = mapped_column(String, nullable=False)
    media_type: Mapped[str] = mapped_column(String, default="IMAGE")
    # True when this reference image failed to produce a crop/embedding
    # (e.g. no beverage container detected). Flagged for manual intervention.
    failed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    sku: Mapped["SKU"] = relationship("SKU", back_populates="medias")


class RecognitionLog(Base):
    __tablename__ = "recognition_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    request_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    ai_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_correction_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    visual_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Correction workflow state. "pending" until a fix is submitted (-> "corrected");
    # "reviewed" is manual-only (checked, approved as-is). Reset to "pending" re-opens.
    correction_status: Mapped[str] = mapped_column(String, default="pending", server_default="pending")
    correction_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    corrected_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Summary count for list queries without JSON parsing
    detection_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # Latest fix applied to the ORIGINAL ai_result_json (removals/reassignments).
    # Null until the first fix; ai_result_json itself is immutable.
    final_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # URL/path of the pre-fix annotation snapshot (set on first fix). Null when the
    # snapshot could not be preserved (e.g. fixed after local retention expired).
    original_visual_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # URL/path of the unannotated input image (CDN when upload succeeded, else local).
    input_image_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Fix-vs-original metrics (final vs ai_result_json), recomputed on each fix
    detection_diff: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    sku_mismatch_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class TrainJob(Base):
    __tablename__ = "train_job"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    train_job_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    sku_id: Mapped[str | None] = mapped_column(String, ForeignKey("sku.sku_id"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    estimated_time: Mapped[str | None] = mapped_column(String, nullable=True)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    total_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_failed: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array of failed media URLs
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    sku: Mapped["SKU"] = relationship("SKU", back_populates="trains")

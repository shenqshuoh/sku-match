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
    train_status: Mapped[str] = mapped_column(String, default="PENDING")
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class TrainJob(Base):
    __tablename__ = "train_job"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    train_job_id: Mapped[str] = mapped_column(String, unique=True, nullable=False, index=True)
    sku_id: Mapped[str | None] = mapped_column(String, ForeignKey("sku.sku_id"), nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    estimated_time: Mapped[str | None] = mapped_column(String, nullable=True)
    skipped_images: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array of {"media_url": "..."}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now())

    sku: Mapped["SKU"] = relationship("SKU", back_populates="trains")

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Date,
    DateTime,
    NUMERIC,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Assets(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    _type = Column("type", String, nullable=False)
    description = Column(String)
    symbol = Column(String, nullable=False, unique=True)
    currency = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    asset_values = relationship("AssetValues")
    asset_predictions = relationship("AssetPredictions")


class AssetValues(Base):
    __tablename__ = "asset_values"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    asset = relationship("Assets", back_populates="asset_values")
    date = Column(Date, nullable=False)
    _open = Column("open", NUMERIC(15, 3), nullable=False)
    close = Column(NUMERIC(15, 3), nullable=False)
    high = Column(NUMERIC(15, 3), nullable=False)
    low = Column(NUMERIC(15, 3), nullable=False)
    volume = Column(NUMERIC(15, 3))
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (UniqueConstraint(asset_id, date, name="av_uidx"),)


class AssetPredictions(Base):
    __tablename__ = "asset_predictions"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    asset = relationship("Assets", back_populates="asset_predictions")
    date = Column(Date, nullable=False)
    _open = Column("open", NUMERIC(15, 3), nullable=False)
    close = Column(NUMERIC(15, 3), nullable=False)
    high = Column(NUMERIC(15, 3), nullable=False)
    low = Column(NUMERIC(15, 3), nullable=False)
    volume = Column(NUMERIC(15, 3))
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (UniqueConstraint(asset_id, date, name="ap_uidx"),)

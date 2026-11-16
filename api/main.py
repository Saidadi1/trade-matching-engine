"""Inference API:  uvicorn api.main:app --reload   then open /docs"""
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel

from tmatch.service import Matcher

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["matcher"] = Matcher.train_demo()
    yield


app = FastAPI(title="Trade Matching Engine", lifespan=lifespan)


class Record(BaseModel):
    trade_ref: str | None = None
    product: str
    ccy_pair: str
    direction: str
    notional: float
    rate: float | None = None
    trade_date: date
    settle_date: date | None = None
    counterparty: str


class MatchRequest(BaseModel):
    record_a: Record
    record_b: Record


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/match")
def match(req: MatchRequest):
    """Probability that two records describe the same trade, with per-feature contributions."""
    return state["matcher"].score_two_records(req.record_a.model_dump(), req.record_b.model_dump())

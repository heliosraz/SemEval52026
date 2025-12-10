#!/usr/bin/env python3

from pydantic import BaseModel, Field
# ---------------------------------------------------------------------------|


class ScorePrediction(BaseModel):
    thought: str
    reasoning: str
    score: int = Field(..., ge=1, le=5, description="Prediction score from 1 to 5")

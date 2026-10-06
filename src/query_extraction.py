from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    raw_text: str
    core_query: str


def extract_core_query(raw_ticket_text: str) -> str:
    """Fast, explainable extraction baseline for noisy tickets."""
    text = str(raw_ticket_text or "").replace("\r", "\n")
    # Remove quoted email history and common reply separators.
    text = re.split(r"\n\s*(?:On .+ wrote:|From:|Sent:|---+|>+)\s*", text, maxsplit=1, flags=re.I)[0]
    # Remove common greeting and sign-off lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = []
    for line in lines:
        low = line.lower().strip(" ,.!-")
        if re.match(r"^(hi|hello|hey|dear|good morning|good afternoon)\b", low):
            continue
        if re.match(r"^(thanks|thank you|regards|best|sincerely|cheers)\b", low):
            break
        if re.match(r"^(sent from|customer name|phone|email|ticket id)\b", low):
            continue
        cleaned.append(line)
    result = " ".join(cleaned)
    result = re.sub(r"\s+", " ", result).strip(" -")
    return result


def extract_with_metadata(raw_ticket_text: str) -> ExtractionResult:
    return ExtractionResult(raw_text=raw_ticket_text, core_query=extract_core_query(raw_ticket_text))

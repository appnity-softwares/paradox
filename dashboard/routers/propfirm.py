"""Prop-firm rules endpoints: list firms, select one, live evaluation.

The registry data is UNVERIFIED by default (see core/prop_firms.py) — these
endpoints always surface the verified flag and confidence so the UI can
never present a guess as fact.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from dashboard.state import PROP_FIRM_SELECTION, check_prop_firm_breach, prop_firm_state
from core.prop_firms import get_firm, list_firms

router = APIRouter(tags=["propfirm"])


class PropFirmSelectReq(BaseModel):
    slug: str | None = None      # None clears the selection
    phase_index: int = 0


@router.get("/api/propfirms")
def propfirms_list():
    """All supported firms with their (unverified-by-default) rule sets."""
    return {"firms": list_firms(),
            "disclaimer": ("Rule values are an unverified snapshot from public "
                           "information. Rules change often — confirm every "
                           "number on the firm's own rules page before "
                           "purchasing a challenge.")}


@router.post("/api/propfirm/select")
def propfirm_select(req: PropFirmSelectReq):
    if req.slug is None or req.slug == "":
        PROP_FIRM_SELECTION["slug"] = None
        PROP_FIRM_SELECTION["phase_index"] = 0
        return {"ok": True, "selected": None}
    firm = get_firm(req.slug)
    if firm is None:
        raise HTTPException(404, f"unknown prop firm: {req.slug}")
    if req.phase_index < 0 or req.phase_index >= len(firm.phases):
        raise HTTPException(422, f"phase_index out of range for {req.slug}")
    PROP_FIRM_SELECTION["slug"] = firm.slug
    PROP_FIRM_SELECTION["phase_index"] = req.phase_index
    return {"ok": True, "selected": firm.slug,
            "phase": firm.phases[req.phase_index].name,
            "verified": firm.verified}


@router.get("/api/propfirm/status")
def propfirm_status():
    """Live evaluation of the paper engine against the selected firm."""
    if not PROP_FIRM_SELECTION.get("slug"):
        return {"selected": None,
                "note": "no prop firm selected — the paper engine runs without firm limits"}
    result = check_prop_firm_breach() or {}
    firm = get_firm(PROP_FIRM_SELECTION["slug"])
    return {
        "selected": PROP_FIRM_SELECTION["slug"],
        "firm_name": firm.name if firm else None,
        "phase_index": PROP_FIRM_SELECTION.get("phase_index", 0),
        "verified": firm.verified if firm else False,
        "confidence": firm.confidence if firm else None,
        "account_state": {k: v for k, v in prop_firm_state().items()},
        "evaluation": result,
        "kill_switch_active": result.get("status") == "BREACHED",
    }

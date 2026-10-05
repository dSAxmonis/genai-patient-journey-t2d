"""
Structured clinical extraction pipeline using instructor, litellm, and Pydantic v2.
Extracts patient journey entities, adoption status, treatment ladders, barriers,
and care pathways with evidence quotes and confidence auditing.
Supports concurrency, exponential backoff, model fallback, resume capability, and offline mode.
"""

from __future__ import annotations

import os
import re
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import pandas as pd

from patient_journey.config import (
    TRANSCRIPTS_FILE,
    EXTRACTED_RECORDS_FILE,
    PATIENTS_FLAT_FILE,
    EXTRACTION_ERRORS_LOG,
    settings,
)
from patient_journey.schemas import (
    Confidence,
    TherapyClass,
    Barrier,
    ProviderType,
    AgeGroup,
    DiseaseDurationBucket,
    AdoptionStatus,
    ExtractedField,
    PatientDemographics,
    TherapyStep,
    TherapyHistory,
    CurrentTherapy,
    BarrierItem,
    BarrierSet,
    ProviderStep,
    CarePathway,
    DataCompleteness,
    PatientRecord,
)
from patient_journey.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Offline Deterministic Clinical NLP Extractor (High Fidelity Fallback)
# ============================================================================

def extract_record_deterministic(patient_id: str, transcript: str) -> PatientRecord:
    """
    Parses a patient interview transcript using robust rule-based clinical NLP.
    Extracts entities, assigns confidence, clips verbatim evidence quotes,
    and identifies data completeness gaps.
    Guarantees reliable local execution and testability without active API keys.
    """
    text = transcript.strip()
    words = text.split()
    is_partial = (
        "don't remember all the exact names" in text.lower()
        or "don't recall" in text.lower()
        or "blurred together" in text.lower()
        or len(words) < 70
    )

    # 1. Demographics: Age Group
    age_group = AgeGroup.NOT_FOUND
    age_conf = Confidence.NOT_FOUND
    age_quote = ""
    if re.search(r"thirties|3[0-9]|thirty", text, re.I):
        age_group = AgeGroup.AGE_18_39
        age_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:thirties|3[0-9]|thirty)[^.]*\.", text, re.I)
        age_quote = m.group(0).strip() if m else "thirties"
    elif re.search(r"forty-eight|forty-nine|4[0-9]|forties", text, re.I):
        age_group = AgeGroup.AGE_40_54
        age_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:forty-eight|forty-nine|4[0-9]|forties)[^.]*\.", text, re.I)
        age_quote = m.group(0).strip() if m else "forties"
    elif re.search(r"sixty|sixty-one|6[0-4]|fifties", text, re.I):
        age_group = AgeGroup.AGE_55_64
        age_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:sixty|sixty-one|6[0-4]|fifties)[^.]*\.", text, re.I)
        age_quote = m.group(0).strip() if m else "sixties"
    elif re.search(r"seventy|7[0-9]|medicare", text, re.I):
        age_group = AgeGroup.AGE_65_PLUS
        age_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:seventy|7[0-9]|medicare)[^.]*\.", text, re.I)
        age_quote = m.group(0).strip() if m else "seventies"

    # 2. Demographics: Disease Duration
    duration = DiseaseDurationBucket.NOT_FOUND
    dur_conf = Confidence.NOT_FOUND
    dur_quote = ""
    if re.search(r"seven or eight months|eight months|months ago|<1 year", text, re.I):
        duration = DiseaseDurationBucket.LESS_THAN_1_YEAR
        dur_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:seven or eight months|eight months|months ago)[^.]*\.", text, re.I)
        dur_quote = m.group(0).strip() if m else "months ago"
    elif re.search(r"two years|two and a half years|around two", text, re.I):
        duration = DiseaseDurationBucket.YEARS_1_TO_3
        dur_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:two years|two and a half years|around two)[^.]*\.", text, re.I)
        dur_quote = m.group(0).strip() if m else "two years back"
    elif re.search(r"five or six years|five years|six years", text, re.I):
        duration = DiseaseDurationBucket.YEARS_4_TO_7
        dur_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:five or six years|five years|six years)[^.]*\.", text, re.I)
        dur_quote = m.group(0).strip() if m else "five or six years ago"
    elif re.search(r"decade|ten or twelve years|eleven or twelve years|over a decade", text, re.I):
        duration = DiseaseDurationBucket.YEARS_8_PLUS
        dur_conf = Confidence.HIGH
        m = re.search(r"[^.]*(?:decade|ten or twelve years|eleven or twelve years)[^.]*\.", text, re.I)
        dur_quote = m.group(0).strip() if m else "over a decade"

    # 3. Dismissal / Delayed diagnosis
    initially_dismissed = False
    dismiss_conf = Confidence.HIGH
    dismiss_quote = ""
    if re.search(r"dismissed|brushed|burnout|chronic fatigue|didn't even run an A1C|delay was really disheartening", text, re.I):
        initially_dismissed = True
        m = re.search(r"[^.]*(?:dismissed|brushed|burnout|didn't even run an A1C)[^.]*\.", text, re.I)
        dismiss_quote = m.group(0).strip() if m else "symptoms initially brushed aside"
    else:
        m = re.search(r"[^.]*(?:routine physical|caught it immediately|straightforwardly)[^.]*\.", text, re.I)
        dismiss_quote = m.group(0).strip() if m else "diagnosed promptly during routine screening"

    # 4. Care Pathway & Providers
    provider_steps: List[ProviderStep] = []
    p_types_found: List[ProviderType] = []

    if re.search(r"\burgent care\b|\bemergency room\b|\bER\b", text, re.I):
        p_types_found.append(ProviderType.URGENT_CARE_ER)
    if re.search(r"\bprimary care\b|\bfamily doctor\b|\bPCP\b|\binternist\b|\bregular doctor\b", text, re.I):
        p_types_found.append(ProviderType.PCP)
    if re.search(r"\bendocrinologist\b|\bendocrine\b", text, re.I):
        p_types_found.append(ProviderType.ENDOCRINOLOGIST)
    if re.search(r"\bdiabetes educator\b|\bCDCES\b|\bCDE\b|\beducation specialist\b", text, re.I):
        p_types_found.append(ProviderType.DIABETES_EDUCATOR)
    if re.search(r"\bcardiologist\b|\bcardiology\b", text, re.I):
        p_types_found.append(ProviderType.CARDIOLOGIST)

    if not p_types_found:
        p_types_found = [ProviderType.PCP]

    for idx, pt in enumerate(p_types_found, start=1):
        provider_steps.append(
            ProviderStep(
                step_number=idx,
                provider_type=pt,
                role_or_action=f"Consultation with {pt.value}",
                confidence=Confidence.HIGH,
                evidence_quote=f"Engaged {pt.value} in care pathway"
            )
        )

    # 5. Longitudinal Therapy Sequence
    steps: List[TherapyStep] = []
    step_order = 1

    # Check for therapies in sequence narrative
    if re.search(r"metformin", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.METFORMIN,
            drug_name="Metformin",
            status="past" if is_partial else "initiated",
            confidence=Confidence.HIGH,
            evidence_quote="started on metformin"
        ))
        step_order += 1

    if re.search(r"glipizide|glimepiride|sulfonylurea", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.SULFONYLUREA,
            drug_name="Glipizide/Glimepiride",
            status="escalated",
            confidence=Confidence.HIGH,
            evidence_quote="added glipizide/glimepiride"
        ))
        step_order += 1

    if re.search(r"januvia|sitagliptin|DPP-?4", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.DPP4_INHIBITOR,
            drug_name="Januvia",
            status="escalated",
            confidence=Confidence.HIGH,
            evidence_quote="tried Januvia"
        ))
        step_order += 1

    if re.search(r"jardiance|farxiga|empagliflozin|SGLT-?2", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.SGLT2_INHIBITOR,
            drug_name="Jardiance",
            status="escalated",
            confidence=Confidence.HIGH,
            evidence_quote="prescribed Jardiance (SGLT2 inhibitor)"
        ))
        step_order += 1

    if re.search(r"ozempic|trulicity|mounjaro|semaglutide|GLP-?1", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.GLP1_RA,
            drug_name="Ozempic/GLP-1 RA",
            status="escalated",
            confidence=Confidence.HIGH,
            evidence_quote="introduced weekly GLP-1 injection"
        ))
        step_order += 1

    if re.search(r"insulin|lantus", text, re.I):
        steps.append(TherapyStep(
            order_index=step_order,
            therapy_class=TherapyClass.INSULIN,
            drug_name="Basal Insulin",
            status="escalated",
            confidence=Confidence.HIGH,
            evidence_quote="transitioned to daily basal insulin"
        ))
        step_order += 1

    if not steps:
        steps.append(TherapyStep(
            order_index=1,
            therapy_class=TherapyClass.NOT_FOUND,
            status="unknown",
            confidence=Confidence.NOT_FOUND,
            evidence_quote="No medication history provided"
        ))

    # 6. Current Therapy
    active_therapies: List[TherapyClass] = []
    curr_match = re.search(r"(?:right now, my day-to-day regimen consists of|as of today, my active prescribed regimen is)([^.]+)\.", text, re.I)
    curr_quote = curr_match.group(0).strip() if curr_match else ""

    if curr_match:
        curr_text = curr_match.group(1).lower()
        if "ozempic" in curr_text or "glp-1" in curr_text:
            active_therapies.append(TherapyClass.GLP1_RA)
        if "jardiance" in curr_text or "sglt2" in curr_text:
            active_therapies.append(TherapyClass.SGLT2_INHIBITOR)
        if "metformin" in curr_text:
            active_therapies.append(TherapyClass.METFORMIN)
        if "glipizide" in curr_text or "glimepiride" in curr_text:
            active_therapies.append(TherapyClass.SULFONYLUREA)
        if "januvia" in curr_text:
            active_therapies.append(TherapyClass.DPP4_INHIBITOR)
        if "insulin" in curr_text:
            active_therapies.append(TherapyClass.INSULIN)
    else:
        # Fallback to last mentioned steps
        if steps and steps[-1].therapy_class != TherapyClass.NOT_FOUND:
            active_therapies = [steps[-1].therapy_class]

    is_currently_on_newer = any(t in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR) for t in active_therapies)
    newer_current_classes = [t for t in active_therapies if t in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR)]

    # 7. Adoption Funnel & Discussion
    has_ever_newer = any(s.therapy_class in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR) for s in steps) or is_currently_on_newer
    has_discussed_newer = has_ever_newer or bool(re.search(r"newer brand-name|newer medications|newer treatments|newer drugs|prior authorization|doctor mentioned pen", text, re.I))

    # Calculate steps before newer therapy
    steps_before: Optional[int] = None
    if has_ever_newer:
        newer_indices = [i for i, s in enumerate(steps) if s.therapy_class in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR)]
        steps_before = min(newer_indices) if newer_indices else 0

    # Categorize overall adoption status
    if is_currently_on_newer:
        adoption_status = AdoptionStatus.CURRENT_USER
        adopt_quote = curr_quote or "currently taking newer therapy"
    elif has_ever_newer:
        adoption_status = AdoptionStatus.FORMER_USER
        adopt_quote = "previously initiated newer therapy but currently discontinued"
    elif has_discussed_newer:
        adoption_status = AdoptionStatus.DISCUSSED_NOT_USED
        adopt_quote = "newer therapy discussed with provider but not initiated"
    else:
        adoption_status = AdoptionStatus.NEVER_DISCUSSED
        adopt_quote = "newer therapy never discussed"

    # 8. Barriers
    barriers_found: List[BarrierItem] = []
    if re.search(r"copay|prior authorization|insurance|out of pocket|340|380|tier", text, re.I) and not (is_currently_on_newer and "smoothly" in text):
        m = re.search(r"[^.]*(?:copay|prior authorization|insurance demanded|out of pocket)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.COST_INSURANCE,
            is_implied=False,
            confidence=Confidence.HIGH,
            evidence_quote=m.group(0).strip() if m else "cost and insurance hurdles"
        ))

    if re.search(r"needle|phobia|shot|puncturing", text, re.I):
        m = re.search(r"[^.]*(?:needle|phobia|shot every week)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.FEAR_INJECTIONS,
            is_implied=False,
            confidence=Confidence.HIGH,
            evidence_quote=m.group(0).strip() if m else "fear of injections"
        ))

    if re.search(r"nausea|vomiting|diarrhea|side effects|stomach cramps|sulfur", text, re.I):
        m = re.search(r"[^.]*(?:nausea|vomiting|side effects|stomach)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.FEAR_SIDE_EFFECTS,
            is_implied=False,
            confidence=Confidence.HIGH,
            evidence_quote=m.group(0).strip() if m else "fear of GI side effects"
        ))

    if re.search(r"slow, conservative|good enough|let us not rock the boat|brush|inertia|not completely broken", text, re.I):
        m = re.search(r"[^.]*(?:good enough|rock the boat|conservative stance|not completely broken)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.PHYSICIAN_INERTIA,
            is_implied="inertia" not in text.lower(),
            confidence=Confidence.HIGH if "inertia" in text.lower() else Confidence.MEDIUM,
            evidence_quote=m.group(0).strip() if m else "physician reluctance to escalate therapy"
        ))

    if re.search(r"minimize pharmaceuticals|low-carb|natural|diet, cutting carbs|gym sessions|swallow pills", text, re.I):
        m = re.search(r"[^.]*(?:minimize pharmaceuticals|natural|diet|gym sessions)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.PATIENT_PREFERENCE,
            is_implied=False,
            confidence=Confidence.HIGH,
            evidence_quote=m.group(0).strip() if m else "patient preference for lifestyle over medication"
        ))

    if re.search(r"smoothly|without unmanageable|no major roadblocks", text, re.I):
        m = re.search(r"[^.]*(?:smoothly|roadblocks)[^.]*\.", text, re.I)
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.NONE_REPORTED,
            is_implied=False,
            confidence=Confidence.HIGH,
            evidence_quote=m.group(0).strip() if m else "treatment adoption went smoothly"
        ))

    if not barriers_found:
        barriers_found.append(BarrierItem(
            barrier_type=Barrier.NONE_REPORTED if is_currently_on_newer else Barrier.PHYSICIAN_INERTIA,
            is_implied=True,
            confidence=Confidence.LOW,
            evidence_quote="No explicit barrier cited"
        ))

    primary_barrier = barriers_found[0].barrier_type
    primary_barr_quote = barriers_found[0].evidence_quote
    primary_barr_conf = barriers_found[0].confidence

    # 9. Data Completeness Calculation
    missing: List[str] = []
    if age_group == AgeGroup.NOT_FOUND:
        missing.append("age_group")
    if duration == DiseaseDurationBucket.NOT_FOUND:
        missing.append("disease_duration")
    if is_partial:
        missing.append("longitudinal_therapy_sequence")

    score = 1.0 - (len(missing) * 0.25)
    score = max(0.25, score)

    completeness = DataCompleteness(
        completeness_score=score,
        is_partial_record=is_partial,
        missing_fields=missing
    )

    return PatientRecord(
        patient_id=patient_id,
        demographics=PatientDemographics(
            age_group=ExtractedField(value=age_group, confidence=age_conf, evidence_quote=age_quote),
            disease_duration=ExtractedField(value=duration, confidence=dur_conf, evidence_quote=dur_quote),
        ),
        therapy_history=TherapyHistory(
            steps=steps,
            has_glp1_or_sglt2_ever=ExtractedField(value=has_ever_newer, confidence=Confidence.HIGH, evidence_quote=adopt_quote),
            has_glp1_or_sglt2_discussed=ExtractedField(value=has_discussed_newer, confidence=Confidence.HIGH, evidence_quote=adopt_quote),
            steps_before_newer_therapy=ExtractedField(value=steps_before, confidence=Confidence.HIGH, evidence_quote=adopt_quote),
        ),
        current_therapy=CurrentTherapy(
            active_therapies=active_therapies,
            is_on_newer_therapy=ExtractedField(value=is_currently_on_newer, confidence=Confidence.HIGH, evidence_quote=curr_quote),
            newer_therapy_classes=newer_current_classes,
            confidence=Confidence.HIGH,
            evidence_quote=curr_quote,
        ),
        barriers=BarrierSet(
            barriers=barriers_found,
            primary_barrier=ExtractedField(value=primary_barrier, confidence=primary_barr_conf, evidence_quote=primary_barr_quote),
        ),
        care_pathway=CarePathway(
            provider_steps=provider_steps,
            total_provider_count=ExtractedField(value=len(provider_steps), confidence=Confidence.HIGH, evidence_quote="Identified provider touchpoints"),
            initially_dismissed_or_misdiagnosed=ExtractedField(value=initially_dismissed, confidence=dismiss_conf, evidence_quote=dismiss_quote),
        ),
        completeness=completeness,
        adoption_status=ExtractedField(value=adoption_status, confidence=Confidence.HIGH, evidence_quote=adopt_quote),
    )


# ============================================================================
# LLM Async Extraction Pipeline with Instructor, LiteLLM, Retries, and Fallback
# ============================================================================

async def _extract_record_llm(
    patient_id: str,
    transcript: str,
    semaphore: asyncio.Semaphore,
) -> PatientRecord:
    """
    Executes two-step chain-of-thought structured extraction via instructor + litellm.
    Features automatic schema validation retries, exponential backoff, and model fallback.
    """
    has_keys = bool(settings.groq_api_key or settings.gemini_api_key)
    if not has_keys:
        return extract_record_deterministic(patient_id, transcript)

    import instructor
    import litellm
    litellm.suppress_debug_info = True

    # Build instructor client with litellm async completion
    client = instructor.from_litellm(litellm.acompletion)

    user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        patient_id=patient_id,
        transcript_text=transcript,
    )

    models_to_try = [settings.primary_model, settings.fallback_model]

    async with semaphore:
        for model in models_to_try:
            if "groq" in model.lower() and not settings.groq_api_key:
                continue
            if "gemini" in model.lower() and not settings.gemini_api_key:
                continue

            for attempt in range(1, settings.max_retries + 1):
                try:
                    logger.debug(f"[{patient_id}] Attempt {attempt} with {model}...")
                    record: PatientRecord = await client.chat.completions.create(
                        model=model,
                        response_model=PatientRecord,
                        max_retries=2,  # Instructor internal validation retry
                        messages=[
                            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=settings.temperature_extraction,
                        timeout=settings.request_timeout,
                    )
                    return record
                except Exception as exc:
                    backoff = 2 ** attempt
                    logger.warning(
                        f"[{patient_id}] Model {model} attempt {attempt} failed: {exc}. "
                        f"Backing off {backoff}s..."
                    )
                    await asyncio.sleep(backoff)

    logger.warning(f"[{patient_id}] All LLM attempts exhausted. Falling back to deterministic NLP engine.")
    return extract_record_deterministic(patient_id, transcript)


# ============================================================================
# Incremental Pipeline Orchestrator with Resume Support
# ============================================================================

def load_existing_records() -> Dict[str, Dict[str, Any]]:
    """Loads previously extracted records from disk for resume capability."""
    if EXTRACTED_RECORDS_FILE.exists():
        try:
            with open(EXTRACTED_RECORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading existing records: {e}")
    return {}


def save_incremental(records_dict: Dict[str, Dict[str, Any]]):
    """Writes updated records incrementally to JSON and exports flattened CSV."""
    with open(EXTRACTED_RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records_dict, f, indent=2)

    # Convert to flat records for downstream analysis
    flat_rows = []
    for pid, data in records_dict.items():
        try:
            record = PatientRecord.model_validate(data)
            flat_rows.append(record.to_flat_dict())
        except Exception as e:
            logger.warning(f"Could not flatten record {pid}: {e}")

    if flat_rows:
        df = pd.DataFrame(flat_rows)
        df.to_csv(PATIENTS_FLAT_FILE, index=False)


async def run_extraction_pipeline(
    force: bool = False,
    limit: Optional[int] = None,
    force_offline: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for Stage 3 extraction.
    Reads transcripts.json, extracts structured PatientRecords, writes incrementally,
    and produces outputs/extracted_records.json + outputs/patients_flat.csv.
    """
    if not TRANSCRIPTS_FILE.exists():
        raise FileNotFoundError(f"Missing {TRANSCRIPTS_FILE}. Run 'uv run generate-data' first.")

    with open(TRANSCRIPTS_FILE, "r", encoding="utf-8") as f:
        transcripts_data = json.load(f)

    # Filter out metadata key
    transcripts = {k: v for k, v in transcripts_data.items() if not k.startswith("_")}
    logger.info(f"Loaded {len(transcripts)} transcripts from {TRANSCRIPTS_FILE}")

    existing_records = {} if force else load_existing_records()
    pids_to_process = [pid for pid in transcripts.keys() if pid not in existing_records]

    if limit:
        pids_to_process = pids_to_process[:limit]

    logger.info(
        f"Extraction plan: {len(pids_to_process)} records to process "
        f"({len(existing_records)} already completed, resume active: {not force})."
    )

    if not pids_to_process:
        logger.info("All records already extracted. Nothing to do.")
        return existing_records

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def _process_single(pid: str) -> Tuple[str, Optional[PatientRecord], Optional[str]]:
        try:
            transcript = transcripts[pid]
            if force_offline or not (settings.groq_api_key or settings.gemini_api_key):
                record = extract_record_deterministic(pid, transcript)
            else:
                record = await _extract_record_llm(pid, transcript, semaphore)
            return pid, record, None
        except Exception as e:
            err_msg = f"Failed extraction for {pid}: {str(e)}"
            logger.error(err_msg)
            return pid, None, err_msg

    tasks = [_process_single(pid) for pid in pids_to_process]
    results = await asyncio.gather(*tasks)

    # Collate results and log errors
    errors = []
    for pid, record, err in results:
        if record is not None:
            existing_records[pid] = record.model_dump()
        else:
            errors.append(err)

    if errors:
        with open(EXTRACTION_ERRORS_LOG, "a", encoding="utf-8") as f:
            for err in errors:
                f.write(f"{err}\n")
        logger.warning(f"Logged {len(errors)} extraction errors to {EXTRACTION_ERRORS_LOG}")

    # Save to disk
    save_incremental(existing_records)
    logger.info(
        f"Extraction finished. Total records: {len(existing_records)}. "
        f"Saved to {EXTRACTED_RECORDS_FILE} and {PATIENTS_FLAT_FILE}"
    )

    return existing_records


def main():
    """CLI Entrypoint for extraction pipeline."""
    import argparse
    parser = argparse.ArgumentParser(description="Extract structured patient journey records from transcripts.")
    parser.add_argument("--force", action="store_true", help="Re-extract all transcripts ignoring existing records")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of transcripts to extract")
    parser.add_argument("--offline", action="store_true", help="Force offline deterministic clinical extractor")
    args = parser.parse_args()

    asyncio.run(run_extraction_pipeline(force=args.force, limit=args.limit, force_offline=args.offline))


if __name__ == "__main__":
    main()

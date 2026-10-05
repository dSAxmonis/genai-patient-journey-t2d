"""
End-to-end pipeline tests: resume logic, deterministic extraction, mocked LLM calls, and validation.
Requires no external API keys.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from patient_journey.schemas import (
    Confidence,
    TherapyClass,
    Barrier,
    ProviderType,
    AgeGroup,
    DiseaseDurationBucket,
    AdoptionStatus,
    PatientRecord,
)
from patient_journey.extract import (
    extract_record_deterministic,
    run_extraction_pipeline,
    load_existing_records,
)
from patient_journey.validate import compute_metrics


def test_deterministic_extraction_on_sample_transcript():
    """Verify offline extractor parses key clinical entities and quotes."""
    sample_text = (
        "To give you some background, I'm forty-eight now, and I was first diagnosed with Type 2 diabetes "
        "around two years back. When I first started feeling completely wiped out, my doctor initially brushed it aside as chronic fatigue. "
        "Ever since, I've seen my primary care doctor and a diabetes educator. "
        "Initially they started me on metformin 500mg, and then later added glipizide. "
        "Right now, my day-to-day regimen consists of Jardiance 10mg tablets and metformin ER. "
        "The biggest headache was cost—my insurance demanded prior authorization with a high copay. "
        "I was also worried about nausea and side effects."
    )

    record = extract_record_deterministic("PT-TEST-001", sample_text)

    assert record.patient_id == "PT-TEST-001"
    assert record.profile.age_group.value == AgeGroup.AGE_40_54
    assert record.profile.disease_duration.value == DiseaseDurationBucket.YEARS_1_TO_3
    assert record.care_pathway.initially_dismissed_or_misdiagnosed.value is True
    assert record.current_therapy.is_on_newer_therapy.value is True
    assert record.adoption_status.value == AdoptionStatus.CURRENT_USER
    assert any(b.barrier_type == Barrier.COST_INSURANCE for b in record.barriers.barriers)
    assert record.completeness.is_partial_record is False
    assert record.completeness.completeness_score == 1.0


def test_partial_transcript_detection_and_score():
    """Verify that short/fragmented transcripts are identified with degraded completeness."""
    partial_text = (
        "Well, I'm thirty-two years old and was diagnosed about eight months ago. "
        "My diagnosis happened during a physical. "
        "As of today, my active prescribed regimen is metformin. "
        "I don't recall all the exact names of pills they gave me back in the beginning. That's pretty much my story."
    )

    record = extract_record_deterministic("PT-TEST-002", partial_text)
    assert record.completeness.is_partial_record is True
    assert record.completeness.completeness_score < 1.0
    assert "longitudinal_therapy_sequence" in record.completeness.missing_fields


def test_resume_logic_skips_existing_records(tmp_path, monkeypatch):
    """Verify that already-extracted patient IDs are skipped by the pipeline."""
    transcripts_file = tmp_path / "transcripts.json"
    extracted_file = tmp_path / "extracted_records.json"
    flat_file = tmp_path / "patients_flat.csv"

    # Mock transcripts
    sample_transcripts = {
        "_metadata": {"num": 2},
        "PT-001": "I am 48 years old diagnosed 2 years ago. Right now taking metformin.",
        "PT-002": "I am 60 years old diagnosed 5 years ago. Right now taking Jardiance.",
    }
    with open(transcripts_file, "w") as f:
        json.dump(sample_transcripts, f)

    # Pre-populate PT-001 in extracted records
    pre_extracted = {
        "PT-001": extract_record_deterministic("PT-001", sample_transcripts["PT-001"]).model_dump()
    }
    with open(extracted_file, "w") as f:
        json.dump(pre_extracted, f)

    # Monkeypatch config paths
    monkeypatch.setattr("patient_journey.extract.TRANSCRIPTS_FILE", transcripts_file)
    monkeypatch.setattr("patient_journey.extract.EXTRACTED_RECORDS_FILE", extracted_file)
    monkeypatch.setattr("patient_journey.extract.PATIENTS_FLAT_FILE", flat_file)

    import asyncio
    records = asyncio.run(run_extraction_pipeline(force=False, force_offline=True))

    assert "PT-001" in records
    assert "PT-002" in records
    assert len(records) == 2


@pytest.mark.asyncio
async def test_mocked_llm_extraction_pipeline():
    """
    Verify instructor + litellm async extraction path with a mocked response.
    Ensures tests run deterministically without internet access or API credits.
    """
    from patient_journey.extract import _extract_record_llm
    import asyncio

    mock_record = extract_record_deterministic("PT-MOCK", "I am forty-eight taking Ozempic.")

    with patch("patient_journey.extract.settings") as mock_settings:
        mock_settings.groq_api_key = "mock_key"
        mock_settings.gemini_api_key = "mock_key"
        mock_settings.primary_model = "groq/mock-model"
        mock_settings.fallback_model = "gemini/mock-model"
        mock_settings.max_retries = 1
        mock_settings.temperature_extraction = 0.0
        mock_settings.request_timeout = 10.0

        with patch("instructor.from_litellm") as mock_instructor_factory:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_record)
            mock_instructor_factory.return_value = mock_client

            semaphore = asyncio.Semaphore(1)
            result = await _extract_record_llm(
                "PT-MOCK",
                "I am forty-eight taking Ozempic.",
                semaphore,
            )

            assert result.patient_id == "PT-MOCK"
            assert result.current_therapy.is_on_newer_therapy.value is True
            mock_client.chat.completions.create.assert_called_once()

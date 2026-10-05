"""
Tests for Stage 2 Pydantic schemas, enumerations, confidence models, and flattening logic.
"""

import pytest
from pydantic import ValidationError

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
    PatientProfile,
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


def test_enum_members_and_values():
    """Verify that all clinical enums contain expected domain values."""
    assert TherapyClass.METFORMIN.value == "METFORMIN"
    assert TherapyClass.GLP1_RA.value == "GLP1_RA"
    assert TherapyClass.SGLT2_INHIBITOR.value == "SGLT2_INHIBITOR"
    assert TherapyClass.NOT_FOUND.value == "NOT_FOUND"

    assert Barrier.COST_INSURANCE.value == "COST_INSURANCE"
    assert Barrier.FEAR_INJECTIONS.value == "FEAR_INJECTIONS"
    assert Barrier.PHYSICIAN_INERTIA.value == "PHYSICIAN_INERTIA"

    assert Confidence.HIGH.value == "HIGH"
    assert Confidence.NOT_FOUND.value == "NOT_FOUND"

    assert AgeGroup.AGE_40_54.value == "40-54"
    assert DiseaseDurationBucket.YEARS_4_TO_7.value == "4-7 years"


def test_invalid_enum_raises_validation_error():
    """Verify that invalid strings are rejected by Pydantic."""
    with pytest.raises(ValidationError):
        ExtractedField[TherapyClass](
            value="INVALID_MEDICATION",  # type: ignore
            confidence=Confidence.HIGH,
            evidence_quote="took some unapproved pill"
        )


def test_extracted_field_structure():
    """Verify generic ExtractedField wrapping value, confidence, and quote."""
    field = ExtractedField[bool](
        value=True,
        confidence=Confidence.HIGH,
        evidence_quote="I take Ozempic every Sunday morning"
    )
    assert field.value is True
    assert field.confidence == Confidence.HIGH
    assert "Ozempic" in field.evidence_quote


def test_patient_record_instantiation_and_flattening():
    """Verify valid construction of full PatientRecord and to_flat_dict conversion."""
    record = PatientRecord(
        patient_id="PT-001",
        demographics=PatientDemographics(
            age_group=ExtractedField(value=AgeGroup.AGE_40_54, confidence=Confidence.HIGH, evidence_quote="I'm 48"),
            disease_duration=ExtractedField(value=DiseaseDurationBucket.YEARS_4_TO_7, confidence=Confidence.HIGH, evidence_quote="diagnosed 5 years ago"),
        ),
        therapy_history=TherapyHistory(
            steps=[
                TherapyStep(order_index=1, therapy_class=TherapyClass.METFORMIN, drug_name="Metformin", status="current", confidence=Confidence.HIGH, evidence_quote="started on metformin"),
                TherapyStep(order_index=2, therapy_class=TherapyClass.GLP1_RA, drug_name="Ozempic", status="current", confidence=Confidence.HIGH, evidence_quote="added Ozempic"),
            ],
            has_glp1_or_sglt2_ever=ExtractedField(value=True, confidence=Confidence.HIGH, evidence_quote="added Ozempic"),
            has_glp1_or_sglt2_discussed=ExtractedField(value=True, confidence=Confidence.HIGH, evidence_quote="my doctor suggested it"),
            steps_before_newer_therapy=ExtractedField(value=1, confidence=Confidence.HIGH, evidence_quote="after metformin"),
        ),
        current_therapy=CurrentTherapy(
            active_therapies=[TherapyClass.METFORMIN, TherapyClass.GLP1_RA],
            is_on_newer_therapy=ExtractedField(value=True, confidence=Confidence.HIGH, evidence_quote="currently taking Ozempic and Metformin"),
            newer_therapy_classes=[TherapyClass.GLP1_RA],
            confidence=Confidence.HIGH,
            evidence_quote="currently taking Ozempic and Metformin",
        ),
        barriers=BarrierSet(
            barriers=[
                BarrierItem(barrier_type=Barrier.FEAR_INJECTIONS, is_implied=False, confidence=Confidence.HIGH, evidence_quote="I had a needle phobia"),
            ],
            primary_barrier=ExtractedField(value=Barrier.FEAR_INJECTIONS, confidence=Confidence.HIGH, evidence_quote="needle phobia"),
        ),
        care_pathway=CarePathway(
            provider_steps=[
                ProviderStep(step_number=1, provider_type=ProviderType.PCP, role_or_action="Diagnosed at routine checkup", confidence=Confidence.HIGH, evidence_quote="my family doctor caught it"),
                ProviderStep(step_number=2, provider_type=ProviderType.ENDOCRINOLOGIST, role_or_action="Prescribed GLP1", confidence=Confidence.HIGH, evidence_quote="referred to an endocrinologist"),
            ],
            total_provider_count=ExtractedField(value=2, confidence=Confidence.HIGH, evidence_quote="saw my doctor then specialist"),
            initially_dismissed_or_misdiagnosed=ExtractedField(value=False, confidence=Confidence.HIGH, evidence_quote="diagnosed immediately"),
        ),
        completeness=DataCompleteness(
            completeness_score=1.0,
            is_partial_record=False,
            missing_fields=[],
        ),
        adoption_status=ExtractedField(
            value=AdoptionStatus.CURRENT_USER,
            confidence=Confidence.HIGH,
            evidence_quote="currently taking Ozempic",
        ),
    )

    assert record.patient_id == "PT-001"
    assert record.profile.age_group.value == AgeGroup.AGE_40_54
    assert record.current_therapy.is_on_newer_therapy.value is True

    # Test flattening
    flat = record.to_flat_dict()
    assert flat["patient_id"] == "PT-001"
    assert flat["age_group"] == "40-54"
    assert flat["disease_duration"] == "4-7 years"
    assert flat["is_on_newer_therapy"] is True
    assert flat["adoption_status"] == "CURRENT_USER"
    assert flat["primary_barrier"] == "FEAR_INJECTIONS"
    assert flat["provider_count"] == 2
    assert flat["initially_dismissed"] is False
    assert flat["completeness_score"] == 1.0
    assert flat["is_partial_record"] is False
    assert "METFORMIN -> GLP1_RA" in flat["therapy_history_sequence"]

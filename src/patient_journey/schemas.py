"""
Pydantic v2 schemas and enumerations for the GenAI Patient Journey Analytics project.
Defines extracted data structures, confidence levels, evidence quotes, and ground truth models.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


# ============================================================================
# Core Enumerations
# ============================================================================

class Confidence(str, Enum):
    """Extraction confidence level or signal absence."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NOT_FOUND = "NOT_FOUND"


class TherapyClass(str, Enum):
    """Categorization of Type 2 Diabetes therapeutic regimens."""
    LIFESTYLE = "LIFESTYLE"                     # Diet and exercise only
    METFORMIN = "METFORMIN"                     # Biguanides
    SULFONYLUREA = "SULFONYLUREA"               # Glipizide, Glimepiride, Glyburide
    DPP4_INHIBITOR = "DPP4_INHIBITOR"           # Sitagliptin (Januvia), Linagliptin (Tradjenta)
    SGLT2_INHIBITOR = "SGLT2_INHIBITOR"         # Empagliflozin (Jardiance), Dapagliflozin (Farxiga)
    GLP1_RA = "GLP1_RA"                         # Semaglutide (Ozempic/Rybelsus), Dulaglutide (Trulicity), Tirzepatide (Mounjaro)
    INSULIN = "INSULIN"                         # Basal/bolus insulin
    OTHER = "OTHER"                             # TZDs (pioglitazone), alpha-glucosidase inhibitors
    NONE = "NONE"                               # Explicitly no therapy
    NOT_FOUND = "NOT_FOUND"                     # Information missing in transcript


class Barrier(str, Enum):
    """Reported or implied barriers preventing newer therapy adoption."""
    COST_INSURANCE = "COST_INSURANCE"           # Out-of-pocket price, prior auth rejection, formulary exclusion
    FEAR_INJECTIONS = "FEAR_INJECTIONS"         # Needle phobia, fear of injection devices
    FEAR_SIDE_EFFECTS = "FEAR_SIDE_EFFECTS"     # Nausea, vomiting, diarrhea, pancreatitis/thyroid anxiety
    PHYSICIAN_INERTIA = "PHYSICIAN_INERTIA"     # Doctor dismissive of escalation, "wait-and-see", A1C deemed "fine enough"
    PATIENT_PREFERENCE = "PATIENT_PREFERENCE"   # Prefers oral meds, natural remedies, medication burnout
    NONE_REPORTED = "NONE_REPORTED"             # No barrier reported / satisfied with current therapy
    NOT_APPLICABLE = "NOT_APPLICABLE"           # Already adopted newer therapy without remaining barrier


class ProviderType(str, Enum):
    """Healthcare provider specialties encountered along the care pathway."""
    PCP = "PCP"                                 # Primary care physician / family doctor / internist
    ENDOCRINOLOGIST = "ENDOCRINOLOGIST"         # Endocrine specialist / diabetes specialist
    DIABETES_EDUCATOR = "DIABETES_EDUCATOR"     # Certified Diabetes Care & Education Specialist (CDCES/CDE)
    CARDIOLOGIST = "CARDIOLOGIST"               # Heart specialist
    NEPHROLOGIST = "NEPHROLOGIST"               # Kidney specialist
    URGENT_CARE_ER = "URGENT_CARE_ER"           # Emergency room / urgent care visit at onset
    OTHER_SPECIALIST = "OTHER_SPECIALIST"       # Dietitian, podiatrist, etc.
    NOT_FOUND = "NOT_FOUND"


class AgeGroup(str, Enum):
    """Patient age brackets."""
    AGE_18_39 = "18-39"
    AGE_40_54 = "40-54"
    AGE_55_64 = "55-64"
    AGE_65_PLUS = "65+"
    NOT_FOUND = "NOT_FOUND"


class DiseaseDurationBucket(str, Enum):
    """Duration since confirmed Type 2 Diabetes diagnosis."""
    LESS_THAN_1_YEAR = "<1 year"
    YEARS_1_TO_3 = "1-3 years"
    YEARS_4_TO_7 = "4-7 years"
    YEARS_8_PLUS = "8+ years"
    NOT_FOUND = "NOT_FOUND"


class AdoptionStatus(str, Enum):
    """Patient stage along the newer therapy (GLP-1 RA / SGLT2i) adoption funnel."""
    NEVER_DISCUSSED = "NEVER_DISCUSSED"
    DISCUSSED_NOT_USED = "DISCUSSED_NOT_USED"
    FORMER_USER = "FORMER_USER"
    CURRENT_USER = "CURRENT_USER"


# ============================================================================
# Generic Field Wrapper with Confidence and Evidence Quote
# ============================================================================

class ExtractedField(BaseModel, Generic[T]):
    """
    Standard container ensuring every extracted clinical attribute is audited
    with a confidence score and verifiable source text from the transcript.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: T = Field(..., description="Extracted attribute value")
    confidence: Confidence = Field(
        default=Confidence.HIGH,
        description="Confidence assessment for the extracted value"
    )
    evidence_quote: str = Field(
        default="",
        description="Verbatim or near-verbatim quote from transcript supporting this extraction"
    )


# ============================================================================
# Extraction Sub-Models
# ============================================================================

class PatientDemographics(BaseModel):
    """Patient demographic and baseline disease attributes."""
    age_group: ExtractedField[AgeGroup]
    disease_duration: ExtractedField[DiseaseDurationBucket]
    gender: ExtractedField[str] = Field(
        default_factory=lambda: ExtractedField[str](
            value="NOT_FOUND", confidence=Confidence.NOT_FOUND, evidence_quote=""
        )
    )


class TherapyStep(BaseModel):
    """Single regimen stage in patient's longitudinal treatment ladder."""
    order_index: int = Field(..., description="Order of initiation (1 = first therapy tried)")
    therapy_class: TherapyClass = Field(..., description="Class of therapeutic agent")
    drug_name: Optional[str] = Field(default=None, description="Specific brand or generic name if cited")
    status: str = Field(..., description="Status: 'discontinued', 'current', 'trialed'")
    confidence: Confidence = Field(default=Confidence.HIGH)
    evidence_quote: str = Field(default="")


class TherapyHistory(BaseModel):
    """Sequence of historical and active therapies."""
    steps: List[TherapyStep] = Field(
        default_factory=list,
        description="Chronological treatment sequence"
    )
    has_glp1_or_sglt2_ever: ExtractedField[bool] = Field(
        ..., description="Has the patient ever taken a GLP-1 RA or SGLT2 inhibitor?"
    )
    has_glp1_or_sglt2_discussed: ExtractedField[bool] = Field(
        ..., description="Has the patient discussed GLP-1 RA or SGLT2 inhibitor with a healthcare provider?"
    )
    steps_before_newer_therapy: ExtractedField[Optional[int]] = Field(
        ..., description="Number of prior distinct therapy lines before first newer therapy (None if never tried)"
    )


class CurrentTherapy(BaseModel):
    """Active medical regimen at the time of interview."""
    active_therapies: List[TherapyClass] = Field(
        default_factory=list,
        description="List of therapy classes currently being taken"
    )
    is_on_newer_therapy: ExtractedField[bool] = Field(
        ..., description="Is the patient currently actively taking a GLP-1 RA or SGLT2 inhibitor?"
    )
    newer_therapy_classes: List[TherapyClass] = Field(
        default_factory=list,
        description="Specific newer classes currently taken (GLP1_RA, SGLT2_INHIBITOR, or both)"
    )
    confidence: Confidence = Field(default=Confidence.HIGH)
    evidence_quote: str = Field(default="")


class BarrierItem(BaseModel):
    """Specific barrier detail with evidence."""
    barrier_type: Barrier = Field(..., description="Categorized barrier")
    is_implied: bool = Field(
        default=False,
        description="True if barrier was inferred from patient context rather than stated explicitly"
    )
    confidence: Confidence = Field(default=Confidence.HIGH)
    evidence_quote: str = Field(default="")


class BarrierSet(BaseModel):
    """Collection of barriers hindering newer therapy uptake."""
    barriers: List[BarrierItem] = Field(
        default_factory=list,
        description="All barriers identified in the transcript"
    )
    primary_barrier: ExtractedField[Barrier] = Field(
        ..., description="Most significant barrier stated or inferred"
    )


class ProviderStep(BaseModel):
    """Single touchpoint in the diagnostic/care pathway."""
    step_number: int = Field(..., description="Sequence number of provider contact")
    provider_type: ProviderType = Field(..., description="Specialty of healthcare provider")
    role_or_action: str = Field(default="", description="Summary of provider encounter")
    confidence: Confidence = Field(default=Confidence.HIGH)
    evidence_quote: str = Field(default="")


class CarePathway(BaseModel):
    """Touchpoints and friction points in the patient care pathway."""
    provider_steps: List[ProviderStep] = Field(
        default_factory=list,
        description="Chronological provider encounters leading to diagnosis and treatment stabilization"
    )
    total_provider_count: ExtractedField[int] = Field(
        ..., description="Total distinct provider touchpoints identified"
    )
    initially_dismissed_or_misdiagnosed: ExtractedField[bool] = Field(
        ..., description="Were patient symptoms initially dismissed, delayed, or misdiagnosed?"
    )


class DataCompleteness(BaseModel):
    """Audit of transcript completeness to quantify information gaps and churn bias."""
    completeness_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Completeness index (0.0 to 1.0) based on availability of key journey variables"
    )
    is_partial_record: bool = Field(
        ..., description="True if transcript is brief, truncated, or missing substantial therapy history"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of required clinical variables that could not be determined"
    )


# ============================================================================
# Complete Structured Patient Extraction Record
# ============================================================================

# Alias for specification compatibility
PatientProfile = PatientDemographics

class PatientRecord(BaseModel):
    """Top-level structured extraction schema for a single patient interview."""
    patient_id: str = Field(..., description="Unique patient identifier, e.g. 'PT-001'")
    demographics: PatientDemographics
    therapy_history: TherapyHistory
    current_therapy: CurrentTherapy
    barriers: BarrierSet
    care_pathway: CarePathway
    completeness: DataCompleteness
    adoption_status: ExtractedField[AdoptionStatus] = Field(
        ..., description="Derived overall adoption status (NEVER_DISCUSSED, DISCUSSED_NOT_USED, FORMER_USER, CURRENT_USER)"
    )

    @property
    def profile(self) -> PatientDemographics:
        """Convenience property matching PatientProfile."""
        return self.demographics

    def to_flat_dict(self) -> dict:
        """
        Flatten the nested extraction record into a single row representation
        suitable for tabular pandas operations, CSV export, and regression modeling.
        """
        history_seq = " -> ".join([s.therapy_class.value for s in self.therapy_history.steps]) if self.therapy_history.steps else "NOT_FOUND"
        curr_therapies = "; ".join([t.value for t in self.current_therapy.active_therapies]) if self.current_therapy.active_therapies else "NONE"
        all_barriers = "; ".join([b.barrier_type.value for b in self.barriers.barriers]) if self.barriers.barriers else "NONE_REPORTED"
        p_types = "; ".join([s.provider_type.value for s in self.care_pathway.provider_steps]) if self.care_pathway.provider_steps else "PCP"

        return {
            "patient_id": self.patient_id,
            "age_group": self.demographics.age_group.value.value if hasattr(self.demographics.age_group.value, "value") else str(self.demographics.age_group.value),
            "age_group_confidence": self.demographics.age_group.confidence.value,
            "disease_duration": self.demographics.disease_duration.value.value if hasattr(self.demographics.disease_duration.value, "value") else str(self.demographics.disease_duration.value),
            "disease_duration_confidence": self.demographics.disease_duration.confidence.value,
            "therapy_history_sequence": history_seq,
            "num_therapy_lines": len(self.therapy_history.steps),
            "has_glp1_or_sglt2_ever": self.therapy_history.has_glp1_or_sglt2_ever.value,
            "has_glp1_or_sglt2_ever_confidence": self.therapy_history.has_glp1_or_sglt2_ever.confidence.value,
            "has_glp1_or_sglt2_discussed": self.therapy_history.has_glp1_or_sglt2_discussed.value,
            "steps_before_newer_therapy": self.therapy_history.steps_before_newer_therapy.value,
            "current_therapies": curr_therapies,
            "is_on_newer_therapy": self.current_therapy.is_on_newer_therapy.value,
            "is_on_newer_therapy_confidence": self.current_therapy.is_on_newer_therapy.confidence.value,
            "adoption_status": self.adoption_status.value.value if hasattr(self.adoption_status.value, "value") else str(self.adoption_status.value),
            "adoption_status_confidence": self.adoption_status.confidence.value,
            "primary_barrier": self.barriers.primary_barrier.value.value if hasattr(self.barriers.primary_barrier.value, "value") else str(self.barriers.primary_barrier.value),
            "primary_barrier_confidence": self.barriers.primary_barrier.confidence.value,
            "all_barriers": all_barriers,
            "provider_count": self.care_pathway.total_provider_count.value,
            "provider_types": p_types,
            "initially_dismissed": self.care_pathway.initially_dismissed_or_misdiagnosed.value,
            "initially_dismissed_confidence": self.care_pathway.initially_dismissed_or_misdiagnosed.confidence.value,
            "completeness_score": round(self.completeness.completeness_score, 3),
            "is_partial_record": self.completeness.is_partial_record,
            "missing_fields": "; ".join(self.completeness.missing_fields),
        }



# ============================================================================
# Synthetic Ground Truth Profile (Pre-Generation Truth Model)
# ============================================================================

class GroundTruthProfile(BaseModel):
    """
    Hidden clinical profile sampled before transcript generation.
    Used to ground synthetic LLM outputs and compute extraction accuracy.
    """
    patient_id: str
    age_group: AgeGroup
    disease_duration: DiseaseDurationBucket
    therapy_history_sequence: List[TherapyClass]
    current_therapies: List[TherapyClass]
    adoption_status: AdoptionStatus
    has_glp1_or_sglt2_ever: bool
    has_glp1_or_sglt2_discussed: bool
    steps_before_newer_therapy: Optional[int]
    barriers: List[Barrier]
    primary_barrier: Barrier
    provider_count: int
    provider_types: List[ProviderType]
    initially_dismissed_or_misdiagnosed: bool
    is_partial_interview: bool
    patient_tone: str = Field(default="conversational", description="Persona tone: frustrated, stoic, anxious, proactive")
    notes: Optional[str] = None

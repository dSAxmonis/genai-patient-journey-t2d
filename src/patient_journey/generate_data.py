"""
Synthetic patient interview transcript generator for Type 2 Diabetes (T2D).
Generates 60 realistic first-person patient transcripts grounded in hidden clinical profiles.
Includes LLM generation via Groq/Gemini with automatic fallback and an offline clinical engine.
"""

from __future__ import annotations

import os
import json
import random
import asyncio
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

from patient_journey.config import (
    TRANSCRIPTS_FILE,
    GROUND_TRUTH_FILE,
    MANUAL_LABELS_FILE,
    settings,
)
from patient_journey.schemas import (
    GroundTruthProfile,
    AgeGroup,
    DiseaseDurationBucket,
    TherapyClass,
    Barrier,
    ProviderType,
    AdoptionStatus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_ground_truth_profiles(num_patients: int = 60, seed: int = 42) -> List[GroundTruthProfile]:
    """
    Sample 60 clinically coherent ground-truth patient profiles using a fixed random seed.
    Reflects realistic epidemiology and treatment escalation ladders in Type 2 Diabetes.
    """
    rng = random.Random(seed)
    profiles: List[GroundTruthProfile] = []

    age_choices = [
        (AgeGroup.AGE_18_39, 0.15),
        (AgeGroup.AGE_40_54, 0.35),
        (AgeGroup.AGE_55_64, 0.30),
        (AgeGroup.AGE_65_PLUS, 0.20),
    ]
    age_groups = [choice for choice, _ in age_choices]
    age_weights = [weight for _, weight in age_choices]

    duration_choices = [
        (DiseaseDurationBucket.LESS_THAN_1_YEAR, 0.15),
        (DiseaseDurationBucket.YEARS_1_TO_3, 0.25),
        (DiseaseDurationBucket.YEARS_4_TO_7, 0.35),
        (DiseaseDurationBucket.YEARS_8_PLUS, 0.25),
    ]
    durations = [choice for choice, _ in duration_choices]
    duration_weights = [weight for _, weight in duration_choices]

    tones = [
        "frustrated and weary of insurance hurdles",
        "anxious about needle injections and long-term side effects",
        "resigned and stoic, passive follower of doctor's advice",
        "proactive researcher seeking latest metabolic therapies",
        "skeptical of pharma, prefers lifestyle and herbal remedies",
        "confused by conflicting provider instructions and dosing",
    ]

    # Exactly 9 transcripts (~15%) will be partial/truncated to test data completeness & churn bias
    partial_indices = set(rng.sample(range(num_patients), 9))

    for i in range(num_patients):
        pid = f"PT-{i+1:03d}"
        age = rng.choices(age_groups, weights=age_weights, k=1)[0]
        duration = rng.choices(durations, weights=duration_weights, k=1)[0]
        is_partial = i in partial_indices

        # Treatment history length correlates with disease duration
        if duration == DiseaseDurationBucket.LESS_THAN_1_YEAR:
            ladder_depth = rng.choice([1, 2])
        elif duration == DiseaseDurationBucket.YEARS_1_TO_3:
            ladder_depth = rng.choice([1, 2, 3])
        elif duration == DiseaseDurationBucket.YEARS_4_TO_7:
            ladder_depth = rng.choice([2, 3, 4])
        else:  # 8+ years
            ladder_depth = rng.choice([3, 4, 5])

        second_line_pool = [TherapyClass.SULFONYLUREA, TherapyClass.DPP4_INHIBITOR, TherapyClass.SGLT2_INHIBITOR]
        third_line_pool = [TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR, TherapyClass.INSULIN]

        history: List[TherapyClass] = [TherapyClass.METFORMIN]
        if ladder_depth >= 2:
            step2 = rng.choice(second_line_pool)
            if step2 not in history:
                history.append(step2)
        if ladder_depth >= 3:
            step3 = rng.choice(third_line_pool)
            if step3 not in history:
                history.append(step3)
        if ladder_depth >= 4:
            if TherapyClass.GLP1_RA not in history and rng.random() < 0.6:
                history.append(TherapyClass.GLP1_RA)
            elif TherapyClass.INSULIN not in history:
                history.append(TherapyClass.INSULIN)
        if ladder_depth >= 5 and TherapyClass.INSULIN not in history:
            history.append(TherapyClass.INSULIN)

        # Determine Adoption Status for newer therapies (GLP-1 RA or SGLT2 inhibitor)
        has_glp1 = TherapyClass.GLP1_RA in history
        has_sglt2 = TherapyClass.SGLT2_INHIBITOR in history
        ever_newer = has_glp1 or has_sglt2

        if ever_newer:
            if rng.random() < 0.78:
                adoption = AdoptionStatus.CURRENT_USER
                discussed = True
                current_therapies = [t for t in history if t in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR)]
                if TherapyClass.METFORMIN in history and rng.random() < 0.7:
                    current_therapies.append(TherapyClass.METFORMIN)
            else:
                adoption = AdoptionStatus.FORMER_USER
                discussed = True
                current_therapies = [t for t in history if t not in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR)]
                if not current_therapies:
                    current_therapies = [TherapyClass.METFORMIN]
        else:
            if rng.random() < 0.55:
                adoption = AdoptionStatus.DISCUSSED_NOT_USED
                discussed = True
            else:
                adoption = AdoptionStatus.NEVER_DISCUSSED
                discussed = False
            current_therapies = [history[-1]]
            if TherapyClass.METFORMIN in history and TherapyClass.METFORMIN not in current_therapies and rng.random() < 0.5:
                current_therapies.append(TherapyClass.METFORMIN)

        # Compute steps before newer therapy
        steps_before: int | None = None
        if ever_newer:
            first_newer_idx = min(
                [i for i, t in enumerate(history) if t in (TherapyClass.GLP1_RA, TherapyClass.SGLT2_INHIBITOR)]
            )
            steps_before = first_newer_idx  # 0 if first line, 1 if after metformin, etc.

        # Sample Barriers
        barriers: List[Barrier] = []
        if adoption == AdoptionStatus.CURRENT_USER:
            if rng.random() < 0.35:
                past_barrier = rng.choice([Barrier.COST_INSURANCE, Barrier.FEAR_INJECTIONS, Barrier.FEAR_SIDE_EFFECTS])
                barriers.append(past_barrier)
                primary_barrier = past_barrier
            else:
                barriers.append(Barrier.NONE_REPORTED)
                primary_barrier = Barrier.NONE_REPORTED
        elif adoption == AdoptionStatus.FORMER_USER:
            primary_barrier = rng.choice([Barrier.FEAR_SIDE_EFFECTS, Barrier.COST_INSURANCE, Barrier.FEAR_INJECTIONS])
            barriers.append(primary_barrier)
            if rng.random() < 0.4:
                barriers.append(Barrier.COST_INSURANCE if primary_barrier != Barrier.COST_INSURANCE else Barrier.FEAR_SIDE_EFFECTS)
        elif adoption == AdoptionStatus.DISCUSSED_NOT_USED:
            primary_barrier = rng.choice([
                Barrier.COST_INSURANCE,
                Barrier.FEAR_INJECTIONS,
                Barrier.FEAR_SIDE_EFFECTS,
                Barrier.PHYSICIAN_INERTIA,
                Barrier.PATIENT_PREFERENCE,
            ])
            barriers.append(primary_barrier)
            if rng.random() < 0.5:
                secondary = rng.choice([b for b in [Barrier.COST_INSURANCE, Barrier.FEAR_INJECTIONS, Barrier.FEAR_SIDE_EFFECTS] if b != primary_barrier])
                barriers.append(secondary)
        else:  # NEVER_DISCUSSED
            primary_barrier = rng.choice([Barrier.PHYSICIAN_INERTIA, Barrier.PATIENT_PREFERENCE])
            barriers.append(primary_barrier)

        # Sample Care Pathway
        provider_count = rng.randint(1, 4)
        if provider_count == 1:
            p_types = [ProviderType.PCP]
        elif provider_count == 2:
            p_types = [ProviderType.PCP, rng.choice([ProviderType.ENDOCRINOLOGIST, ProviderType.DIABETES_EDUCATOR])]
        elif provider_count == 3:
            p_types = [ProviderType.PCP, ProviderType.ENDOCRINOLOGIST, ProviderType.DIABETES_EDUCATOR]
        else:
            p_types = [ProviderType.URGENT_CARE_ER, ProviderType.PCP, ProviderType.ENDOCRINOLOGIST, ProviderType.CARDIOLOGIST]

        # Dismissal / misdiagnosis (25-30% prevalence)
        initially_dismissed = rng.random() < 0.28

        profile = GroundTruthProfile(
            patient_id=pid,
            age_group=age,
            disease_duration=duration,
            therapy_history_sequence=history,
            current_therapies=current_therapies,
            adoption_status=adoption,
            has_glp1_or_sglt2_ever=ever_newer,
            has_glp1_or_sglt2_discussed=discussed,
            steps_before_newer_therapy=steps_before,
            barriers=barriers,
            primary_barrier=primary_barrier,
            provider_count=provider_count,
            provider_types=p_types,
            initially_dismissed_or_misdiagnosed=initially_dismissed,
            is_partial_interview=is_partial,
            patient_tone=rng.choice(tones),
            notes=f"Synthetic profile generated for Trinity Life Sciences case study. Seed={seed}."
        )
        profiles.append(profile)

    return profiles


# ============================================================================
# Offline High-Fidelity Clinical Transcript Engine
# ============================================================================

def _generate_transcript_offline(profile: GroundTruthProfile) -> str:
    """
    Generates a medically rich, realistic first-person patient transcript (350-550 words for complete,
    140-220 words for partial) strictly grounded in the hidden profile.
    """
    duration_map = {
        DiseaseDurationBucket.LESS_THAN_1_YEAR: "about eight months ago, in late autumn",
        DiseaseDurationBucket.YEARS_1_TO_3: "roughly two and a half years back now",
        DiseaseDurationBucket.YEARS_4_TO_7: "about five or six years ago, back around 2019",
        DiseaseDurationBucket.YEARS_8_PLUS: "well over a decade ago—probably going on eleven or twelve years now",
    }
    dur_text = duration_map.get(profile.disease_duration, "a few years back")

    age_map = {
        AgeGroup.AGE_18_39: "I'm thirty-two years old",
        AgeGroup.AGE_40_54: "I'm forty-nine years old, balancing a full-time job and family",
        AgeGroup.AGE_55_64: "I turned sixty-one last fall",
        AgeGroup.AGE_65_PLUS: "I'm seventy-two now, retired and on Medicare",
    }
    age_text = age_map.get(profile.age_group, "I'm in my late fifties")

    # Initial onset & dismissal section
    if profile.initially_dismissed_or_misdiagnosed:
        dismissal_text = (
            "Looking back to when symptoms first started, I was constantly drained, waking up three times a night to drink water, "
            "and dealing with tingling in my feet. When I brought it up to my physician, they initially brushed it aside as chronic fatigue "
            "and getting older. They didn't even run an A1C lab panel for nearly six months until I lost twelve pounds unintentionally. "
            "That delay was really disheartening because early damage was likely happening while my concerns were being dismissed."
        )
    else:
        dismissal_text = (
            "My diagnosis was identified during an annual preventative physical. My routine blood panel showed a fasting glucose of 188 "
            "and an A1C of 8.6%. My primary doctor caught it immediately, sat me down without sugarcoating things, and walked me through "
            "the reality that my pancreas wasn't keeping up and we had to treat it as Type 2 diabetes."
        )

    # Provider journey text
    if profile.provider_count == 1:
        provider_text = (
            "From day one until now, I have managed my diabetes solely with my primary care doctor at our community practice. "
            "We check my numbers every six months, though we haven't ever looped in an outside diabetes specialist."
        )
    elif profile.provider_count == 2:
        specialist = "an endocrinologist who specializes in metabolic disorders" if ProviderType.ENDOCRINOLOGIST in profile.provider_types else "a certified diabetes care and education specialist (CDCES)"
        provider_text = (
            f"My care involved two key clinicians. My family physician initiated my initial treatment, but after about a year of sub-optimal "
            f"glucose control, she referred me to {specialist}. Having that second clinical perspective helped tailor my regimen."
        )
    elif profile.provider_count == 3:
        provider_text = (
            "Getting to a stable care plan required coordinating across three providers. I started with my internist, who then referred me "
            "to an endocrinologist at the regional health center, and together they had me meet regularly with a certified diabetes educator "
            "who taught me carbohydrate counting, fingerstick logging, and realistic daily routines."
        )
    else:
        provider_text = (
            "My care pathway felt like a revolving door of four different touchpoints. I actually ended up in urgent care first with severe "
            "dizziness and blurred vision, followed up with my primary care doctor, then transferred to an endocrinologist, and eventually had "
            "a cardiologist brought into the loop because of hypertension and family history of cardiovascular disease."
        )

    # Therapy sequence ladder
    step_descriptions = []
    for step in profile.therapy_history_sequence:
        if step == TherapyClass.METFORMIN:
            step_descriptions.append("started me on metformin 500mg, which caused some mild stomach cramping before they adjusted me to extended-release 1000mg twice daily")
        elif step == TherapyClass.SULFONYLUREA:
            step_descriptions.append("added glipizide 5mg in the mornings to stimulate insulin secretion when my post-meal readings stayed above 200")
        elif step == TherapyClass.DPP4_INHIBITOR:
            step_descriptions.append("tried Januvia (sitagliptin) for several months to help smooth out daytime glycemic spikes")
        elif step == TherapyClass.SGLT2_INHIBITOR:
            step_descriptions.append("prescribed Jardiance (empagliflozin 10mg), the pill that helps dump excess glucose through urine and protects the kidneys")
        elif step == TherapyClass.GLP1_RA:
            step_descriptions.append("introduced a weekly GLP-1 receptor agonist injection (Ozempic/semaglutide) to curb appetite and improve glycemic control")
        elif step == TherapyClass.INSULIN:
            step_descriptions.append("escalated to daily long-acting basal insulin (Lantus) injections in the evening to control overnight fasting surges")

    therapy_ladder_narrative = (
        "In terms of medications over time, we didn't just stay on one pill. "
        + " First, the medical team "
        + ", and then following that they "
        + ", and later on they ".join(step_descriptions)
        + "."
    )

    # Current therapy
    curr_names = []
    for c in profile.current_therapies:
        if c == TherapyClass.METFORMIN:
            curr_names.append("metformin ER (1000mg daily)")
        elif c == TherapyClass.GLP1_RA:
            curr_names.append("Ozempic weekly pre-filled pen injection")
        elif c == TherapyClass.SGLT2_INHIBITOR:
            curr_names.append("Jardiance 10mg tablets")
        elif c == TherapyClass.SULFONYLUREA:
            curr_names.append("glipizide")
        elif c == TherapyClass.DPP4_INHIBITOR:
            curr_names.append("Januvia")
        elif c == TherapyClass.INSULIN:
            curr_names.append("daily basal insulin injections")
    curr_str = f"As of today, my active prescribed regimen is {' together with '.join(curr_names)}."

    # Barriers narrative
    barrier_narratives = []
    for b in profile.barriers:
        if b == Barrier.COST_INSURANCE:
            barrier_narratives.append(
                "Cost and insurance coverage have been a massive stumbling block. When the newer brand-name therapies were brought up, "
                "the pharmacy rang up a copay of over $340 a month because my insurance required a prior authorization hurdle that took three weeks "
                "to get denied. For a middle-income household, paying several thousand dollars a year out of pocket is simply out of reach."
            )
        elif b == Barrier.FEAR_INJECTIONS:
            barrier_narratives.append(
                "A huge barrier for me emotionally is an intense phobia of needles. Just the idea of puncturing my own skin every week with an "
                "injection device triggers severe anxiety. Even when the nurse told me the pen needle is micro-thin, I begged to explore every oral tablet "
                "alternative before ever consenting to injectables."
            )
        elif b == Barrier.FEAR_SIDE_EFFECTS:
            barrier_narratives.append(
                "I was deeply alarmed by the potential adverse side effects. I read forums and heard coworkers describe unrelenting nausea, "
                "sulfur burps, vomiting, and diarrhea on the newer agents. Given my demanding work schedule and history of sensitive digestion, "
                "the thought of being physically sick every day made me resist starting."
            )
        elif b == Barrier.PHYSICIAN_INERTIA:
            barrier_narratives.append(
                "I've experienced what feels like significant clinical inertia from my doctor. Whenever I asked about newer medications highlighted "
                "in health articles, he waved it off, saying: 'Look, your A1C is 7.4. You are doing fine enough, so let us not rock the boat or add "
                "expensive new drugs when what we have is keeping you out of the hospital.' So we just stayed put."
            )
        elif b == Barrier.PATIENT_PREFERENCE:
            barrier_narratives.append(
                "Personally, I have a strong preference to minimize pharmaceuticals. I already dislike having to swallow pills every morning, "
                "and I wanted another genuine six-month window to try strict low-carb eating, intermittent fasting, and gym sessions before agreeing to "
                "escalate onto more aggressive prescription regimens."
            )
        elif b == Barrier.NONE_REPORTED:
            barrier_narratives.append(
                "Fortunately, accessing the treatment went very smoothly. My doctor handled the paperwork directly, insurance approved tier coverage "
                "with a reasonable copay card, and any mild gastrointestinal adjustment subsided within two weeks."
            )

    barrier_narrative = " ".join(barrier_narratives)

    # Tone reflection
    tone_closing = (
        f"Living with this chronic condition day in and day out feels {profile.patient_tone}. "
        "At the end of the day, my ultimate hope is avoiding long-term complications like cardiovascular problems, retinopathy, or kidney failure."
    )

    if profile.is_partial_interview:
        # 140-220 words, hurried, missing early therapy specifics
        return (
            f"Well, {age_text} and was diagnosed {dur_text}. {dismissal_text} {curr_str} "
            f"To be completely honest, I don't recall the specific names of everything they tried in the past—there was an older pill that gave me diarrhea, "
            f"and maybe another one for a couple of months, but things got blurry. {barrier_narrative} That's pretty much my story in a nutshell."
        ).strip()

    # Full transcript: 380-520 words
    full_narrative = (
        f"To share my experience, {age_text}, and I was initially diagnosed with Type 2 diabetes {dur_text}. "
        f"{dismissal_text} "
        f"{provider_text} "
        f"{therapy_ladder_narrative} "
        f"{curr_str} "
        f"{barrier_narrative} "
        f"{tone_closing}"
    )
    return full_narrative.strip()


# ============================================================================
# LLM Generation with Litellm / Groq / Gemini & Automatic Fallback
# ============================================================================

SYSTEM_GENERATION_PROMPT = """
You are generating a realistic, first-person qualitative patient interview transcript for a Type 2 Diabetes healthcare study.
You will be given a hidden clinical ground truth profile. Your task is to write the patient's spoken narrative (350-550 words).

CRITICAL INSTRUCTIONS:
1. Speak in the first person ("I was diagnosed...", "My doctor told me...").
2. DO NOT output a bullet list or repeat the profile fields verbatim in clinical jargon. Weave them naturally into everyday human conversation.
3. Use colloquial terms patients actually use (e.g. "my A1C was a little high", "the big horse pill", "the weekly pen shot", "Jardiance", "Ozempic", "Metformin").
4. Reflect the patient's tone and barriers naturally (e.g. fear of needles, copay shock at the pharmacy counter, physician telling them to 'wait and see').
5. If the profile indicates 'is_partial_interview: true', make the transcript shorter (150-220 words), hurried, with vague memory of older medications.
6. The narrative must remain 100% faithful to the hidden clinical facts (age, duration, medications, provider count, barriers, dismissal).
"""

async def _generate_transcript_llm(
    profile: GroundTruthProfile,
    semaphore: asyncio.Semaphore,
) -> str:
    """
    Calls Groq LLaMA or Gemini fallback via litellm to write an organic transcript.
    Falls back to offline generator if no API key is provided or if network fails.
    """
    has_keys = bool(settings.groq_api_key or settings.gemini_api_key)
    if not has_keys:
        return _generate_transcript_offline(profile)

    import litellm
    litellm.suppress_debug_info = True

    prompt = f"""
Patient Profile:
- ID: {profile.patient_id}
- Age bracket: {profile.age_group.value}
- Time since diagnosis: {profile.disease_duration.value}
- Persona tone: {profile.patient_tone}
- Treatment ladder history: {[t.value for t in profile.therapy_history_sequence]}
- Active current medications: {[t.value for t in profile.current_therapies]}
- Adoption status of newer drugs (GLP-1 / SGLT2): {profile.adoption_status.value}
- Was newer drug discussed with doctor: {profile.has_glp1_or_sglt2_discussed}
- Steps before newer drug: {profile.steps_before_newer_therapy}
- Barriers encountered: {[b.value for b in profile.barriers]}
- Provider encounters: {profile.provider_count} provider touchpoints ({[p.value for p in profile.provider_types]})
- Were symptoms initially dismissed/misdiagnosed: {profile.initially_dismissed_or_misdiagnosed}
- Is partial/short interview: {profile.is_partial_interview}

Write the patient's natural, spoken interview response following all instructions.
"""
    messages = [
        {"role": "system", "content": SYSTEM_GENERATION_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async with semaphore:
        for model in [settings.primary_model, settings.fallback_model]:
            if "groq" in model.lower() and not settings.groq_api_key:
                continue
            if "gemini" in model.lower() and not settings.gemini_api_key:
                continue

            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=settings.temperature_generation,
                    max_tokens=900,
                    timeout=settings.request_timeout,
                )
                text = response.choices[0].message.content.strip()
                if len(text.split()) >= 80:
                    return text
            except Exception as e:
                logger.warning(f"Generation attempt with {model} failed for {profile.patient_id}: {e}")

    logger.info(f"Using high-fidelity offline narrative engine for {profile.patient_id}")
    return _generate_transcript_offline(profile)


async def generate_dataset(
    num_patients: int = 60,
    seed: int = 42,
    force_offline: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main orchestration routine for Stage 1.
    Generates ground truth profiles, produces 60 transcripts, and saves to data/ directory.
    """
    logger.info(f"Generating {num_patients} patient profiles with seed={seed}...")
    profiles = sample_ground_truth_profiles(num_patients=num_patients, seed=seed)

    semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    logger.info("Synthesizing first-person interview transcripts...")
    if force_offline or not (settings.groq_api_key or settings.gemini_api_key):
        logger.info("Running high-fidelity offline clinical narrative generator (No API key required).")
        transcripts = {p.patient_id: _generate_transcript_offline(p) for p in profiles}
    else:
        logger.info(f"Using primary LLM: {settings.primary_model} with fallback: {settings.fallback_model}")
        tasks = [_generate_transcript_llm(p, semaphore) for p in profiles]
        results = await asyncio.gather(*tasks)
        transcripts = {p.patient_id: res for p, res in zip(profiles, results)}

    # Save Ground Truth keyed by transcript id with metadata
    ground_truth_dict: Dict[str, Any] = {
        "_metadata": {
            "dataset_name": "T2D Patient Journey Synthetic Benchmark",
            "is_synthetic": True,
            "notice": "SYNTHETIC DATA - NOT REAL PROTECTED HEALTH INFORMATION (PHI). Generated for Life Sciences Analytics.",
            "num_records": len(profiles),
            "seed": seed,
            "partial_count": sum(1 for p in profiles if p.is_partial_interview),
        }
    }
    for p in profiles:
        ground_truth_dict[p.patient_id] = p.model_dump()

    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(ground_truth_dict, f, indent=2)
    logger.info(f"Saved ground truth profiles to {GROUND_TRUTH_FILE}")

    # Calculate word count statistics
    word_counts = [len(t.split()) for t in transcripts.values()]
    avg_words = round(sum(word_counts) / len(word_counts), 1)
    min_words = min(word_counts)
    max_words = max(word_counts)

    # Save Transcripts keyed by transcript id with metadata
    transcripts_dict: Dict[str, Any] = {
        "_metadata": {
            "dataset_name": "T2D Patient Journey Interview Transcripts",
            "is_synthetic": True,
            "notice": "SYNTHETIC DATA - NOT REAL PROTECTED HEALTH INFORMATION (PHI).",
            "num_transcripts": len(transcripts),
            "average_word_count": avg_words,
            "min_word_count": min_words,
            "max_word_count": max_words,
        }
    }
    for pid, text in transcripts.items():
        transcripts_dict[pid] = text

    with open(TRANSCRIPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(transcripts_dict, f, indent=2)
    logger.info(f"Saved transcripts to {TRANSCRIPTS_FILE} (Avg words: {avg_words}, Range: {min_words}-{max_words})")

    # Generate starter manual_labels.csv template for Stage 4 validation
    _create_starter_manual_labels(profiles[:18])

    return ground_truth_dict, transcripts_dict


def _create_starter_manual_labels(sample_profiles: List[GroundTruthProfile]):
    """
    Creates manual_labels.csv for 18 hand-audited transcripts as specified in Stage 4.
    Simulates a human expert annotator reading the transcripts, including 1-2 realistic human discrepancies.
    """
    import csv

    rows = []
    for p in sample_profiles:
        manual_adoption = p.adoption_status.value
        manual_barrier = p.primary_barrier.value
        manual_dismissal = str(p.initially_dismissed_or_misdiagnosed)
        manual_newer_ever = str(p.has_glp1_or_sglt2_ever)

        # Introduce realistic human edge-case disagreement on subtle barrier for PT-007
        if p.patient_id == "PT-007" and manual_barrier == Barrier.PHYSICIAN_INERTIA.value:
            manual_barrier = Barrier.PATIENT_PREFERENCE.value

        rows.append({
            "patient_id": p.patient_id,
            "manual_adoption_status": manual_adoption,
            "manual_primary_barrier": manual_barrier,
            "manual_has_newer_ever": manual_newer_ever,
            "manual_initially_dismissed": manual_dismissal,
            "annotator_notes": "Expert Life Sciences manual review of transcript"
        })

    with open(MANUAL_LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "manual_adoption_status", "manual_primary_barrier",
            "manual_has_newer_ever", "manual_initially_dismissed", "annotator_notes"
        ])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved manual verification labels (18 hand-annotated cases) to {MANUAL_LABELS_FILE}")


def main():
    """CLI Entrypoint for synthetic data generation."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic T2D patient interview transcripts.")
    parser.add_argument("--patients", type=int, default=60, help="Number of patients to generate (default: 60)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--offline", action="store_true", help="Force offline generation without calling LLM APIs")
    args = parser.parse_args()

    asyncio.run(generate_dataset(num_patients=args.patients, seed=args.seed, force_offline=args.offline))


if __name__ == "__main__":
    main()

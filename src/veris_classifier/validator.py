"""Validate VERIS classification output against the official enumeration values.

Takes a VERIS classification dict (with actor, action, asset, attribute keys)
and checks that all values match the enumerations defined in enums.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from veris_classifier.enums import (
        ACTION_ERROR_VARIETY,
        ACTION_HACKING_VARIETY,
        ACTION_MALWARE_VARIETY,
        ACTION_MISUSE_VARIETY,
        ACTION_PHYSICAL_VARIETY,
        ACTION_SOCIAL_VARIETY,
        ACTOR_EXTERNAL_VARIETY,
        ACTOR_INTERNAL_VARIETY,
        ACTOR_MOTIVE,
        ASSET_VARIETY,
        ATTRIBUTE_AVAILABILITY_VARIETY,
        ATTRIBUTE_CONFIDENTIALITY_DATA_VARIETY,
        ATTRIBUTE_INTEGRITY_VARIETY,
        DATA_DISCLOSURE,
    )
except ModuleNotFoundError:
    from src.veris_classifier.enums import (
        ACTION_ERROR_VARIETY,
        ACTION_HACKING_VARIETY,
        ACTION_MALWARE_VARIETY,
        ACTION_MISUSE_VARIETY,
        ACTION_PHYSICAL_VARIETY,
        ACTION_SOCIAL_VARIETY,
        ACTOR_EXTERNAL_VARIETY,
        ACTOR_INTERNAL_VARIETY,
        ACTOR_MOTIVE,
        ASSET_VARIETY,
        ATTRIBUTE_AVAILABILITY_VARIETY,
        ATTRIBUTE_CONFIDENTIALITY_DATA_VARIETY,
        ATTRIBUTE_INTEGRITY_VARIETY,
        DATA_DISCLOSURE,
    )

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

VALID_ACTOR_TYPES = {"external", "internal", "partner"}

ACTOR_VARIETY_BY_TYPE: dict[str, set[str]] = {
    "external": set(ACTOR_EXTERNAL_VARIETY),
    "internal": set(ACTOR_INTERNAL_VARIETY),
    # VERIS does not define a separate partner variety list; partner entries
    # typically have an empty variety or reuse generic values.
    "partner": set(),
}

VALID_MOTIVES: set[str] = set(ACTOR_MOTIVE)

VALID_ACTION_TYPES = {
    "malware", "hacking", "social", "misuse",
    "physical", "error", "environmental",
}

ACTION_VARIETY_BY_TYPE: dict[str, set[str]] = {
    "malware": set(ACTION_MALWARE_VARIETY),
    "hacking": set(ACTION_HACKING_VARIETY),
    "social": set(ACTION_SOCIAL_VARIETY),
    "misuse": set(ACTION_MISUSE_VARIETY),
    "physical": set(ACTION_PHYSICAL_VARIETY),
    "error": set(ACTION_ERROR_VARIETY),
    # environmental has no variety list defined in our enums.
    "environmental": set(),
}

VALID_ASSET_VARIETIES: set[str] = set(ASSET_VARIETY)

VALID_ATTRIBUTE_TYPES = {"confidentiality", "integrity", "availability"}

ATTRIBUTE_VARIETY_BY_TYPE: dict[str, set[str]] = {
    "integrity": set(ATTRIBUTE_INTEGRITY_VARIETY),
    "availability": set(ATTRIBUTE_AVAILABILITY_VARIETY),
}

VALID_DATA_VARIETY: set[str] = set(ATTRIBUTE_CONFIDENTIALITY_DATA_VARIETY)
VALID_DATA_DISCLOSURE: set[str] = set(DATA_DISCLOSURE)

# Threshold for "too many unknowns" warning.  If more than this fraction of
# all list values across the classification are "Unknown", a warning is raised.
_UNKNOWN_WARNING_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of validating a VERIS classification dict."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def _add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.valid = False

    def _add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_list_values(
    values: list[str],
    valid_set: set[str],
    label: str,
    result: ValidationResult,
) -> int:
    """Validate each value in *values* against *valid_set*.

    Returns the count of "Unknown" values encountered.
    """
    unknown_count = 0
    for val in values:
        if val == "Unknown":
            unknown_count += 1
        if valid_set and val not in valid_set:
            result._add_error(f"{label}: invalid value '{val}'")
    return unknown_count


def _ensure_list(obj: Any) -> list:
    """Coerce *obj* to a list if it is not one already."""
    if isinstance(obj, list):
        return obj
    if obj is None:
        return []
    return [obj]


# ---------------------------------------------------------------------------
# Section validators
# ---------------------------------------------------------------------------

def _validate_actor(actor: dict, result: ValidationResult) -> int:
    """Validate the ``actor`` section.  Returns total unknown count."""
    unknown_count = 0

    if not isinstance(actor, dict):
        result._add_error("actor: expected a dict")
        return 0

    for actor_type, info in actor.items():
        if actor_type not in VALID_ACTOR_TYPES:
            result._add_error(f"actor: unknown actor type '{actor_type}'")
            continue

        if not isinstance(info, dict):
            result._add_error(f"actor.{actor_type}: expected a dict")
            continue

        # --- variety ---
        variety = _ensure_list(info.get("variety", []))
        valid_set = ACTOR_VARIETY_BY_TYPE.get(actor_type, set())
        unknown_count += _check_list_values(
            variety, valid_set,
            f"actor.{actor_type}.variety", result,
        )

        # --- motive ---
        motive = _ensure_list(info.get("motive", []))
        unknown_count += _check_list_values(
            motive, VALID_MOTIVES,
            f"actor.{actor_type}.motive", result,
        )

    return unknown_count


def _validate_action(action: dict, result: ValidationResult) -> int:
    """Validate the ``action`` section.  Returns total unknown count."""
    unknown_count = 0

    if not isinstance(action, dict):
        result._add_error("action: expected a dict")
        return 0

    for action_type, info in action.items():
        if action_type not in VALID_ACTION_TYPES:
            result._add_error(f"action: unknown action type '{action_type}'")
            continue

        if not isinstance(info, dict):
            result._add_error(f"action.{action_type}: expected a dict")
            continue

        # --- variety ---
        variety = _ensure_list(info.get("variety", []))
        valid_set = ACTION_VARIETY_BY_TYPE.get(action_type, set())
        unknown_count += _check_list_values(
            variety, valid_set,
            f"action.{action_type}.variety", result,
        )

        # --- vector ---
        # Vectors are present in the dataset but not yet enumerated in
        # enums.py.  We accept any non-empty string and skip strict
        # validation to avoid false positives.
        vector = _ensure_list(info.get("vector", []))
        for val in vector:
            if not isinstance(val, str) or not val.strip():
                result._add_error(
                    f"action.{action_type}.vector: "
                    f"expected non-empty string, got '{val}'"
                )
            if val == "Unknown":
                unknown_count += 1

    return unknown_count


def _validate_asset(asset: dict, result: ValidationResult) -> int:
    """Validate the ``asset`` section.  Returns total unknown count."""
    unknown_count = 0

    if not isinstance(asset, dict):
        result._add_error("asset: expected a dict")
        return 0

    variety = _ensure_list(asset.get("variety", []))
    unknown_count += _check_list_values(
        variety, VALID_ASSET_VARIETIES,
        "asset.variety", result,
    )

    return unknown_count


def _validate_attribute(attribute: dict, result: ValidationResult) -> int:
    """Validate the ``attribute`` section.  Returns total unknown count."""
    unknown_count = 0

    if not isinstance(attribute, dict):
        result._add_error("attribute: expected a dict")
        return 0

    for attr_type, info in attribute.items():
        if attr_type not in VALID_ATTRIBUTE_TYPES:
            result._add_error(
                f"attribute: unknown attribute type '{attr_type}'"
            )
            continue

        if not isinstance(info, dict):
            result._add_error(f"attribute.{attr_type}: expected a dict")
            continue

        if attr_type == "confidentiality":
            # --- data_disclosure ---
            disclosure = info.get("data_disclosure")
            if disclosure is not None:
                if disclosure not in VALID_DATA_DISCLOSURE:
                    result._add_error(
                        f"attribute.confidentiality.data_disclosure: "
                        f"invalid value '{disclosure}'"
                    )
                if disclosure == "Unknown":
                    unknown_count += 1

            # --- data_variety ---
            data_variety = _ensure_list(info.get("data_variety", []))
            unknown_count += _check_list_values(
                data_variety, VALID_DATA_VARIETY,
                "attribute.confidentiality.data_variety", result,
            )
        else:
            # integrity / availability
            variety = _ensure_list(info.get("variety", []))
            valid_set = ATTRIBUTE_VARIETY_BY_TYPE.get(attr_type, set())
            unknown_count += _check_list_values(
                variety, valid_set,
                f"attribute.{attr_type}.variety", result,
            )

    return unknown_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_classification(classification: dict) -> ValidationResult:
    """Validate a VERIS classification dict against known enumerations.

    Parameters
    ----------
    classification:
        A dict with top-level keys ``actor``, ``action``, ``asset``, and
        ``attribute``, following the structure produced by the classifier
        and used in the training dataset.

    Returns
    -------
    ValidationResult
        Contains ``valid`` (bool), ``errors`` (list of str), and
        ``warnings`` (list of str).
    """
    result = ValidationResult()

    if not isinstance(classification, dict):
        result._add_error("classification must be a dict")
        return result

    # Track totals for the "too many unknowns" warning.
    total_values = 0
    total_unknowns = 0

    # --- Required top-level keys ---
    required_keys = {"actor", "action", "asset", "attribute"}
    missing = required_keys - set(classification.keys())
    if missing:
        result._add_warning(
            f"missing top-level keys: {', '.join(sorted(missing))}"
        )

    # --- Actor ---
    actor = classification.get("actor")
    if actor is not None:
        unknowns = _validate_actor(actor, result)
        total_unknowns += unknowns
        # Count total values contributed by the actor section.
        for info in actor.values():
            if isinstance(info, dict):
                total_values += len(_ensure_list(info.get("variety", [])))
                total_values += len(_ensure_list(info.get("motive", [])))

    # --- Action ---
    action = classification.get("action")
    if action is not None:
        unknowns = _validate_action(action, result)
        total_unknowns += unknowns
        for info in action.values():
            if isinstance(info, dict):
                total_values += len(_ensure_list(info.get("variety", [])))
                total_values += len(_ensure_list(info.get("vector", [])))

    # --- Asset ---
    asset = classification.get("asset")
    if asset is not None:
        unknowns = _validate_asset(asset, result)
        total_unknowns += unknowns
        total_values += len(_ensure_list(asset.get("variety", [])))

    # --- Attribute ---
    attribute = classification.get("attribute")
    if attribute is not None:
        unknowns = _validate_attribute(attribute, result)
        total_unknowns += unknowns
        for attr_type, info in attribute.items():
            if isinstance(info, dict):
                if attr_type == "confidentiality":
                    total_values += len(
                        _ensure_list(info.get("data_variety", []))
                    )
                    if info.get("data_disclosure") is not None:
                        total_values += 1
                else:
                    total_values += len(
                        _ensure_list(info.get("variety", []))
                    )

    # --- Unknown saturation warning ---
    if total_values > 0:
        unknown_ratio = total_unknowns / total_values
        if unknown_ratio > _UNKNOWN_WARNING_THRESHOLD:
            result._add_warning(
                f"high ratio of 'Unknown' values: "
                f"{total_unknowns}/{total_values} "
                f"({unknown_ratio:.0%})"
            )

    return result

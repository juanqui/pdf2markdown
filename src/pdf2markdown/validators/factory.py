"""Factory for creating validators."""

import logging
from typing import Any

from pdf2markdown.validators.base import BaseValidator
from pdf2markdown.validators.markdown_validator import MarkdownValidator
from pdf2markdown.validators.repetition_validator import RepetitionValidator

logger = logging.getLogger(__name__)

# Registry of available validators
VALIDATOR_REGISTRY = {
    "markdown": MarkdownValidator,
    "repetition": RepetitionValidator,
}


def create_validator(validator_type: str, config: dict[str, Any]) -> BaseValidator:
    """Create a validator instance based on type.

    Args:
        validator_type: Type of validator to create (markdown, repetition, etc.)
        config: Configuration for the validator

    Returns:
        Configured validator instance

    Raises:
        ValueError: If validator type is not recognized
    """
    if validator_type not in VALIDATOR_REGISTRY:
        available = ", ".join(VALIDATOR_REGISTRY.keys())
        raise ValueError(f"Unknown validator type: {validator_type}. Available: {available}")

    validator_class = VALIDATOR_REGISTRY[validator_type]
    logger.debug(f"Creating {validator_type} validator with config: {config}")

    return validator_class(config)


def create_validators(validation_config: dict[str, Any]) -> list[BaseValidator]:
    """Create multiple validators from configuration.

    Args:
        validation_config: Full validation configuration with:
            - validators: List of validator names to enable
            - Individual validator configs by name

    Returns:
        List of configured validator instances

    Example config:
        {
            "validators": ["markdown", "repetition"],
            "markdown": {
                "enabled": true,
                "attempt_correction": true,
                "strict_mode": false
            },
            "repetition": {
                "enabled": true,
                "consecutive_threshold": 3
            }
        }
    """
    validators = []

    # Get list of validators to create
    validator_names = validation_config.get("validators", ["markdown"])

    for validator_name in validator_names:
        # Get config for this specific validator
        validator_config = validation_config.get(validator_name, {})

        # Skip if explicitly disabled
        if not validator_config.get("enabled", True):
            logger.info(f"Skipping disabled validator: {validator_name}")
            continue

        try:
            validator = create_validator(validator_name, validator_config)
            validators.append(validator)
            logger.debug(f"Created {validator_name} validator")
        except Exception as e:
            logger.error(f"Failed to create {validator_name} validator: {e}")
            # Continue with other validators

    return validators


def register_validator(name: str, validator_class: type[BaseValidator]) -> None:
    """Register a custom validator type.

    Args:
        name: Name to register the validator under
        validator_class: Validator class (must inherit from BaseValidator)
    """
    if not issubclass(validator_class, BaseValidator):
        raise TypeError(f"{validator_class} must inherit from BaseValidator")

    VALIDATOR_REGISTRY[name] = validator_class
    logger.info(f"Registered validator: {name}")

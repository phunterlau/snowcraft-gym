from snowgym_training.executor import EntityPolicy, ModelConfig, model_config
from snowgym_training.model import (
    EntityPolicy as CompatibilityEntityPolicy,
    ModelConfig as CompatibilityModelConfig,
    model_config as compatibility_model_config,
)


def test_executor_package_is_canonical_and_legacy_imports_remain_compatible() -> None:
    assert CompatibilityEntityPolicy is EntityPolicy
    assert CompatibilityModelConfig is ModelConfig
    assert compatibility_model_config is model_config


def test_executor_package_builds_the_default_repository_model() -> None:
    configuration = ModelConfig()
    policy = EntityPolicy(configuration)

    assert policy.config == configuration
    assert sum(parameter.numel() for parameter in policy.parameters()) > 0

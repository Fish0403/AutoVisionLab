"""Editable parameter spaces for supported classification models."""

from app.schemas.parameter_space import EditableParameterSpace


MOBILENET_V2_PARAMETER_SPACE = EditableParameterSpace.model_validate(
    {
        "model_name": "mobilenet_v2",
        "version": "mobilenet_v2@v1",
        "editable_params": {
            "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
            "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
            "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
            "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
            "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
            "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
            "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
            "augmentation_level": {"type": "enum", "choices": ["low", "medium", "high"]},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
        },
    }
)

GOOGLENET_PARAMETER_SPACE = EditableParameterSpace.model_validate(
    {
        "model_name": "googlenet",
        "version": "googlenet@v1",
        "editable_params": {
            "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
            "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
            "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
            "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
            "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
            "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
            "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
            "augmentation_level": {"type": "enum", "choices": ["low", "medium", "high"]},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            "aux_logits": {"type": "enum", "choices": [True, False]},
        },
    }
)


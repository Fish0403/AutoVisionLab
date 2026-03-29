"""Editable parameter spaces for supported classification models."""

from app.schemas.parameter_space import EditableParameterSpace


MOBILENET_V3_SMALL_PARAMETER_SPACE = EditableParameterSpace.model_validate(
    {
        "model_name": "mobilenet_v3_small",
        "version": "mobilenet_v3_small@v1",
        "editable_params": {
            "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
            "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
            "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
            "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
            "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
            "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
            "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
            "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
            "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
            "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
            "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            "neck_name": {"type": "enum", "choices": ["avg_pool", "gem_pool"]},
            "head_name": {"type": "enum", "choices": ["native_classifier", "linear", "dropout_linear"]},
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
            "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
            "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
            "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
            "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            "aux_logits": {"type": "enum", "choices": [True, False]},
        },
    }
)

RESNET18_PARAMETER_SPACE = EditableParameterSpace.model_validate(
    {
        "model_name": "resnet18",
        "version": "resnet18@v1",
        "editable_params": {
            "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
            "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
            "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
            "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
            "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
            "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
            "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
            "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
            "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
            "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
            "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
            "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
            "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
        },
    }
)

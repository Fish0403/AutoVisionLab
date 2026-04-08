"""Test-only builders for structured experiment config payloads."""

from __future__ import annotations

from app.schemas.parameter_space import (
    DatasetRecipe,
    DatasetRecipeSource,
    DatasetRecipeSplits,
    ExperimentParams,
    TrainHyp,
    TrainHypAugmentation,
    TrainHypLoss,
)


def build_train_hyp_from_params(*, task_type: str, params: ExperimentParams) -> TrainHyp:
    """Build one structured train_hyp object from normalized params for tests."""
    return TrainHyp(
        task_type=task_type,
        optimizer=params.optimizer,
        lr0=params.learning_rate,
        weight_decay=params.weight_decay,
        scheduler=params.scheduler,
        epochs=params.epochs,
        batch_size=params.batch_size,
        image_size=params.image_size,
        label_smoothing=params.label_smoothing,
        fl_gamma=params.loss_params.focal_gamma if params.loss_name == "focal_loss" else 0.0,
        augmentation=TrainHypAugmentation(
            mixup=params.augmentation_params.mixup_alpha,
            cutmix=params.augmentation_params.cutmix_alpha,
            random_erasing=params.augmentation_params.random_erasing_prob,
        ),
        loss=TrainHypLoss(name=params.loss_name),
    )


def build_default_dataset_recipe(*, dataset_name: str, task_type: str) -> DatasetRecipe:
    """Build one default dataset recipe from repository path conventions for tests."""
    dataset_root = dataset_name.strip()
    return DatasetRecipe(
        task_type=task_type,
        dataset_name=dataset_name,
        source=DatasetRecipeSource(root_dir=f"data/raw/{dataset_root}"),
        splits=DatasetRecipeSplits(
            train_manifest=f"data/classification/{dataset_root}/train.txt",
            val_manifest=f"data/classification/{dataset_root}/val.txt",
            test_manifest=f"data/classification/{dataset_root}/test.txt",
        ),
    )

import type { DatasetSummary } from "../../types/domain";

export type SupportedModelName = "mobilenet_v2" | "mobilenet_v3_small" | "googlenet" | "resnet18";

export const MODEL_LABELS: Record<SupportedModelName, string> = {
  mobilenet_v2: "MobileNetV2",
  mobilenet_v3_small: "MobileNetV3 Small",
  googlenet: "GoogLeNet",
  resnet18: "ResNet18"
};

export const COMPARE_CANDIDATE_MODELS: SupportedModelName[] = [
  "mobilenet_v2",
  "mobilenet_v3_small",
  "googlenet"
];

export function getModelFamily(modelName: SupportedModelName): string {
  if (modelName.startsWith("mobilenet")) {
    return "mobilenet";
  }
  if (modelName === "googlenet") {
    return "googlenet";
  }
  return "resnet";
}

export interface TrainingFormValues {
  dataset: string;
  modelName: SupportedModelName;
  compareCandidateModels: SupportedModelName[];
  allowBasicHparamSearch: boolean;
  allowStrategySearch: boolean;
  allowLossSearch: boolean;
  allowAugmentationSearch: boolean;
  allowModelModuleSearch: boolean;
  useDemoMode: boolean;
  optimizer: "adamw" | "adam" | "sgd";
  learningRate: number;
  batchSize: number;
  imageSize: number;
  epochs: number;
  weightDecay: number;
  scheduler: "cosine" | "step" | "none";
  augmentationPolicy: "basic" | "none";
  labelSmoothing: number;
}

export function defaultFormValues(dataset: string, imageSize: number): TrainingFormValues {
  return {
    dataset,
    modelName: "mobilenet_v3_small",
    compareCandidateModels: [...COMPARE_CANDIDATE_MODELS],
    allowBasicHparamSearch: true,
    allowStrategySearch: true,
    allowLossSearch: true,
    allowAugmentationSearch: true,
    allowModelModuleSearch: false,
    useDemoMode: true,
    optimizer: "adamw",
    learningRate: 0.003,
    batchSize: 64,
    imageSize,
    epochs: 10,
    weightDecay: 0.0001,
    scheduler: "cosine",
    augmentationPolicy: "basic",
    labelSmoothing: 0.1
  };
}

export function buildRunName(dataset: string, modelName: SupportedModelName): string {
  const dateLabel = new Date().toISOString().slice(0, 10).replace(/-/g, "");
  return `${dataset}-${modelName.replace(/_/g, "-")}-${dateLabel}`;
}

export function buildTaskTitle(
  mode: "compare" | "search",
  dataset: string,
  modelName: SupportedModelName = "mobilenet_v3_small"
): string {
  if (mode === "compare") {
    return `Compare Models on ${dataset.toUpperCase()}`;
  }
  return `Optimize ${MODEL_LABELS[modelName]} on ${dataset.toUpperCase()}`;
}

export function getDatasetImageOptions(datasets: DatasetSummary[], datasetName: string): number[] {
  const dataset = datasets.find((item) => item.name === datasetName);
  if (dataset?.image_size_options?.length) {
    return dataset.image_size_options.map((value) => Number(value));
  }
  return [64];
}

export function buildSearchPolicy(values: TrainingFormValues) {
  return {
    allow_basic_hparam_search: values.allowBasicHparamSearch,
    allowed_basic_hparam_fields: values.allowBasicHparamSearch
      ? [
          "optimizer",
          "learning_rate",
          "batch_size",
          "weight_decay",
          "scheduler",
          "label_smoothing"
        ]
      : [],
    allow_strategy_search: values.allowStrategySearch,
    allow_loss_search: values.allowLossSearch,
    allow_augmentation_search: values.allowAugmentationSearch,
    allow_model_module_search: values.allowModelModuleSearch,
    require_manual_approval_for_high_impact_changes: true
  };
}

export function buildRankingPolicy() {
  return {
    primary_metric: "top1_acc",
    primary_metric_mode: "max",
    min_primary_metric_improvement: 0.01,
    primary_metric_parity_epsilon: 0.0005,
    tie_breaker_metric: "latency_ms",
    tie_breaker_mode: "min",
    min_tie_breaker_metric_improvement: 0.5,
    max_image_size: null
  };
}

export function buildExperimentConfig(
  values: TrainingFormValues,
  parameterSpaceVersion: string,
  modelName?: SupportedModelName,
) {
  const effectiveModelName = modelName ?? values.modelName;
  return {
    task_type: "classification",
    dataset: values.dataset,
    model_family: getModelFamily(effectiveModelName),
    model_name: effectiveModelName,
    parameter_space_version: parameterSpaceVersion,
    use_demo_mode: values.useDemoMode,
    participates_in_ranking: true,
    search_policy: buildSearchPolicy(values),
    ranking_policy: buildRankingPolicy(),
    params: {
      optimizer: values.optimizer,
      learning_rate: values.learningRate,
      batch_size: values.batchSize,
      image_size: values.imageSize,
      epochs: values.epochs,
      weight_decay: values.weightDecay,
      scheduler: values.scheduler,
      augmentation_policy: values.augmentationPolicy,
      augmentation_params: {
        mixup_alpha: 0,
        cutmix_alpha: 0,
        random_erasing_prob: 0
      },
      loss_name: "cross_entropy_with_label_smoothing",
      loss_params: {
        focal_gamma: 2
      },
      label_smoothing: values.labelSmoothing,
      aux_logits: effectiveModelName === "googlenet" ? false : null
    }
  };
}

import type { DatasetSummary, ModelDefaults, ModelSummary, SearchPolicy } from "../../types/domain";

const KNOWN_DATASET_IMAGE_OPTIONS: Record<string, number[]> = {
  neu: [200, 224, 256]
};

const DEFAULT_MODEL_NAME = "mobilenet_v3_small";

export type SupportedModelName = string;

export interface TrainingFormValues {
  dataset: string;
  modelName: SupportedModelName;
  compareCandidateModels: SupportedModelName[];
  useDemoMode: boolean;
  allowEpochSearch: boolean;
  optimizer: "adamw" | "adam" | "sgd";
  learningRate: number;
  batchSize: number;
  epochs: number;
  weightDecay: number;
  scheduler: "cosine" | "step" | "none";
  augmentationPolicy: "basic" | "none";
  labelSmoothing: number;
}

function formatModelLabel(modelName: string): string {
  return modelName
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function buildModelLookup(models: ModelSummary[]): Record<string, ModelSummary> {
  return Object.fromEntries(models.map((model) => [model.model_name, model]));
}

export function getDefaultModelName(models: ModelSummary[]): string {
  const defaultModel = models.find((model) => model.is_default && model.supports_search);
  if (defaultModel) {
    return defaultModel.model_name;
  }
  const firstSearchModel = models.find((model) => model.supports_search);
  return firstSearchModel?.model_name ?? DEFAULT_MODEL_NAME;
}

export function getCompareCandidateModels(models: ModelSummary[]): string[] {
  return models.filter((model) => model.supports_compare).map((model) => model.model_name);
}

export function getSearchModelNames(models: ModelSummary[]): string[] {
  return models.filter((model) => model.supports_search).map((model) => model.model_name);
}

export function getModelLabel(modelName: string, modelLookup?: Record<string, ModelSummary>): string {
  return modelLookup?.[modelName]?.label ?? formatModelLabel(modelName);
}

export function defaultFormValues(
  dataset: string,
  options?: {
    defaultModelName?: string;
    compareCandidateModels?: string[];
  }
): TrainingFormValues {
  return {
    dataset,
    modelName: options?.defaultModelName ?? DEFAULT_MODEL_NAME,
    compareCandidateModels: [...(options?.compareCandidateModels ?? [])],
    useDemoMode: true,
    allowEpochSearch: false,
    optimizer: "adamw",
    learningRate: 0.003,
    batchSize: 64,
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
  modelName: SupportedModelName = DEFAULT_MODEL_NAME,
  modelLabel?: string
): string {
  if (mode === "compare") {
    return `Compare Models on ${dataset.toUpperCase()}`;
  }
  return `Optimize ${modelLabel ?? formatModelLabel(modelName)} on ${dataset.toUpperCase()}`;
}

export function getPreferredDatasetName(datasets: DatasetSummary[], fallbackDataset = "neu"): string {
  const preferredDataset = datasets.find((item) => item.name.toLowerCase() === fallbackDataset.toLowerCase());
  if (preferredDataset) {
    return preferredDataset.name;
  }
  return datasets[0]?.name ?? fallbackDataset;
}

export function getDatasetImageOptions(datasets: DatasetSummary[], datasetName: string): number[] {
  const dataset = datasets.find((item) => item.name === datasetName);
  if (dataset?.image_size_options?.length) {
    return dataset.image_size_options.map((value) => Number(value));
  }
  const knownOptions = KNOWN_DATASET_IMAGE_OPTIONS[datasetName.toLowerCase()];
  if (knownOptions?.length) {
    return knownOptions;
  }
  return [64];
}

export function getDatasetBaselineImageSize(datasets: DatasetSummary[], datasetName: string): number {
  const dataset = datasets.find((item) => item.name === datasetName);
  const originalImageSize = Number(dataset?.original_image_size ?? 0);
  if (originalImageSize > 0) {
    return originalImageSize;
  }
  return getDatasetImageOptions(datasets, datasetName)[0] ?? 64;
}

function cloneJson<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function buildSearchPolicyFromDefaults(baseSearchPolicy: SearchPolicy, allowEpochSearch = false): SearchPolicy {
  const nextSearchPolicy = cloneJson(baseSearchPolicy);
  const allowedFields = new Set(nextSearchPolicy.allowed_basic_hparam_fields);
  if (allowEpochSearch) {
    allowedFields.add("epochs");
  } else {
    allowedFields.delete("epochs");
  }
  nextSearchPolicy.allow_basic_hparam_search = allowedFields.size > 0;
  nextSearchPolicy.allowed_basic_hparam_fields = [...allowedFields];
  return nextSearchPolicy;
}

export function buildExperimentConfig(
  values: TrainingFormValues,
  imageSize: number,
  modelDefaults: ModelDefaults,
  options?: {
    allowEpochSearch?: boolean;
  }
) {
  const modelRecipe = cloneJson(modelDefaults.default_model_recipe);
  const trainHyp = cloneJson(modelDefaults.default_train_hyp);
  const searchPolicy = buildSearchPolicyFromDefaults(
    modelDefaults.default_search_policy,
    Boolean(options?.allowEpochSearch)
  );
  const rankingPolicy = cloneJson(modelDefaults.default_ranking_policy);
  const effectiveModelName = modelDefaults.summary.model_name;
  const modelFamily = modelDefaults.summary.model_family;

  trainHyp.optimizer = values.optimizer;
  trainHyp.lr0 = values.learningRate;
  trainHyp.weight_decay = values.weightDecay;
  trainHyp.scheduler = values.scheduler;
  trainHyp.epochs = values.epochs;
  trainHyp.batch_size = values.batchSize;
  trainHyp.image_size = imageSize;
  trainHyp.label_smoothing = values.labelSmoothing;

  return {
    task_type: "classification",
    dataset: values.dataset,
    model_family: modelFamily,
    model_name: effectiveModelName,
    parameter_space_version: modelDefaults.parameter_space.version,
    use_demo_mode: values.useDemoMode,
    participates_in_ranking: true,
    search_policy: searchPolicy,
    ranking_policy: rankingPolicy,
    model_recipe: modelRecipe,
    train_hyp: trainHyp,
    dataset_recipe: {
      version: "dataset_recipe@v1",
      task_type: "classification",
      dataset_name: values.dataset,
      class_names: [],
      source: {
        root_dir: `data/raw/${values.dataset}`
      },
      splits: {
        train_manifest: `data/classification/${values.dataset}/train.txt`,
        val_manifest: `data/classification/${values.dataset}/val.txt`,
        test_manifest: `data/classification/${values.dataset}/test.txt`
      },
      metadata: {
        image_size_options: []
      }
    }
  };
}

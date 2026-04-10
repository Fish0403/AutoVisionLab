export interface DatasetSummary {
  name: string;
  is_ready_for_training?: boolean;
  train_manifest_exists?: boolean;
  val_manifest_exists?: boolean;
  classification_dir?: string | null;
  original_image_size?: number | null;
  image_width?: number | null;
  image_height?: number | null;
  image_size_options?: number[] | null;
  train_sample_count?: number;
  val_sample_count?: number;
  test_sample_count?: number;
  class_names?: string[];
  train_class_distribution?: Record<string, number>;
  val_class_distribution?: Record<string, number>;
  test_class_distribution?: Record<string, number>;
}

export interface RunListItem {
  id: string;
  name: string;
  dataset: string;
  model_name: string;
  status: string;
  experiment_count: number;
}

export interface ExperimentSummary {
  id: string;
  run_id: string;
  status: string;
  model_name?: string;
  decision?: string | null;
}

export interface RunDetail {
  id: string;
  name: string;
  dataset: string;
  model_name: string;
  status: string;
  best_experiment_id?: string | null;
  experiments: ExperimentSummary[];
}

export interface RunSummary {
  run_id: string;
  best_experiment_id?: string | null;
  keep_count: number;
  discard_count: number;
  crash_count: number;
  timeout_count: number;
}

export interface MetricsPayload {
  metric_name: string;
  available_metrics: string[];
  points: Array<{
    experiment_id: string;
    experiment_index: number;
    metric_name: string;
    metric_value: number;
  }>;
}

export interface RunTrendPayload {
  run_id: string;
  available_metrics: string[];
  series: Array<{
    metric_name: string;
    points: Array<{
      experiment_id: string;
      experiment_index: number;
      metric_name: string;
      metric_value: number;
    }>;
  }>;
}

export interface ExperimentDetail {
  id: string;
  run_id: string;
  status: string;
  decision?: string | null;
  decision_reason?: string | null;
  is_best_so_far?: boolean;
  config: {
    dataset: string;
    model_name: string;
    params: Record<string, unknown>;
  };
  proposal?: {
    hypothesis?: string;
    rationale?: string;
    expected_effect?: string;
    changes?: Record<string, unknown>;
  } | null;
  result?: {
    metrics?: Record<string, number>;
    resource?: Record<string, number>;
  } | null;
  reflection?: {
    summary?: string;
    insights?: string[];
    next_step_hint?: string;
  } | null;
}

export interface ParameterSpace {
  model_name: string;
  version: string;
  editable_params: Record<string, unknown>;
}

export interface SearchPolicy {
  allow_basic_hparam_search: boolean;
  allowed_basic_hparam_fields: string[];
  allow_strategy_search: boolean;
  allow_loss_search: boolean;
  allow_augmentation_search: boolean;
  allow_model_module_search: boolean;
  require_manual_approval_for_high_impact_changes: boolean;
}

export interface RankingPolicy {
  primary_metric: string;
  primary_metric_mode: string;
  min_primary_metric_improvement: number;
  primary_metric_parity_epsilon: number;
  tie_breaker_metric: string;
  tie_breaker_mode: string;
  min_tie_breaker_metric_improvement: number;
  max_image_size: number | null;
}

export interface ModelRecipePayload {
  version: string;
  task_type: "classification";
  model_family: string;
  base_model: string;
  nc?: number | null;
  input_channels?: number;
  width_multiple?: number;
  components?: Record<string, unknown> | null;
  backbone_config?: Record<string, unknown>;
  backbone?: unknown[];
  neck?: unknown[];
  head_config?: Record<string, unknown>;
  head?: unknown[];
  modules: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface TrainHypPayload {
  version: string;
  task_type: "classification";
  optimizer: string;
  lr0: number;
  weight_decay: number;
  scheduler: string;
  epochs: number;
  batch_size: number;
  image_size: number;
  label_smoothing: number;
  fl_gamma?: number;
  augmentation: {
    mixup: number;
    cutmix: number;
    random_erasing: number;
    [key: string]: unknown;
  };
  loss: {
    name: string;
    [key: string]: unknown;
  };
  runtime?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ModelSummary {
  model_name: string;
  label: string;
  task_type: "classification";
  model_family: string;
  supports_compare: boolean;
  supports_search: boolean;
  is_default: boolean;
  display_order: number;
}

export interface ModelDefaults {
  summary: ModelSummary;
  parameter_space: ParameterSpace;
  default_model_recipe: ModelRecipePayload;
  default_train_hyp: TrainHypPayload;
  default_search_policy: SearchPolicy;
  default_ranking_policy: RankingPolicy;
}

export interface ModelManifestValidationResult {
  is_valid: boolean;
  yaml_parse_ok: boolean;
  schema_ok: boolean;
  dry_run_build_ok: boolean;
  dry_run_forward_ok: boolean;
  errors: string[];
  warnings: string[];
}

export interface ModelManifestDraftResponse {
  query: string;
  resolved_model_name: string;
  ai_preview_text: string;
  yaml_text: string;
  target_path: string;
  provider_warnings: string[];
  validation: ModelManifestValidationResult;
}

export interface ModelManifestCommitResponse {
  model_name: string;
  manifest_path: string;
  reloaded_model_count: number;
}

export interface AutoTrainTask {
  task_id: string;
  title?: string | null;
  status: string;
  dataset?: string | null;
  model_name?: string | null;
  policy_preset?: string | null;
  search_scope_summary?: string | null;
  run_id?: string | null;
  source_task_type?: "model_compare" | null;
  source_task_id?: string | null;
  source_task_title?: string | null;
  source_model_name?: string | null;
  current_round?: number;
  elapsed_seconds?: number;
  current_experiment_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  activity_message?: string | null;
  proposal_warning?: string | null;
  dataset_summary?: string | null;
  training_image_size?: number | null;
  ai_model_name?: string | null;
  logs: string[];
  summary?: {
    mode?: "auto";
    run_id?: string;
    baseline?: {
      experiment_id?: string;
      status?: string;
      decision?: string | null;
      decision_reason?: string | null;
      metrics?: Record<string, number>;
      resource?: Record<string, number>;
      summary?: string;
    };
    rounds?: Array<{
      round_index?: number;
      proposal?: {
        hypothesis?: string;
        changes?: Record<string, unknown>;
        train_hyp_changes?: Record<string, unknown> | null;
        recipe_changes?: Record<string, unknown> | null;
      };
      result?: {
        experiment_id?: string;
        status?: string;
        decision?: string | null;
        decision_reason?: string | null;
        metrics?: Record<string, number>;
        resource?: Record<string, number>;
        summary?: string;
      };
    }>;
    current_proposal?: {
      hypothesis?: string;
      changes?: Record<string, unknown>;
      train_hyp_changes?: Record<string, unknown> | null;
      recipe_changes?: Record<string, unknown> | null;
    } | null;
    ai_summary?: string | null;
    ai_summary_error?: string | null;
    stop_reason?: string | null;
  } | null;
  error?: string | null;
  stop_reason?: string | null;
  estimated_prompt_tokens_total?: number;
  latest_prompt_history_items?: number | null;
  latest_provider_prompt_tokens?: number | null;
  latest_provider_completion_tokens?: number | null;
  latest_provider_total_tokens?: number | null;
  provider_prompt_tokens_total?: number;
  provider_completion_tokens_total?: number;
  provider_total_tokens_total?: number;
}

export interface ModelCompareCandidateResult {
  model_name: string;
  run_id?: string | null;
  baseline_experiment_id?: string | null;
  status: string;
  top1_acc?: number | null;
  latency_ms?: number | null;
  parameter_count_million?: number | null;
  normalized_config_notes?: string[];
}

export interface ModelCompareTask {
  task_id: string;
  title?: string | null;
  status: string;
  dataset?: string | null;
  candidate_models?: string[];
  elapsed_seconds?: number;
  current_model_name?: string | null;
  current_model_index?: number;
  total_models?: number;
  current_run_id?: string | null;
  current_experiment_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  activity_message?: string | null;
  dataset_summary?: string | null;
  training_image_size?: number | null;
  ai_model_name?: string | null;
  logs: string[];
  summary?: {
    mode: "model_compare";
    shared_baseline_config: Record<string, unknown>;
    candidate_results: ModelCompareCandidateResult[];
    ai_summary?: string | null;
    ai_summary_error?: string | null;
  } | null;
  error?: string | null;
  stop_requested?: boolean;
  stop_reason?: string | null;
}

export interface TaskHistoryItem {
  task_id: string;
  task_type: "auto_train" | "model_compare";
  title: string;
  status: string;
  summary?: string | null;
  dataset?: string | null;
  model_name?: string | null;
  candidate_models?: string[];
  policy_preset?: string | null;
  run_id?: string | null;
  source_task_type?: "model_compare" | null;
  source_task_id?: string | null;
  source_task_title?: string | null;
  source_model_name?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

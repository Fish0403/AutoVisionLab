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

import { useMutation, useQuery } from "@tanstack/react-query";

import { getJson, postJson } from "../../lib/api";
import type {
  AutoTrainTask,
  DatasetSummary,
  ExperimentDetail,
  MetricsPayload,
  ModelCompareTask,
  ParameterSpace,
  RunDetail,
  RunListItem,
  RunSummary,
  TaskHistoryItem
} from "../../types/domain";

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: () => getJson<DatasetSummary[]>("/datasets")
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => getJson<{ status: string }>("/health"),
    retry: false,
    refetchInterval: 10000
  });
}

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: () => getJson<RunListItem[]>("/runs")
  });
}

export function useTaskHistory() {
  return useQuery({
    queryKey: ["task-history"],
    queryFn: () => getJson<TaskHistoryItem[]>("/runs/tasks"),
    refetchInterval: 2000
  });
}

export function useRunDetail(runId: string | null) {
  return useQuery({
    queryKey: ["run-detail", runId],
    queryFn: () => getJson<RunDetail>(`/runs/${runId}`),
    enabled: Boolean(runId)
  });
}

export function useRunSummary(runId: string | null) {
  return useQuery({
    queryKey: ["run-summary", runId],
    queryFn: () => getJson<RunSummary>(`/runs/${runId}/summary`),
    enabled: Boolean(runId)
  });
}

export function useRunMetrics(runId: string | null, metricName = "top1_acc") {
  return useQuery({
    queryKey: ["run-metrics", runId, metricName],
    queryFn: () => getJson<MetricsPayload>(`/runs/${runId}/metrics?metric_name=${metricName}`),
    enabled: Boolean(runId)
  });
}

export function useExperimentDetail(experimentId: string | null) {
  return useQuery({
    queryKey: ["experiment-detail", experimentId],
    queryFn: () => getJson<ExperimentDetail>(`/experiments/${experimentId}`),
    enabled: Boolean(experimentId)
  });
}

export function useParameterSpace(modelName: string) {
  return useQuery({
    queryKey: ["parameter-space", modelName],
    queryFn: () => getJson<ParameterSpace>(`/models/${modelName}/parameter-space`),
    enabled: Boolean(modelName)
  });
}

export function useAutoTrainTask(taskId: string | null) {
  return useQuery({
    queryKey: ["auto-train-task", taskId],
    queryFn: () => getJson<AutoTrainTask>(`/runs/auto-train/${taskId}`),
    enabled: Boolean(taskId),
    refetchIntervalInBackground: true,
    refetchOnWindowFocus: true,
    refetchInterval: (query) => {
      const status = (query.state.data as AutoTrainTask | undefined)?.status;
      return status && !["stopped", "stopped_by_policy", "failed"].includes(status) ? 2000 : false;
    }
  });
}

export function useActiveAutoTrainTask() {
  return useQuery({
    queryKey: ["auto-train-task", "active"],
    queryFn: () => getJson<AutoTrainTask>("/runs/auto-train/active"),
    retry: false,
    refetchIntervalInBackground: true,
    refetchOnWindowFocus: true,
    refetchInterval: 2000
  });
}

export function useModelCompareTask(taskId: string | null) {
  return useQuery({
    queryKey: ["model-compare-task", taskId],
    queryFn: () => getJson<ModelCompareTask>(`/runs/model-compare/${taskId}`),
    enabled: Boolean(taskId),
    refetchInterval: (query) => {
      const status = (query.state.data as ModelCompareTask | undefined)?.status;
      return status && !["success", "failed", "stopped"].includes(status) ? 2000 : false;
    }
  });
}

export function useActiveModelCompareTask() {
  return useQuery({
    queryKey: ["model-compare-task", "active"],
    queryFn: () => getJson<ModelCompareTask>("/runs/model-compare/active"),
    retry: false,
    refetchInterval: 2000
  });
}

export function useStartAutoTrain() {
  return useMutation({
    mutationFn: (payload: unknown) => postJson<AutoTrainTask>("/runs/auto-train", payload)
  });
}

export function useStopAutoTrain() {
  return useMutation({
    mutationFn: (taskId: string) => postJson<AutoTrainTask>(`/runs/auto-train/${taskId}/stop`)
  });
}

export function useStartModelCompare() {
  return useMutation({
    mutationFn: (payload: unknown) => postJson<ModelCompareTask>("/runs/model-compare", payload)
  });
}

export function useStopModelCompare() {
  return useMutation({
    mutationFn: (taskId: string) => postJson<ModelCompareTask>(`/runs/model-compare/${taskId}/stop`)
  });
}

export function useRenameTaskTitle() {
  return useMutation({
    mutationFn: ({ taskId, taskType, title }: { taskId: string; taskType: "auto_train" | "model_compare"; title: string }) =>
      postJson<{ task_id: string; title: string }>(`/runs/tasks/${taskType}/${taskId}/title`, { title })
  });
}

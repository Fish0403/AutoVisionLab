import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { Link, useOutletContext, useSearchParams } from "react-router-dom";

import { EMPTY_APP_SHELL_HEADER_CONTENT, type AppShellOutletContext } from "../app/AppShell";
import { FormField } from "../components/FormField";
import {
  useActiveAutoTrainTask,
  useActiveModelCompareTask,
  useAutoTrainTask,
  useDatasets,
  useModelCompareTask,
  useParameterSpace,
  useRenameTaskTitle,
  useRunDetail,
  useStartAutoTrain,
  useStartModelCompare,
  useStopAutoTrain,
  useStopModelCompare
} from "../features/api/hooks";
import {
  buildExperimentConfig,
  buildRunName,
  buildTaskTitle,
  COMPARE_CANDIDATE_MODELS,
  defaultFormValues,
  getDatasetImageOptions,
  MODEL_LABELS,
  type SupportedModelName,
  type TrainingFormValues
} from "../features/training/config";
import { getJson } from "../lib/api";
import { readLocalStorage, writeLocalStorage } from "../lib/storage";
import type {
  AutoTrainTask,
  ExperimentDetail,
  MetricsPayload,
  ModelCompareCandidateResult,
  ModelCompareTask,
} from "../types/domain";

const SELECTED_TASK_ID_STORAGE_KEY = "autovisionlab:selected-task-id";
const SELECTED_TASK_TYPE_STORAGE_KEY = "autovisionlab:selected-task-type";
const COMPARE_ANCHOR_MODEL: SupportedModelName = "mobilenet_v3_small";
const SEARCH_TREND_METRICS = [
  { metricName: "top1_acc", label: "Top1 Acc", color: "#2b59ff", axis: "left" as const, family: "accuracy" as const, defaultVisible: true },
  { metricName: "val_loss", label: "Val Loss", color: "#d9485f", axis: "right" as const, family: "loss" as const, defaultVisible: true },
  { metricName: "train_loss", label: "Train Loss", color: "#ff8a00", axis: "right" as const, family: "loss" as const, defaultVisible: true },
  { metricName: "latency_ms", label: "Latency", color: "#11a36a", axis: "right" as const, family: "latency" as const, defaultVisible: false },
  { metricName: "best_epoch", label: "Best Epoch", color: "#7a5af8", axis: "right" as const, family: "epoch" as const, defaultVisible: false }
];

type WorkspaceMode = "compare" | "search";
type TaskType = "auto_train" | "model_compare";
type DisplayMode = "chart" | "table";
type CompareSearchCandidate = {
  modelName: SupportedModelName;
  runId: string;
  dataset: string;
  top1Acc?: number | null;
  latencyMs?: number | null;
  parameterCountMillion?: number | null;
};

export function WorkspacePage() {
  const queryClient = useQueryClient();
  const { setHeaderContent } = useOutletContext<AppShellOutletContext>();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedTaskId = searchParams.get("taskId");
  const requestedTaskType = searchParams.get("taskType") as TaskType | null;
  const requestedNewTask = searchParams.get("new") === "1";

  const datasetsQuery = useDatasets();
  const activeAutoTrainQuery = useActiveAutoTrainTask();
  const activeModelCompareQuery = useActiveModelCompareTask();

  const datasets = datasetsQuery.data ?? [];
  const defaultDataset = datasets[0]?.name ?? "neu";
  const defaultImageSize = getDatasetImageOptions(datasets, defaultDataset)[0] ?? 64;

  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(
    () => requestedTaskId ?? readLocalStorage(SELECTED_TASK_ID_STORAGE_KEY)
  );
  const [selectedTaskType, setSelectedTaskType] = useState<TaskType | null>(
    () => requestedTaskType ?? (readLocalStorage(SELECTED_TASK_TYPE_STORAGE_KEY) as TaskType | null)
  );
  const [mode, setMode] = useState<WorkspaceMode>(() => (requestedTaskType === "model_compare" ? "compare" : "search"));
  const [displayMode, setDisplayMode] = useState<DisplayMode>("chart");
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [isSearchLaunchPending, setIsSearchLaunchPending] = useState(false);
  const [draftTaskTitle, setDraftTaskTitle] = useState("Untitled task");
  const [formValues, setFormValues] = useState<TrainingFormValues>(() => defaultFormValues(defaultDataset, defaultImageSize));
  const [selectedCompareCandidate, setSelectedCompareCandidate] = useState<CompareSearchCandidate | null>(null);

  const autoTrainTaskQuery = useAutoTrainTask(selectedTaskType === "auto_train" ? selectedTaskId : null);
  const modelCompareTaskQuery = useModelCompareTask(selectedTaskType === "model_compare" ? selectedTaskId : null);

  const currentAutoTask =
    selectedTaskType === "auto_train"
      ? resolveCurrentAutoTask(
          selectedTaskId,
          activeAutoTrainQuery.data ?? null,
          autoTrainTaskQuery.data ?? null,
          activeAutoTrainQuery.isError
        )
      : null;
  const currentCompareTask =
    selectedTaskType === "model_compare"
      ? resolveCurrentCompareTask(
          selectedTaskId,
          activeModelCompareQuery.data ?? null,
          modelCompareTaskQuery.data ?? null,
          activeModelCompareQuery.isError
        )
      : null;

  useEffect(() => {
    if (requestedNewTask) {
      setSelectedTaskId(null);
      setSelectedTaskType(null);
      setSelectedCompareCandidate(null);
      setDisplayMode("chart");
      setIsEditingTitle(false);
      setIsSearchLaunchPending(false);
      setDraftTaskTitle("Untitled task");
      setMode("compare");
      setFormValues(defaultFormValues(defaultDataset, defaultImageSize));
      writeLocalStorage(SELECTED_TASK_ID_STORAGE_KEY, null);
      writeLocalStorage(SELECTED_TASK_TYPE_STORAGE_KEY, null);
      return;
    }
    if (requestedTaskId) {
      setSelectedTaskId(requestedTaskId);
      setSelectedCompareCandidate(null);
    }
    if (requestedTaskType) {
      setSelectedTaskType(requestedTaskType);
      setMode(requestedTaskType === "model_compare" ? "compare" : "search");
      setSelectedCompareCandidate(null);
    }
  }, [defaultDataset, defaultImageSize, requestedNewTask, requestedTaskId, requestedTaskType]);

  useEffect(() => {
    writeLocalStorage(SELECTED_TASK_ID_STORAGE_KEY, selectedTaskId);
    writeLocalStorage(SELECTED_TASK_TYPE_STORAGE_KEY, selectedTaskType);
  }, [selectedTaskId, selectedTaskType]);

  useEffect(() => {
    if (requestedNewTask) {
      return;
    }
    if (selectedTaskId || selectedTaskType) {
      return;
    }
    if (activeModelCompareQuery.data?.task_id) {
      selectTask(activeModelCompareQuery.data.task_id, "model_compare");
      return;
    }
    if (activeAutoTrainQuery.data?.task_id) {
      selectTask(activeAutoTrainQuery.data.task_id, "auto_train");
    }
  }, [activeAutoTrainQuery.data?.task_id, activeModelCompareQuery.data?.task_id, requestedNewTask, selectedTaskId, selectedTaskType]);

  useEffect(() => {
    if (!datasets.length) {
      return;
    }
    setFormValues((previous) => {
      const nextDataset = datasets.some((dataset) => dataset.name === previous.dataset) ? previous.dataset : defaultDataset;
      const imageOptions = getDatasetImageOptions(datasets, nextDataset);
      const nextImageSize = imageOptions.includes(previous.imageSize) ? previous.imageSize : imageOptions[0] ?? previous.imageSize;
      return {
        ...previous,
        dataset: nextDataset,
        imageSize: nextImageSize
      };
    });
  }, [datasets, defaultDataset]);

  useEffect(() => {
    if (!selectedTaskId || !currentAutoTask) {
      return;
    }
    setSelectedCompareCandidate(null);
    setMode("search");
    setDisplayMode("chart");
    setDraftTaskTitle(currentAutoTask.title ?? buildTaskTitle("search", currentAutoTask.dataset ?? defaultDataset, (currentAutoTask.model_name as SupportedModelName | undefined) ?? "mobilenet_v3_small"));
    setFormValues((previous) => ({
      ...previous,
      dataset: currentAutoTask.dataset ?? previous.dataset,
      modelName: (currentAutoTask.model_name as SupportedModelName | undefined) ?? previous.modelName
    }));
  }, [currentAutoTask, defaultDataset, selectedTaskId]);

  useEffect(() => {
    if (!selectedTaskId || !currentCompareTask) {
      return;
    }
    setSelectedCompareCandidate(null);
    setMode("compare");
    setDisplayMode("chart");
    setDraftTaskTitle(currentCompareTask.title ?? buildTaskTitle("compare", currentCompareTask.dataset ?? defaultDataset));
    setFormValues((previous) => ({
      ...previous,
      dataset: currentCompareTask.dataset ?? previous.dataset,
      compareCandidateModels: normalizeCandidateModels(currentCompareTask.candidate_models)
    }));
  }, [currentCompareTask, defaultDataset, selectedTaskId]);

  useEffect(() => {
    if (formValues.modelName === "mobilenet_v3_small") {
      return;
    }
    if (!formValues.allowModelModuleSearch) {
      return;
    }
    setFormValues((previous) => ({
      ...previous,
      allowModelModuleSearch: false
    }));
  }, [formValues.allowModelModuleSearch, formValues.modelName]);

  const isSearchDraftFromCompare = Boolean(selectedCompareCandidate && selectedTaskType === "model_compare" && !currentAutoTask);
  const controlMode: WorkspaceMode = isSearchDraftFromCompare ? "search" : mode;
  const activeCompareSource = isSearchDraftFromCompare ? selectedCompareCandidate : null;
  const parameterSpaceModelName = controlMode === "compare" ? COMPARE_ANCHOR_MODEL : formValues.modelName;
  const parameterSpaceQuery = useParameterSpace(parameterSpaceModelName);
  const renameTaskTitleMutation = useRenameTaskTitle();
  const startAutoTrainMutation = useStartAutoTrain();
  const startModelCompareMutation = useStartModelCompare();
  const stopAutoTrainMutation = useStopAutoTrain();
  const stopModelCompareMutation = useStopModelCompare();

  const compareTask = currentCompareTask;
  const compareSummary = compareTask?.summary ?? null;
  const candidateResults = compareSummary?.candidate_results ?? [];
  const successfulCandidates = candidateResults.filter((candidate) => candidate.status === "success" && candidate.run_id);
  const completedCandidateCount = candidateResults.filter((candidate) => isCandidateTerminalStatus(candidate.status)).length;
  const compareDisplayStatus = getCompareDisplayStatus(compareTask);
  const compareRecommendation = getRecommendedCandidate(successfulCandidates);
  const compareResultSummary = buildCompareResultSummary(compareTask, compareRecommendation);

  const isDetachedSearchDraft = Boolean(
    currentAutoTask &&
      isAutoTrainTerminalStatus(currentAutoTask.status) &&
      ((currentAutoTask.model_name && formValues.modelName !== currentAutoTask.model_name) ||
        (currentAutoTask.dataset && formValues.dataset !== currentAutoTask.dataset))
  );
  const displayAutoTask = isDetachedSearchDraft || isSearchLaunchPending ? null : currentAutoTask;
  const autoTrainSummary = readAutoTrainSummary(displayAutoTask);
  const latestRound = [...(autoTrainSummary?.rounds ?? [])].reverse()[0] ?? null;
  const searchTerminalSummary = buildSearchTerminalSummary(displayAutoTask, autoTrainSummary, latestRound);
  const showSearchTerminalSummary = Boolean(searchTerminalSummary && displayAutoTask && isAutoTrainTerminalStatus(displayAutoTask.status));

  const currentTaskTitle =
    (mode === "compare" ? currentCompareTask?.title : displayAutoTask?.title) ??
    (mode === "search" && isSearchLaunchPending ? draftTaskTitle : null) ??
    (isDetachedSearchDraft ? buildSearchTaskTitle(formValues.dataset, formValues.modelName) : null) ??
    (selectedTaskId || selectedCompareCandidate ? draftTaskTitle : "Untitled task");
  const currentTaskStatus =
    (mode === "compare" ? compareDisplayStatus : displayAutoTask?.status) ??
    (mode === "search" && isSearchLaunchPending ? "running" : "draft");
  const compareSourceTaskId = displayAutoTask?.source_task_id ?? null;
  const hasCompareBackLink = Boolean(
    controlMode === "search" && displayAutoTask?.source_task_type === "model_compare" && compareSourceTaskId
  );

  const isCompareRunning = Boolean(compareTask && !isCompareTerminalStatus(compareDisplayStatus));
  const isSearchRunning = Boolean(displayAutoTask && !["stopped", "stopped_by_policy", "failed"].includes(displayAutoTask.status));
  const hasEnabledSearchDimension =
    formValues.allowBasicHparamSearch ||
    formValues.allowStrategySearch ||
    formValues.allowLossSearch ||
    formValues.allowAugmentationSearch ||
    formValues.allowModelModuleSearch;
  const isPrimaryActionBusy =
    startAutoTrainMutation.isPending || startModelCompareMutation.isPending || (controlMode === "compare" ? isCompareRunning : isSearchRunning);

  const currentSearchRunId = displayAutoTask?.run_id ?? null;
  const runDetailQuery = useRunDetail(currentSearchRunId);
  const searchMetricQueries = useQueries({
    queries: SEARCH_TREND_METRICS.map((metric) => ({
      queryKey: ["run-metrics", currentSearchRunId, metric.metricName],
      queryFn: () => getJson<MetricsPayload>(`/runs/${currentSearchRunId}/metrics?metric_name=${metric.metricName}`),
      enabled: Boolean(currentSearchRunId)
    }))
  });
  const runDetail = runDetailQuery.data ?? null;

  const searchTrendSeries = useMemo(
    () =>
      SEARCH_TREND_METRICS.map((metric, index) => ({
        ...metric,
        points: searchMetricQueries[index]?.data?.points ?? []
      })).filter((series) => series.points.length > 0),
    [searchMetricQueries]
  );

  const recentExperimentIds = useMemo(() => {
    if (!runDetail?.experiments?.length) {
      return [];
    }
    return [...runDetail.experiments].slice(-6).reverse().map((experiment) => experiment.id);
  }, [runDetail?.experiments]);

  const recentExperimentQueries = useQueries({
    queries: recentExperimentIds.map((experimentId) => ({
      queryKey: ["experiment-detail", experimentId],
      queryFn: () => getJson<ExperimentDetail>(`/experiments/${experimentId}`),
      enabled: Boolean(experimentId)
    }))
  });

  const recentExperiments = recentExperimentQueries
    .map((query) => query.data)
    .filter((experiment): experiment is ExperimentDetail => Boolean(experiment));

  const handleModeSelect = (nextMode: WorkspaceMode) => {
    if (nextMode === "compare" && isSearchDraftFromCompare) {
      setSelectedCompareCandidate(null);
      return;
    }
    setMode(nextMode);
    setDisplayMode("chart");
    if (selectedTaskType && selectedTaskType !== toTaskType(nextMode)) {
      setSelectedTaskId(null);
      setSelectedTaskType(null);
      setSearchParams({});
      setIsEditingTitle(false);
      setDraftTaskTitle("Untitled task");
    }
  };

  const handleRun = async () => {
    const parameterSpace = parameterSpaceQuery.data;
    if (!parameterSpace) {
      return;
    }

    if (controlMode === "compare") {
      const generatedTitle = selectedTaskId ? currentTaskTitle : buildTaskTitle("compare", formValues.dataset);
      setDraftTaskTitle(generatedTitle);
      const response = await startModelCompareMutation.mutateAsync({
        title: generatedTitle,
        dataset: formValues.dataset,
        candidate_models: formValues.compareCandidateModels,
        config: buildExperimentConfig(formValues, parameterSpace.version, COMPARE_ANCHOR_MODEL)
      });
      await queryClient.invalidateQueries({ queryKey: ["task-history"] });
      selectTask(response.task_id, "model_compare");
      return;
    }

    const generatedTitle =
      displayAutoTask?.task_id && !isDetachedSearchDraft ? currentTaskTitle : buildSearchTaskTitle(formValues.dataset, formValues.modelName);
    setDraftTaskTitle(generatedTitle);
    const shouldStartFreshSearchRun = Boolean(
      (activeCompareSource && selectedTaskType === "model_compare" && !displayAutoTask) || isDetachedSearchDraft
    );
    setIsSearchLaunchPending(true);
    try {
      const response = await startAutoTrainMutation.mutateAsync({
        title: generatedTitle,
        run_id: shouldStartFreshSearchRun ? null : (displayAutoTask?.run_id ?? null),
        run_name: runDetail?.name ?? buildRunName(formValues.dataset, formValues.modelName),
        dataset: formValues.dataset,
        model_name: formValues.modelName,
        source_task_type: activeCompareSource && selectedTaskType === "model_compare" ? "model_compare" : null,
        source_task_id: activeCompareSource && selectedTaskType === "model_compare" ? selectedTaskId : null,
        source_task_title: activeCompareSource && selectedTaskType === "model_compare" ? currentCompareTask?.title ?? null : null,
        source_model_name: activeCompareSource?.modelName ?? null,
        config: buildExperimentConfig(formValues, parameterSpace.version),
        parameter_space: parameterSpace
      });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["task-history"] }),
        queryClient.invalidateQueries({ queryKey: ["auto-train-task", response.task_id] }),
        queryClient.invalidateQueries({ queryKey: ["auto-train-task", "active"] })
      ]);
      selectTask(response.task_id, "auto_train");
    } finally {
      setIsSearchLaunchPending(false);
    }
  };

  const handleTitleSave = async () => {
    if (!selectedTaskId || !selectedTaskType) {
      setIsEditingTitle(false);
      return;
    }
    const normalizedTitle = draftTaskTitle.trim();
    if (!normalizedTitle) {
      setDraftTaskTitle(currentTaskTitle || "Untitled task");
      setIsEditingTitle(false);
      return;
    }
    await renameTaskTitleMutation.mutateAsync({
      taskId: selectedTaskId,
      taskType: selectedTaskType,
      title: normalizedTitle
    });
    setIsEditingTitle(false);
    await queryClient.invalidateQueries({ queryKey: ["task-history"] });
    await queryClient.invalidateQueries({ queryKey: ["auto-train-task", selectedTaskId] });
    await queryClient.invalidateQueries({ queryKey: ["model-compare-task", selectedTaskId] });
  };

  const handleStopSearch = async () => {
    if (!displayAutoTask?.task_id) {
      return;
    }
    await stopAutoTrainMutation.mutateAsync(displayAutoTask.task_id);
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["task-history"] }),
      queryClient.invalidateQueries({ queryKey: ["auto-train-task", displayAutoTask.task_id] }),
      queryClient.invalidateQueries({ queryKey: ["auto-train-task", "active"] })
    ]);
  };

  const handleStopCompare = async () => {
    if (!compareTask?.task_id) {
      return;
    }
    await stopModelCompareMutation.mutateAsync(compareTask.task_id);
    await queryClient.invalidateQueries({ queryKey: ["task-history"] });
  };

  const prepareSearchFromCompareCandidate = (candidate: CompareSearchCandidate) => {
    setSelectedCompareCandidate(candidate);
    setIsEditingTitle(false);
    setIsSearchLaunchPending(false);
    setDraftTaskTitle(buildTaskTitle("search", candidate.dataset, candidate.modelName));
    setFormValues((previous) => ({
      ...previous,
      dataset: candidate.dataset,
      modelName: candidate.modelName
    }));
  };

  const clearCompareSearchSource = () => {
    setSelectedCompareCandidate(null);
  };

  function selectTask(taskId: string, taskType: TaskType) {
    setSelectedCompareCandidate(null);
    setIsSearchLaunchPending(false);
    setSelectedTaskId(taskId);
    setSelectedTaskType(taskType);
    setMode(taskType === "model_compare" ? "compare" : "search");
    setDisplayMode("chart");
    setSearchParams({
      taskId,
      taskType
    });
  }

  const headerTitle = useMemo(() => {
    if (isEditingTitle && selectedTaskId) {
      return (
        <input
          autoFocus
          className="title-input global-title-input"
          onBlur={() => void handleTitleSave()}
          onChange={(event) => setDraftTaskTitle(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void handleTitleSave();
            }
            if (event.key === "Escape") {
              setDraftTaskTitle(currentTaskTitle);
              setIsEditingTitle(false);
            }
          }}
          type="text"
          value={draftTaskTitle}
        />
      );
    }
    return (
      <h1
        className={selectedTaskId ? "global-page-title global-page-title-editable" : "global-page-title"}
        onClick={() => {
          if (!selectedTaskId) {
            return;
          }
          setDraftTaskTitle(currentTaskTitle);
          setIsEditingTitle(true);
        }}
      >
        {selectedTaskId ? currentTaskTitle : "Untitled task"}
      </h1>
    );
  }, [currentTaskTitle, draftTaskTitle, handleTitleSave, isEditingTitle, selectedTaskId]);

  const headerActions = useMemo(
    () => (
      <>
        <Link className="text-button workspace-back-link" to="/">
          Back to History
        </Link>
        <div className={`status-pill status-${normalizeStatusTone(currentTaskStatus)}`}>{formatStatusLabel(currentTaskStatus)}</div>
      </>
    ),
    [currentTaskStatus]
  );

  useEffect(() => {
    setHeaderContent({
      title: headerTitle,
      actions: headerActions
    });
    return () => setHeaderContent(EMPTY_APP_SHELL_HEADER_CONTENT);
  }, [headerActions, headerTitle, setHeaderContent]);

  return (
    <div className="workspace-page">
      <section className="workspace-layout">
        <aside className="control-panel">
          <div className="control-section">
            <div className="section-heading">
              <h2>Parameters</h2>
            </div>
            <div className="form-grid">
              <FormField label="Dataset">
                <select
                  disabled={datasets.length === 0}
                  value={formValues.dataset}
                  onChange={(event) => {
                    const nextDataset = event.target.value;
                    const nextImageSize = getDatasetImageOptions(datasets, nextDataset)[0] ?? formValues.imageSize;
                    setFormValues((previous) => ({
                      ...previous,
                      dataset: nextDataset,
                      imageSize: nextImageSize
                    }));
                  }}
                >
                  {datasets.length === 0 ? <option value={formValues.dataset}>Loading datasets...</option> : null}
                  {datasets.map((dataset) => (
                    <option key={dataset.name} value={dataset.name}>
                      {dataset.name}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Image Size">
                <select
                  disabled={datasets.length === 0}
                  value={formValues.imageSize}
                  onChange={(event) => setFormValues((previous) => ({ ...previous, imageSize: Number(event.target.value) }))}
                >
                  {(getDatasetImageOptions(datasets, formValues.dataset).length
                    ? getDatasetImageOptions(datasets, formValues.dataset)
                    : [formValues.imageSize]
                  ).map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Epochs">
                <select
                  value={formValues.epochs}
                  onChange={(event) => setFormValues((previous) => ({ ...previous, epochs: Number(event.target.value) }))}
                >
                  {[10, 20, 30, 50].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Batch Size">
                <select
                  value={formValues.batchSize}
                  onChange={(event) => setFormValues((previous) => ({ ...previous, batchSize: Number(event.target.value) }))}
                >
                  {[32, 64, 128, 256].map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </FormField>
              <FormField label="Learning Rate">
                <input
                  type="number"
                  min={0.0001}
                  max={0.05}
                  step={0.0001}
                  value={formValues.learningRate}
                  onChange={(event) => setFormValues((previous) => ({ ...previous, learningRate: Number(event.target.value) }))}
                />
              </FormField>
              <FormField label="Demo Mode">
                <select
                  value={formValues.useDemoMode ? "true" : "false"}
                  onChange={(event) =>
                    setFormValues((previous) => ({
                      ...previous,
                      useDemoMode: event.target.value === "true"
                    }))
                  }
                >
                  <option value="true">Enabled</option>
                  <option value="false">Disabled</option>
                </select>
              </FormField>
            </div>
          </div>

          <div className="control-section control-section-mode">
            <div className="section-heading">
              <h2>Mode</h2>
            </div>
            <div className="segmented-control">
              <button
                className={`segment-button${controlMode === "compare" ? " segment-button-active" : ""}`}
                onClick={() => handleModeSelect("compare")}
                type="button"
              >
                Compare
              </button>
              <button
                className={`segment-button${controlMode === "search" ? " segment-button-active" : ""}`}
                onClick={() => handleModeSelect("search")}
                type="button"
              >
                Search
              </button>
            </div>
            {hasCompareBackLink ? (
              <div className="search-model-link-row">
                <span className="workspace-label">Model</span>
                <Link
                  className="text-button"
                  to={`/workspace?taskId=${compareSourceTaskId}&taskType=model_compare`}
                >
                  Back to Compare
                </Link>
              </div>
            ) : null}
            {controlMode === "search" ? (
              isSearchDraftFromCompare ? (
                <div className="details-panel search-model-panel">
                  {hasCompareBackLink ? null : (
                    <span className="workspace-label">Model</span>
                  )}
                  <strong>{MODEL_LABELS[formValues.modelName]}</strong>
                  <p>Locked to the selected compare candidate.</p>
                </div>
              ) : (
                <>
                  {hasCompareBackLink ? (
                    <div className="form-field">
                      <select
                        value={formValues.modelName}
                        onChange={(event) =>
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: event.target.value as SupportedModelName
                          }))
                        }
                      >
                        {Object.entries(MODEL_LABELS).map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : (
                    <FormField label="Model">
                      <select
                        value={formValues.modelName}
                        onChange={(event) =>
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: event.target.value as SupportedModelName
                          }))
                        }
                      >
                        {Object.entries(MODEL_LABELS).map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </FormField>
                  )}
                </>
              )
            ) : null}
            {controlMode === "compare" ? (
              <div className="popover-panel mode-options-panel">
                <div className="section-heading">
                  <h3>Models</h3>
                </div>
                <div className="option-stack">
                  {COMPARE_CANDIDATE_MODELS.map((modelName) => {
                    const isChecked = formValues.compareCandidateModels.includes(modelName);
                    return (
                      <label className="option-row" key={modelName}>
                        <input
                          checked={isChecked}
                          onChange={(event) => {
                            setFormValues((previous) => ({
                              ...previous,
                              compareCandidateModels: event.target.checked
                                ? [...previous.compareCandidateModels, modelName]
                                : previous.compareCandidateModels.filter((candidateModel) => candidateModel !== modelName)
                            }));
                          }}
                          type="checkbox"
                        />
                        <span>{MODEL_LABELS[modelName]}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            ) : null}
            {controlMode === "search" ? (
              <div className="popover-panel mode-options-panel">
                <div className="section-heading">
                  <h3>Search Scope</h3>
                </div>
                <div className="option-stack">
                  <label className="option-row">
                    <input
                      checked={formValues.allowBasicHparamSearch}
                      onChange={(event) =>
                        setFormValues((previous) => ({
                          ...previous,
                          allowBasicHparamSearch: event.target.checked
                        }))
                      }
                      type="checkbox"
                    />
                    <span>Basic Hyperparameter Search</span>
                  </label>
                  <label className="option-row">
                    <input
                      checked={formValues.allowStrategySearch}
                      onChange={(event) =>
                        setFormValues((previous) => ({
                          ...previous,
                          allowStrategySearch: event.target.checked
                        }))
                      }
                      type="checkbox"
                    />
                    <span>Strategy Search</span>
                  </label>
                  <label className="option-row">
                    <input
                      checked={formValues.allowLossSearch}
                      onChange={(event) =>
                        setFormValues((previous) => ({
                          ...previous,
                          allowLossSearch: event.target.checked
                        }))
                      }
                      type="checkbox"
                    />
                    <span>Loss Search</span>
                  </label>
                  <label className="option-row">
                    <input
                      checked={formValues.allowAugmentationSearch}
                      onChange={(event) =>
                        setFormValues((previous) => ({
                          ...previous,
                          allowAugmentationSearch: event.target.checked
                        }))
                      }
                      type="checkbox"
                    />
                    <span>Augmentation Search</span>
                  </label>
                  <label className="option-row">
                    <input
                      checked={formValues.allowModelModuleSearch}
                      disabled={formValues.modelName !== "mobilenet_v3_small"}
                      onChange={(event) =>
                        setFormValues((previous) => ({
                          ...previous,
                          allowModelModuleSearch: event.target.checked
                        }))
                      }
                      type="checkbox"
                    />
                    <span>Model Architecture Search</span>
                  </label>
                </div>
              </div>
            ) : null}
          </div>

          {activeCompareSource ? (
            <div className="details-panel search-source-panel">
              <div className="search-source-row">
                <span className="workspace-label">Selected Source</span>
                <button className="text-button" onClick={() => clearCompareSearchSource()} type="button">
                  Clear
                </button>
              </div>
              <div className="search-source-copy">
                <span>
                  {MODEL_LABELS[activeCompareSource.modelName]} · {truncateId(activeCompareSource.runId)}
                </span>
                <em>{buildCompareCandidateMeta(activeCompareSource) ?? activeCompareSource.dataset}</em>
              </div>
            </div>
          ) : null}

          <div className="control-actions">
            <button
              className="primary-button"
              disabled={
                isPrimaryActionBusy ||
                !parameterSpaceQuery.data ||
                (controlMode === "compare" && formValues.compareCandidateModels.length === 0) ||
                (controlMode === "search" && !hasEnabledSearchDimension)
              }
              onClick={() => void handleRun()}
              type="button"
            >
              {isPrimaryActionBusy ? "Running..." : controlMode === "compare" ? "Run Compare" : "Start Search"}
            </button>
            {controlMode === "search" && !hasEnabledSearchDimension ? (
              <p className="control-hint-copy">Enable at least one search dimension to start search.</p>
            ) : null}
            {displayAutoTask ? (
              <button
                className="secondary-button"
                disabled={!isSearchRunning || stopAutoTrainMutation.isPending || displayAutoTask.status === "stopping"}
                onClick={() => void handleStopSearch()}
                type="button"
              >
                {displayAutoTask.status === "stopping" || stopAutoTrainMutation.isPending ? "Stopping..." : "Stop Search"}
              </button>
            ) : (
              <button
                className="secondary-button"
                disabled={!isCompareRunning || stopModelCompareMutation.isPending}
                onClick={() => void handleStopCompare()}
                type="button"
              >
                Stop Compare
              </button>
            )}
            {(
              datasetsQuery.error ??
              parameterSpaceQuery.error ??
              startAutoTrainMutation.error ??
              startModelCompareMutation.error ??
              stopAutoTrainMutation.error ??
              stopModelCompareMutation.error
            ) ? (
              <p className="error-copy">
                {datasetsQuery.error?.message ??
                  parameterSpaceQuery.error?.message ??
                  startAutoTrainMutation.error?.message ??
                  startModelCompareMutation.error?.message ??
                  stopAutoTrainMutation.error?.message ??
                  stopModelCompareMutation.error?.message}
              </p>
            ) : null}
          </div>
        </aside>

        <section className="display-panel">
          <div className={`progress-panel${mode === "search" ? " progress-panel-plain" : ""}`}>
            {mode === "compare" ? (
              <CompareProgressCard
                task={compareTask}
                completedCandidateCount={completedCandidateCount}
                displayStatus={compareDisplayStatus}
              />
            ) : (
              <SearchProgressCard task={displayAutoTask} />
            )}
          </div>

          <div className="display-header">
            <div>
              <h2>{mode === "compare" ? "Compare Results" : "Search Results"}</h2>
            </div>
            <div className="segmented-control">
              <button
                className={`segment-button${displayMode === "chart" ? " segment-button-active" : ""}`}
                onClick={() => setDisplayMode("chart")}
                type="button"
              >
                Chart
              </button>
              <button
                className={`segment-button${displayMode === "table" ? " segment-button-active" : ""}`}
                onClick={() => setDisplayMode("table")}
                type="button"
              >
                Table
              </button>
            </div>
          </div>

          {mode === "compare" ? (
            displayMode === "chart" ? (
              <div className="display-stack">
                {compareResultSummary ? <p className="result-summary-copy">{compareResultSummary}</p> : null}
                <CompareScatterChart
                  candidates={successfulCandidates}
                  dataset={compareTask?.dataset ?? formValues.dataset}
                  onSelectCandidate={(candidate) => prepareSearchFromCompareCandidate(candidate)}
                  selectedCandidateModelName={selectedCompareCandidate?.modelName ?? null}
                />
              </div>
            ) : (
              <div className="table-card">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Status</th>
                      <th>Top1 Acc</th>
                      <th>Latency</th>
                      <th>Params</th>
                    </tr>
                  </thead>
                  <tbody>
                    {candidateResults.length === 0 ? (
                      <tr>
                        <td colSpan={5}>No compare results yet.</td>
                      </tr>
                    ) : (
                      candidateResults.map((candidate) => (
                        <tr
                          className={candidate.run_id && selectedCompareCandidate?.runId === candidate.run_id ? "compare-result-row-selected" : ""}
                          key={candidate.model_name}
                          onClick={() => {
                            const searchCandidate = buildCompareSearchCandidate(candidate, compareTask?.dataset ?? formValues.dataset);
                            if (!searchCandidate) {
                              return;
                            }
                            prepareSearchFromCompareCandidate(searchCandidate);
                          }}
                        >
                          <td>{MODEL_LABELS[candidate.model_name as SupportedModelName] ?? candidate.model_name}</td>
                          <td>{formatStatusLabel(candidate.status)}</td>
                          <td>{formatAccuracyDetailed(candidate.top1_acc)}</td>
                          <td>{formatLatency(candidate.latency_ms)}</td>
                          <td>{formatParameterCount(candidate.parameter_count_million)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )
          ) : displayMode === "chart" ? (
            <div className="display-stack">
              {showSearchTerminalSummary && searchTerminalSummary ? (
                <div className="summary-highlight">
                  <div className="summary-highlight-header">
                    <SummaryIcon />
                    <span>Latest Summary</span>
                  </div>
                  <h3>{searchTerminalSummary.title}</h3>
                  <p>{searchTerminalSummary.body}</p>
                </div>
              ) : null}
              <MetricTrendChart series={searchTrendSeries} />
            </div>
          ) : (
            <div className="table-card">
              <table>
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Status</th>
                    <th>Decision</th>
                    <th>Top1 Acc</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {recentExperiments.length === 0 ? (
                    <tr>
                      <td colSpan={5}>No experiment results yet.</td>
                    </tr>
                  ) : (
                    recentExperiments.map((experiment) => (
                      <tr key={experiment.id}>
                        <td className="experiment-cell" title={experiment.id}>
                          {experiment.id}
                        </td>
                        <td>{formatStatusLabel(experiment.status)}</td>
                        <td>{experiment.decision ?? "—"}</td>
                        <td>{formatAccuracyDetailed(experiment.result?.metrics?.top1_acc)}</td>
                        <td>{formatLatency(experiment.result?.resource?.latency_ms)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </section>
    </div>
  );
}

function CompareProgressCard({
  task,
  completedCandidateCount,
  displayStatus
}: {
  task: ModelCompareTask | null;
  completedCandidateCount: number;
  displayStatus: string;
}) {
  const totalModels = task?.total_models ?? task?.candidate_models?.length ?? 0;
  const progress = totalModels > 0 ? Math.min(100, Math.round((completedCandidateCount / totalModels) * 100)) : 0;
  const isTerminal = isCompareTerminalStatus(displayStatus);
  const compareMetaEntries = buildCompareMetaEntries(task, completedCandidateCount, totalModels, displayStatus);
  const compareSummary =
    task?.current_model_name && !isTerminal
      ? `Currently running ${MODEL_LABELS[task.current_model_name as SupportedModelName] ?? task.current_model_name}.`
      : task
        ? `Ready to compare ${totalModels || 0} selected models.`
        : "Waiting for a compare task to start.";

  return (
    <div>
      <div className="progress-activity-block">
        <div className="progress-activity-header">
          <h2>Activity Log</h2>
        </div>
        <div className="progress-activity-divider" />
      </div>
      <p className="progress-suggestion-copy">{compareSummary}</p>
      <div className="progress-meta-row">
        {compareMetaEntries.map((entry) => (
          <span className="progress-meta-chip" key={entry.label}>
            {entry.value}
          </span>
        ))}
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}

function SearchProgressCard({ task }: { task: AutoTrainTask | null }) {
  const summary = readAutoTrainSummary(task);
  const activeProposal = task && !isAutoTrainTerminalStatus(task.status) ? summary?.current_proposal ?? null : getActiveSearchProposal(summary);
  const searchFocus = describeProposalChangeSummary(activeProposal);
  const suggestionText = getSearchSuggestionText(task, summary, activeProposal);
  const [elapsedDisplay, setElapsedDisplay] = useState(task?.elapsed_seconds ?? 0);

  useEffect(() => {
    setElapsedDisplay(task?.elapsed_seconds ?? 0);
  }, [task?.task_id, task?.elapsed_seconds]);

  useEffect(() => {
    if (!task || isAutoTrainTerminalStatus(task.status)) {
      return;
    }
    const timerId = window.setInterval(() => {
      setElapsedDisplay((previous) => previous + 1);
    }, 1000);
    return () => window.clearInterval(timerId);
  }, [task?.task_id, task?.status]);

  const progressHeadline = searchFocus ? `Focus ${searchFocus}` : null;
  const metaEntries = buildSearchMetaEntries(task, elapsedDisplay);

  return (
    <div>
      <div className="progress-activity-block">
        <div className="progress-activity-header">
          <h2>Activity Log</h2>
        </div>
        <div className="progress-activity-divider" />
      </div>
      {progressHeadline ? (
        <div className="progress-inline-note">
          <SearchFocusIcon />
          <span>{progressHeadline}</span>
        </div>
      ) : null}
      {suggestionText ? (
        <div className="progress-suggestion-row">
          <SuggestionIcon />
          <p className="progress-suggestion-copy">{suggestionText}</p>
        </div>
      ) : null}
      <div className="progress-meta-row">
        {metaEntries.map((entry) => (
          <span className="progress-meta-chip" key={entry.label}>
            {entry.value}
          </span>
        ))}
      </div>
    </div>
  );
}

function SearchFocusIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon" viewBox="0 0 16 16">
      <circle cx="7" cy="7" fill="none" r="4.25" stroke="currentColor" strokeWidth="1.5" />
      <path d="M10.4 10.4 14 14" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function SuggestionIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon" viewBox="0 0 16 16">
      <path
        d="M8 2.25a4.25 4.25 0 0 0-2.92 7.34c.5.46.88 1.02 1.11 1.66h3.62c.23-.64.61-1.2 1.11-1.66A4.25 4.25 0 0 0 8 2.25Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
      <path d="M6.25 13.1h3.5M6.75 15h2.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function SummaryIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon summary-inline-icon" viewBox="0 0 16 16">
      <path d="M3 3.25h10v9.5H3z" fill="none" stroke="currentColor" strokeLinejoin="round" strokeWidth="1.5" />
      <path d="M5.25 6h5.5M5.25 8.5h5.5M5.25 11h3.5" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function CompareScatterChart({
  candidates,
  dataset,
  onSelectCandidate,
  selectedCandidateModelName
}: {
  candidates: ModelCompareCandidateResult[];
  dataset: string;
  onSelectCandidate: (candidate: CompareSearchCandidate) => void;
  selectedCandidateModelName: string | null;
}) {
  if (candidates.length === 0) {
    return <EmptyStateCard copy="Run compare to populate the scatter chart." />;
  }

  const width = 760;
  const height = 360;
  const padding = 68;
  const latencies = candidates.map((candidate) => candidate.latency_ms ?? 0);
  const accuracies = candidates.map((candidate) => candidate.top1_acc ?? 0);
  const rawMinLatency = Math.min(...latencies);
  const rawMaxLatency = Math.max(...latencies);
  const rawMinAccuracy = Math.min(...accuracies);
  const rawMaxAccuracy = Math.max(...accuracies);
  const latencySpan = rawMaxLatency - rawMinLatency;
  const accuracySpan = rawMaxAccuracy - rawMinAccuracy;
  const latencyPadding = Math.max(latencySpan * 0.18, rawMaxLatency * 0.08, 2);
  const accuracyPadding = Math.max(accuracySpan * 0.22, 0.015);
  const minLatency = Math.max(0, rawMinLatency - latencyPadding);
  const maxLatency = rawMaxLatency + latencyPadding;
  const minAccuracy = Math.max(0, rawMinAccuracy - accuracyPadding);
  const maxAccuracy = Math.min(1, rawMaxAccuracy + accuracyPadding);
  const latencyRange = Math.max(maxLatency - minLatency, 1);
  const accuracyRange = Math.max(maxAccuracy - minAccuracy, 0.01);
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const xTicks = [minLatency, (minLatency + maxLatency) / 2, maxLatency];
  const yTicks = [maxAccuracy, (minAccuracy + maxAccuracy) / 2, minAccuracy];
  const [activeModelName, setActiveModelName] = useState<string | null>(null);

  const points = candidates.map((candidate) => {
    const latency = candidate.latency_ms ?? 0;
    const accuracy = candidate.top1_acc ?? 0;
    const x = padding + ((latency - minLatency) / latencyRange) * plotWidth;
    const y = height - padding - ((accuracy - minAccuracy) / accuracyRange) * plotHeight;
    return {
      candidate,
      x,
      y
    };
  });
  const activePoint =
    points.find((point) => point.candidate.model_name === activeModelName) ??
    null;
  const tooltipWidth = 188;
  const tooltipHeight = 104;
  const tooltipOffset = 14;
  const tooltipX = activePoint
    ? Math.min(
        Math.max(
          activePoint.x + (activePoint.x > width - padding - tooltipWidth ? -tooltipWidth - tooltipOffset : tooltipOffset),
          padding + 6
        ),
        width - padding - tooltipWidth - 6
      )
    : 0;
  const tooltipY = activePoint
    ? Math.min(
        Math.max(
          activePoint.y + (activePoint.y < padding + tooltipHeight ? tooltipOffset : -tooltipHeight - tooltipOffset),
          padding + 6
        ),
        height - padding - tooltipHeight - 6
      )
    : 0;

  return (
    <div className="chart-card">
      <svg aria-label="Compare scatter chart" className="chart-svg" role="img" viewBox={`0 0 ${width} ${height}`}>
        {xTicks.map((tick) => {
          const x = padding + ((tick - minLatency) / latencyRange) * plotWidth;
          return (
            <g key={`x-${tick}`}>
              <line className="chart-grid" x1={x} x2={x} y1={padding} y2={height - padding} />
              <text className="chart-axis-label" textAnchor="middle" x={x} y={height - padding + 26}>
                {formatLatencyTick(tick)}
              </text>
            </g>
          );
        })}
        {yTicks.map((tick) => {
          const y = height - padding - ((tick - minAccuracy) / accuracyRange) * plotHeight;
          return (
            <g key={`y-${tick}`}>
              <line className="chart-grid" x1={padding} x2={width - padding} y1={y} y2={y} />
              <text className="chart-axis-label" textAnchor="end" x={padding - 12} y={y + 4}>
                {formatAccuracyTick(tick)}
              </text>
            </g>
          );
        })}
        <line className="chart-axis" x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
        <line className="chart-axis" x1={padding} x2={padding} y1={padding} y2={height - padding} />
        {points.map((point) => (
          <g
            key={point.candidate.model_name}
            onMouseEnter={() => setActiveModelName(point.candidate.model_name)}
            onMouseLeave={() => setActiveModelName(null)}
            onClick={() => {
              const searchCandidate = buildCompareSearchCandidate(point.candidate, dataset);
              if (!searchCandidate) {
                return;
              }
              onSelectCandidate(searchCandidate);
            }}
          >
            <circle
              className="chart-dot compare-dot"
              cx={point.x}
              cy={point.y}
              r={
                activePoint?.candidate.model_name === point.candidate.model_name ||
                selectedCandidateModelName === point.candidate.model_name
                  ? 10
                  : 8
              }
              style={{
                fill: getComparePointColor(point.candidate.model_name),
                stroke:
                  activePoint?.candidate.model_name === point.candidate.model_name ||
                  selectedCandidateModelName === point.candidate.model_name
                    ? "#ffffff"
                    : "#f7f8fb"
              }}
            />
            <text className="chart-label" x={point.x + 10} y={point.y - 8}>
              {MODEL_LABELS[point.candidate.model_name as SupportedModelName] ?? point.candidate.model_name}
            </text>
            {activePoint?.candidate.model_name === point.candidate.model_name && point.candidate.run_id ? (
              <g transform={`translate(${tooltipX} ${tooltipY})`}>
                <rect className="chart-tooltip-box" height={tooltipHeight} rx={14} ry={14} width={tooltipWidth} x={0} y={0} />
                <circle
                  cx={14}
                  cy={18}
                  fill={getComparePointColor(activePoint.candidate.model_name)}
                  r={5}
                />
                <text className="chart-tooltip-title" x={26} y={22}>
                  {MODEL_LABELS[activePoint.candidate.model_name as SupportedModelName] ?? activePoint.candidate.model_name}
                </text>
                <text className="chart-tooltip-line" x={14} y={42}>
                  {`Top1 ${formatAccuracy(activePoint.candidate.top1_acc)}`}
                </text>
                <text className="chart-tooltip-line" x={14} y={56}>
                  {`Latency ${formatLatency(activePoint.candidate.latency_ms)}`}
                </text>
                <text className="chart-tooltip-line" x={14} y={70}>
                  {`Params ${formatParameterCount(activePoint.candidate.parameter_count_million)}`}
                </text>
              </g>
            ) : null}
          </g>
        ))}
        <text className="chart-axis-title" textAnchor="middle" x={width / 2} y={height - 6}>
          Latency (ms)
        </text>
        <text className="chart-axis-title" textAnchor="middle" transform={`translate(16 ${height / 2}) rotate(-90)`}>
          Top1 Accuracy
        </text>
      </svg>
    </div>
  );
}

function MetricTrendChart({
  series
}: {
  series: Array<{
    metricName: string;
    label: string;
    color: string;
    axis: "left" | "right";
    family: "accuracy" | "loss" | "latency" | "epoch";
    defaultVisible: boolean;
    points: Array<{
      experiment_id: string;
      experiment_index: number;
      metric_name: string;
      metric_value: number;
    }>;
  }>;
}) {
  if (!series.length) {
    return <EmptyStateCard copy="Start or continue a search task to populate the trend chart." />;
  }

  const width = 760;
  const height = 380;
  const padding = {
    top: 30,
    right: 68,
    bottom: 52,
    left: 68
  };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const [visibleMetricNames, setVisibleMetricNames] = useState<string[]>(() =>
    series.filter((item) => item.defaultVisible).map((item) => item.metricName)
  );
  const availableMetricNames = series.map((item) => item.metricName);

  useEffect(() => {
    setVisibleMetricNames((previous) => {
      const availableNameSet = new Set(availableMetricNames);
      const retained = previous.filter((metricName) => availableNameSet.has(metricName));
      if (retained.length > 0) {
        return retained;
      }
      const defaultVisible = series.filter((item) => item.defaultVisible).map((item) => item.metricName);
      return defaultVisible.length > 0 ? defaultVisible : availableMetricNames.slice(0, 1);
    });
  }, [availableMetricNames, series]);

  const visibleSeries = series.filter((item) => visibleMetricNames.includes(item.metricName));
  if (!visibleSeries.length) {
    return <EmptyStateCard copy="Select at least one metric to display the trend chart." />;
  }

  const experimentIndexes = Array.from(
    new Set(visibleSeries.flatMap((item) => item.points.map((point) => point.experiment_index)))
  ).sort((left, right) => left - right);
  const firstExperimentIndex = experimentIndexes[0] ?? 1;
  const lastExperimentIndex = experimentIndexes[experimentIndexes.length - 1] ?? firstExperimentIndex;
  const xTickIndexes =
    experimentIndexes.length <= 6
      ? experimentIndexes
      : Array.from(new Set([firstExperimentIndex, ...sampleTickIndexes(experimentIndexes, 6), lastExperimentIndex]));

  const leftValues = visibleSeries.filter((item) => item.axis === "left").flatMap((item) => item.points.map((point) => point.metric_value));
  const rightFamilies = Array.from(new Set(visibleSeries.filter((item) => item.axis === "right").map((item) => item.family)));
  const rightFamily = (rightFamilies[0] as "loss" | "latency" | "epoch" | undefined) ?? "loss";
  const rightValues = visibleSeries.filter((item) => item.axis === "right").flatMap((item) => item.points.map((point) => point.metric_value));
  const leftDomain = buildMetricDomain(leftValues, "accuracy");
  const rightDomain = buildMetricDomain(rightValues, rightFamily);
  const leftTicks = buildYAxisTicks(leftDomain.min, leftDomain.max, 5);
  const rightTicks = buildYAxisTicks(rightDomain.min, rightDomain.max, 5);

  const getX = (experimentIndex: number) =>
    padding.left + ((experimentIndex - firstExperimentIndex) / Math.max(lastExperimentIndex - firstExperimentIndex, 1)) * chartWidth;
  const getY = (value: number, axis: "left" | "right") => {
    const domain = axis === "left" ? leftDomain : rightDomain;
    const normalized = (value - domain.min) / Math.max(domain.max - domain.min, 1e-9);
    return height - padding.bottom - normalized * chartHeight;
  };

  const chartSeries = visibleSeries.map((item) => {
    const points = [...item.points]
      .sort((left, right) => left.experiment_index - right.experiment_index)
      .map((point) => ({
        ...point,
        x: getX(point.experiment_index),
        y: getY(point.metric_value, item.axis)
      }));
    const pathData = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
    return {
      ...item,
      plottedPoints: points,
      pathData
    };
  });

  const toggleMetricVisibility = (metricName: string) => {
    const selectedMetric = series.find((item) => item.metricName === metricName);
    if (!selectedMetric) {
      return;
    }
    setVisibleMetricNames((previous) => {
      const isVisible = previous.includes(metricName);
      if (isVisible) {
        if (previous.length === 1) {
          return previous;
        }
        return previous.filter((name) => name !== metricName);
      }
      if (selectedMetric.family === "accuracy") {
        return [...previous, metricName];
      }
      const allowedFamilies = new Set(["accuracy", selectedMetric.family]);
      const next = previous.filter((name) => {
        const metric = series.find((item) => item.metricName === name);
        return metric ? allowedFamilies.has(metric.family) : false;
      });
      return [...next, metricName];
    });
  };

  return (
    <div className="chart-card">
      <div className="chart-legend">
        {series.map((item) => {
          const isActive = visibleMetricNames.includes(item.metricName);
          return (
            <button
              className={`chart-legend-item${isActive ? " chart-legend-item-active" : ""}`}
              key={item.metricName}
              onClick={() => toggleMetricVisibility(item.metricName)}
              type="button"
            >
            <span aria-hidden="true" className="chart-legend-swatch" style={{ backgroundColor: item.color }} />
            {item.label}
            </button>
          );
        })}
      </div>
      <svg aria-label="Metric trend chart" className="chart-svg" role="img" viewBox={`0 0 ${width} ${height}`}>
        {leftValues.length > 0
          ? leftTicks.map((tick) => {
          const y = getY(tick, "left");
          return (
            <g key={`grid-${tick}`}>
              <line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text className="chart-axis-label" textAnchor="end" x={padding.left - 12} y={y + 4}>
                {formatAccuracyTick(tick)}
              </text>
            </g>
          );
        })
          : null}
        {rightValues.length > 0
          ? rightTicks.map((tick) => {
          const y = getY(tick, "right");
          return (
            <text className="chart-axis-label" key={`right-${tick}`} textAnchor="start" x={width - padding.right + 12} y={y + 4}>
              {formatMetricTick(tick, rightFamily)}
            </text>
          );
        })
          : null}
        {xTickIndexes.map((experimentIndex) => {
          const x = getX(experimentIndex);
          return (
            <g key={`x-${experimentIndex}`}>
              <line className="chart-grid chart-grid-vertical" x1={x} x2={x} y1={padding.top} y2={height - padding.bottom} />
              <text className="chart-axis-label" textAnchor="middle" x={x} y={height - padding.bottom + 24}>
                {`R${experimentIndex}`}
              </text>
            </g>
          );
        })}
        <line className="chart-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
        {leftValues.length > 0 ? (
          <line className="chart-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
        ) : null}
        {rightValues.length > 0 ? (
          <line className="chart-axis" x1={width - padding.right} x2={width - padding.right} y1={padding.top} y2={height - padding.bottom} />
        ) : null}
        {chartSeries.map((item) => (
          <g key={item.metricName}>
            {item.pathData ? (
              <path className="chart-line trend-chart-line" d={item.pathData} style={{ stroke: item.color }} />
            ) : null}
            {item.plottedPoints.map((point) => (
              <circle
                className="chart-dot trend-chart-dot"
                cx={point.x}
                cy={point.y}
                key={`${item.metricName}-${point.experiment_id}`}
                r={3.25}
                style={{ fill: item.color, stroke: item.color }}
              />
            ))}
          </g>
        ))}
        <text className="chart-axis-title" textAnchor="middle" x={width / 2} y={height - 10}>
          Experiment Round
        </text>
        {leftValues.length > 0 ? (
          <text className="chart-axis-title" textAnchor="middle" transform={`translate(18 ${height / 2}) rotate(-90)`}>
            Accuracy
          </text>
        ) : null}
        {rightValues.length > 0 ? (
          <text className="chart-axis-title" textAnchor="middle" transform={`translate(${width - 18} ${height / 2}) rotate(90)`}>
            {getRightAxisLabel(rightFamily)}
          </text>
        ) : null}
      </svg>
    </div>
  );
}

function EmptyStateCard({ copy }: { copy: string }) {
  return (
    <div className="empty-state-card">
      <p>{copy}</p>
    </div>
  );
}

type ProposalSnapshot = {
  hypothesis?: string;
  changes?: Record<string, unknown>;
  train_hyp_changes?: Record<string, unknown> | null;
  recipe_changes?: Record<string, unknown> | null;
};

type SummarySnapshot = {
  experiment_id?: string;
  summary?: string;
};

type RoundSnapshot = {
  round_index?: number;
  proposal?: ProposalSnapshot;
  result?: SummarySnapshot;
};

type AutoTrainSummarySnapshot = {
  rounds?: RoundSnapshot[];
  current_proposal?: ProposalSnapshot | null;
  final_proposal?: ProposalSnapshot | null;
  final_suggestion_error?: string | null;
  stop_reason?: string | null;
};

function readAutoTrainSummary(task: AutoTrainTask | null): AutoTrainSummarySnapshot | null {
  if (!task?.summary || typeof task.summary !== "object") {
    return null;
  }
  return task.summary as AutoTrainSummarySnapshot;
}

function getRecommendedCandidate(candidates: ModelCompareCandidateResult[]) {
  if (!candidates.length) {
    return null;
  }
  return [...candidates].sort((left, right) => {
    const accuracyDelta = (right.top1_acc ?? 0) - (left.top1_acc ?? 0);
    if (accuracyDelta !== 0) {
      return accuracyDelta;
    }
    return (left.latency_ms ?? Number.MAX_SAFE_INTEGER) - (right.latency_ms ?? Number.MAX_SAFE_INTEGER);
  })[0];
}

function resolveCurrentAutoTask(
  selectedTaskId: string | null,
  activeTask: AutoTrainTask | null,
  persistedTask: AutoTrainTask | null,
  activeTaskErrored: boolean
) {
  if (!selectedTaskId) {
    return null;
  }
  if (persistedTask && isAutoTrainTerminalStatus(persistedTask.status)) {
    return persistedTask;
  }
  if (activeTask?.task_id === selectedTaskId && !activeTaskErrored) {
    return activeTask;
  }
  return persistedTask ?? (activeTask?.task_id === selectedTaskId ? activeTask : null);
}

function resolveCurrentCompareTask(
  selectedTaskId: string | null,
  activeTask: ModelCompareTask | null,
  persistedTask: ModelCompareTask | null,
  activeTaskErrored: boolean
) {
  if (!selectedTaskId) {
    return null;
  }
  if (persistedTask && isCompareRecommendationReady(persistedTask)) {
    return persistedTask;
  }
  if (activeTask?.task_id === selectedTaskId && !activeTaskErrored) {
    return activeTask;
  }
  return persistedTask ?? (activeTask?.task_id === selectedTaskId ? activeTask : null);
}

function isCandidateTerminalStatus(status: string) {
  return ["success", "failed", "discarded"].includes(status);
}

function hasCompareCompletedByCandidates(task: ModelCompareTask | null) {
  const totalModels = task?.total_models ?? task?.candidate_models?.length ?? 0;
  if (!task || totalModels === 0) {
    return false;
  }
  const completedCandidateCount =
    task.summary?.candidate_results?.filter((candidate) => isCandidateTerminalStatus(candidate.status)).length ?? 0;
  return completedCandidateCount >= totalModels;
}

function isCompareTerminalStatus(status: string | null | undefined) {
  return ["success", "failed", "stopped"].includes(status ?? "");
}

function isAutoTrainTerminalStatus(status: string | null | undefined) {
  return ["stopped", "stopped_by_policy", "failed", "success"].includes(status ?? "");
}

function getCompareDisplayStatus(task: ModelCompareTask | null) {
  if (!task) {
    return "draft";
  }
  if (task.status === "stopping") {
    return "stopping";
  }
  if (isCompareTerminalStatus(task.status)) {
    return task.status;
  }
  if (hasCompareCompletedByCandidates(task)) {
    return "success";
  }
  return task.status;
}

function isCompareRecommendationReady(task: ModelCompareTask | null) {
  if (!task) {
    return false;
  }
  return isCompareTerminalStatus(task.status) || hasCompareCompletedByCandidates(task);
}

function buildCompareResultSummary(
  task: ModelCompareTask | null,
  bestCandidate: ModelCompareCandidateResult | null
) {
  const aiSummary = task?.summary?.ai_summary?.trim();
  if (aiSummary) {
    return aiSummary;
  }
  if (!task || !isCompareRecommendationReady(task)) {
    return null;
  }
  if (bestCandidate) {
    const modelLabel = MODEL_LABELS[bestCandidate.model_name as SupportedModelName] ?? bestCandidate.model_name;
    return `${modelLabel} currently leads the compare results with Top1 ${formatAccuracy(bestCandidate.top1_acc)} and ${formatLatency(bestCandidate.latency_ms)} latency.`;
  }
  return task.stop_reason ?? task.error ?? "No candidate finished successfully yet.";
}

function buildCompareSearchCandidate(
  candidate: ModelCompareCandidateResult,
  dataset: string
): CompareSearchCandidate | null {
  if (!candidate.run_id) {
    return null;
  }
  return {
    modelName: candidate.model_name as SupportedModelName,
    runId: candidate.run_id,
    dataset,
    top1Acc: candidate.top1_acc,
    latencyMs: candidate.latency_ms,
    parameterCountMillion: candidate.parameter_count_million
  };
}

function buildCompareCandidateMeta(candidate: CompareSearchCandidate) {
  const parts = [
    candidate.top1Acc != null ? formatAccuracy(candidate.top1Acc) : null,
    candidate.latencyMs != null ? formatLatency(candidate.latencyMs) : null,
    candidate.parameterCountMillion != null ? formatParameterCount(candidate.parameterCountMillion) : null
  ].filter((value): value is string => Boolean(value));
  return parts.length ? parts.join(" / ") : null;
}

function buildSearchTaskTitle(dataset: string, modelName: SupportedModelName) {
  return `Search - ${MODEL_LABELS[modelName] ?? modelName} - ${dataset}`;
}

function buildSearchMetaEntries(task: AutoTrainTask | null, elapsedSeconds: number) {
  const entries: Array<{
    label: string;
    value: string;
  }> = [];

  if (task?.current_round) {
    entries.push({
      label: "round",
      value: `Round ${task.current_round}`
    });
  }
  if (task) {
    entries.push({
      label: "elapsed",
      value: formatElapsed(elapsedSeconds)
    });
  }
  if ((task?.provider_total_tokens_total ?? 0) > 0) {
    entries.push({
      label: "tokens",
      value: `Tokens ${formatCompactInteger(task?.provider_total_tokens_total)}`
    });
  }
  if (!entries.length) {
    entries.push({
      label: "status",
      value: task ? formatStatusLabel(task.status) : "Waiting to start"
    });
  }
  return entries;
}

function buildCompareMetaEntries(
  task: ModelCompareTask | null,
  completedCandidateCount: number,
  totalModels: number,
  displayStatus: string
) {
  const entries: Array<{
    label: string;
    value: string;
  }> = [
    {
      label: "status",
      value: `Status ${task ? formatStatusLabel(displayStatus) : "Waiting to start"}`
    }
  ];

  if (task) {
    entries.push({
      label: "progress",
      value: `${completedCandidateCount}/${totalModels || 0}`
    });
    entries.push({
      label: "elapsed",
      value: formatElapsed(task.elapsed_seconds)
    });
    if (task.current_model_name && !isCompareTerminalStatus(displayStatus)) {
      entries.push({
        label: "current_model",
        value: `Current ${MODEL_LABELS[task.current_model_name as SupportedModelName] ?? task.current_model_name}`
      });
    }
  }

  return entries;
}

function getActiveSearchProposal(summary: AutoTrainSummarySnapshot | null) {
  if (summary?.current_proposal) {
    return summary.current_proposal;
  }
  const latestRound = [...(summary?.rounds ?? [])].reverse()[0] ?? null;
  return latestRound?.proposal ?? null;
}

function getSearchSuggestionText(
  task: AutoTrainTask | null,
  summary: AutoTrainSummarySnapshot | null,
  activeProposal: ProposalSnapshot | null
) {
  const activeHypothesis = activeProposal?.hypothesis?.trim();
  if (activeHypothesis) {
    return activeHypothesis;
  }
  if (task && isAutoTrainTerminalStatus(task.status)) {
    return summary?.final_proposal?.hypothesis?.trim() ?? null;
  }
  return null;
}

function buildSearchTerminalSummary(
  task: AutoTrainTask | null,
  summary: AutoTrainSummarySnapshot | null,
  latestRound: RoundSnapshot | null
) {
  if (!task || !isAutoTrainTerminalStatus(task.status)) {
    return null;
  }
  const title =
    (task.status === "failed" ? "Search failed" : null) ??
    summary?.final_proposal?.hypothesis?.trim() ??
    latestRound?.proposal?.hypothesis?.trim() ??
    "Search finished";
  const body =
    task.stop_reason ??
    task.error ??
    latestRound?.result?.summary?.trim() ??
    summary?.final_suggestion_error?.trim() ??
    "Search finished.";
  return { title, body };
}

const PROPOSAL_CHANGE_LABELS: Record<string, string> = {
  optimizer: "Optimizer",
  learning_rate: "LR",
  batch_size: "Batch Size",
  image_size: "Image Size",
  epochs: "Epochs",
  weight_decay: "Weight Decay",
  scheduler: "Scheduler",
  augmentation_policy: "Augmentation",
  mixup_alpha: "Mixup",
  cutmix_alpha: "CutMix",
  random_erasing_prob: "Random Erasing",
  loss_name: "Loss",
  focal_gamma: "Focal Gamma",
  label_smoothing: "Label Smoothing",
  aux_logits: "Aux Logits",
  width_multiple: "Width",
  pooling_type: "Pooling",
  classifier_dropout: "Dropout",
  backbone_name: "Backbone",
  neck_name: "Neck",
  head_name: "Head"
};

function describeProposalChangeSummary(proposal: ProposalSnapshot | null) {
  if (!proposal) {
    return null;
  }
  const changedFields = Object.entries(proposal.changes ?? {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([fieldName]) => PROPOSAL_CHANGE_LABELS[fieldName] ?? fieldName.replace(/_/g, " "));
  if (!changedFields.length) {
    return null;
  }
  return changedFields.join(" / ");
}

function normalizeCandidateModels(candidateModels: string[] | undefined): SupportedModelName[] {
  if (!candidateModels?.length) {
    return [...COMPARE_CANDIDATE_MODELS];
  }
  return candidateModels.filter((modelName): modelName is SupportedModelName => modelName in MODEL_LABELS);
}

function formatStatusLabel(status: string) {
  return status.replace(/_/g, " ");
}

function normalizeStatusTone(status: string) {
  if (["failed", "discarded"].includes(status)) {
    return "danger";
  }
  if (["running", "queued", "stopping"].includes(status)) {
    return "active";
  }
  if (["success", "stopped", "stopped_by_policy"].includes(status)) {
    return "success";
  }
  return "neutral";
}

function formatElapsed(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0s";
  }
  if (value < 60) {
    return `${Math.round(value)}s`;
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

function formatCompactInteger(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value) || value <= 0) {
    return "0";
  }
  return new Intl.NumberFormat("en", {
    notation: value >= 1000 ? "compact" : "standard",
    maximumFractionDigits: value >= 1000 ? 1 : 0
  }).format(value);
}

function formatLatencyTick(value: number) {
  if (value >= 100) {
    return `${Math.round(value)} ms`;
  }
  if (value >= 10) {
    return `${value.toFixed(1)} ms`;
  }
  return `${value.toFixed(2)} ms`;
}

function formatAccuracyTick(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatLossTick(value: number) {
  if (value >= 10) {
    return value.toFixed(1);
  }
  if (value >= 1) {
    return value.toFixed(2);
  }
  return value.toFixed(3);
}

function formatMetricTick(value: number, family: "loss" | "latency" | "epoch") {
  if (family === "latency") {
    return formatLatencyTick(value);
  }
  if (family === "epoch") {
    return `${Math.round(value)}`;
  }
  return formatLossTick(value);
}

function getRightAxisLabel(family: "loss" | "latency" | "epoch") {
  if (family === "latency") {
    return "Latency (ms)";
  }
  if (family === "epoch") {
    return "Best Epoch";
  }
  return "Loss";
}

function sampleTickIndexes(values: number[], limit: number) {
  if (values.length <= limit) {
    return values;
  }
  const step = (values.length - 1) / Math.max(limit - 1, 1);
  return Array.from({ length: limit }, (_, index) => values[Math.round(index * step)]);
}

function buildYAxisTicks(minValue: number, maxValue: number, count: number) {
  if (count <= 1) {
    return [minValue];
  }
  const step = (maxValue - minValue) / (count - 1);
  return Array.from({ length: count }, (_, index) => minValue + step * index);
}

function buildMetricDomain(values: number[], kind: "accuracy" | "loss" | "latency" | "epoch") {
  if (!values.length) {
    return kind === "accuracy" ? { min: 0, max: 1 } : { min: 0, max: 1 };
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (kind === "accuracy") {
    const padding = Math.max((maxValue - minValue) * 0.18, 0.01);
    return {
      min: Math.max(0, minValue - padding),
      max: Math.min(1, maxValue + padding)
    };
  }
  if (kind === "epoch") {
    const padding = Math.max((maxValue - minValue) * 0.18, 1);
    return {
      min: Math.max(0, minValue - padding),
      max: maxValue + padding
    };
  }
  const padding = Math.max((maxValue - minValue) * 0.12, maxValue * 0.08, 0.02);
  return {
    min: Math.max(0, minValue - padding),
    max: maxValue + padding
  };
}

function getComparePointColor(modelName: string) {
  switch (modelName) {
    case "mobilenet_v2":
      return "#ff8a00";
    case "mobilenet_v3_small":
      return "#2b59ff";
    case "googlenet":
      return "#11a36a";
    case "resnet18":
      return "#d9485f";
    default:
      return "#5b6475";
  }
}

function formatAccuracy(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(2)}%`;
  }
  return value.toFixed(3);
}

function formatAccuracyDetailed(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(3)}%`;
  }
  return value.toFixed(4);
}

function formatLatency(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(2)} ms`;
}

function formatParameterCount(value: number | null | undefined) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  return `${value.toFixed(2)} M`;
}

function truncateId(value: string | null | undefined) {
  if (!value) {
    return "—";
  }
  return value.length > 10 ? `${value.slice(0, 10)}…` : value;
}

function toTaskType(mode: WorkspaceMode): TaskType {
  return mode === "compare" ? "model_compare" : "auto_train";
}

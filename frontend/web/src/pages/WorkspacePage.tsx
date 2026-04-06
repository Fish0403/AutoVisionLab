import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
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
  useRunTrend,
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
  getDatasetBaselineImageSize,
  getModelFamily,
  getPreferredDatasetName,
  MODEL_LABELS,
  type SupportedModelName,
  type TrainingFormValues
} from "../features/training/config";
import { readLocalStorage, writeLocalStorage } from "../lib/storage";
import type {
  AutoTrainTask,
  ModelCompareCandidateResult,
  ModelCompareTask,
  RunTrendPayload,
} from "../types/domain";

const SELECTED_TASK_ID_STORAGE_KEY = "autovisionlab:selected-task-id";
const SELECTED_TASK_TYPE_STORAGE_KEY = "autovisionlab:selected-task-type";
const COMPARE_ANCHOR_MODEL: SupportedModelName = "mobilenet_v3_small";
const DEFAULT_COMPARE_MODEL_COUNT = 3;
const MODEL_FAMILY_ORDER = ["mobilenet", "efficientnet", "resnet", "googlenet"] as const;
const MODEL_FAMILY_LABELS: Record<(typeof MODEL_FAMILY_ORDER)[number], string> = {
  mobilenet: "MobileNet",
  efficientnet: "EfficientNet",
  resnet: "ResNet",
  googlenet: "GoogLeNet"
};
const DEMO_TOGGLE_TOOLTIP = "Use a smaller deterministic subset for quicker local testing.";
const EPOCH_SEARCH_TOOLTIP = "Allow Auto Train to adjust epochs as part of the search.";
const SEARCH_TREND_METRICS = [
  { metricName: "top1_acc", label: "Top1 Acc", color: "#2b59ff", axis: "left" as const, family: "accuracy" as const, defaultVisible: true },
  { metricName: "val_loss", label: "Val Loss", color: "#d9485f", axis: "right" as const, family: "loss" as const, defaultVisible: true },
  { metricName: "train_loss", label: "Train Loss", color: "#ff8a00", axis: "right" as const, family: "loss" as const, defaultVisible: true },
  { metricName: "latency_ms", label: "Latency", color: "#11a36a", axis: "right" as const, family: "latency" as const, defaultVisible: false },
  { metricName: "best_epoch", label: "Best Epoch", color: "#7a5af8", axis: "right" as const, family: "epoch" as const, defaultVisible: false }
];

type WorkspaceMode = "compare" | "search";
type TaskType = "auto_train" | "model_compare";
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
  const defaultDataset = getPreferredDatasetName(datasets, "neu");
  const areDatasetsReady = !datasetsQuery.isPending && !datasetsQuery.isError && datasets.length > 0;

  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(
    () => requestedTaskId ?? readLocalStorage(SELECTED_TASK_ID_STORAGE_KEY)
  );
  const [selectedTaskType, setSelectedTaskType] = useState<TaskType | null>(
    () => requestedTaskType ?? (readLocalStorage(SELECTED_TASK_TYPE_STORAGE_KEY) as TaskType | null)
  );
  const [mode, setMode] = useState<WorkspaceMode>(() => (requestedTaskType === "model_compare" ? "compare" : "search"));
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [isSearchLaunchPending, setIsSearchLaunchPending] = useState(false);
  const [showAllCompareModels, setShowAllCompareModels] = useState(false);
  const [draftTaskTitle, setDraftTaskTitle] = useState("Untitled task");
  const [formValues, setFormValues] = useState<TrainingFormValues>(() => defaultFormValues(defaultDataset));
  const [selectedCompareCandidate, setSelectedCompareCandidate] = useState<CompareSearchCandidate | null>(null);
  const selectedDatasetOption = datasets.find((dataset) => dataset.name === formValues.dataset) ?? null;
  const selectedDatasetOptionLabel = selectedDatasetOption?.name ?? formValues.dataset;
  const hasPersistedTaskContext = Boolean(requestedTaskId || selectedTaskId);
  const isDatasetSelectorReady = areDatasetsReady || hasPersistedTaskContext;
  const selectedModelFamily = getModelFamily(formValues.modelName);
  const availableSearchModels = MODEL_FAMILY_ORDER.flatMap((familyName) =>
    (Object.keys(MODEL_LABELS) as SupportedModelName[]).filter((modelName) => getModelFamily(modelName) === familyName)
  );
  const visibleSearchModels = availableSearchModels.filter((modelName) => getModelFamily(modelName) === selectedModelFamily);
  const visibleCompareModels = showAllCompareModels
    ? COMPARE_CANDIDATE_MODELS
    : COMPARE_CANDIDATE_MODELS.slice(0, DEFAULT_COMPARE_MODEL_COUNT);
  const areAllCompareModelsSelected = formValues.compareCandidateModels.length === COMPARE_CANDIDATE_MODELS.length;

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
      setIsEditingTitle(false);
      setIsSearchLaunchPending(false);
      setDraftTaskTitle("Untitled task");
      setMode("compare");
      setFormValues(defaultFormValues(defaultDataset));
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
  }, [defaultDataset, requestedNewTask, requestedTaskId, requestedTaskType]);

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
      return {
        ...previous,
        dataset: nextDataset
      };
    });
  }, [datasets, defaultDataset]);

  useEffect(() => {
    if (!selectedTaskId || !currentAutoTask) {
      return;
    }
    setSelectedCompareCandidate(null);
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
    setDraftTaskTitle(currentCompareTask.title ?? buildTaskTitle("compare", currentCompareTask.dataset ?? defaultDataset));
    setFormValues((previous) => ({
      ...previous,
      dataset: currentCompareTask.dataset ?? previous.dataset,
      compareCandidateModels: normalizeCandidateModels(currentCompareTask.candidate_models)
    }));
  }, [currentCompareTask, defaultDataset, selectedTaskId]);

  const isSearchDraftFromCompare = Boolean(mode === "search" && selectedTaskType === "model_compare" && !currentAutoTask);
  const controlMode: WorkspaceMode = mode;
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
  const compareResultSummary = buildCompareResultSummary(compareTask);
  const compareResultSummarySource = buildCompareResultSummarySource(compareTask);
  const baselineImageSize = getDatasetBaselineImageSize(datasets, formValues.dataset);

  const isDetachedSearchDraft = Boolean(
    currentAutoTask &&
      isAutoTrainTerminalStatus(currentAutoTask.status) &&
      ((currentAutoTask.model_name && formValues.modelName !== currentAutoTask.model_name) ||
        (currentAutoTask.dataset && formValues.dataset !== currentAutoTask.dataset))
  );
  const displayAutoTask = isDetachedSearchDraft || isSearchLaunchPending ? null : currentAutoTask;
  const autoTrainSummary = readAutoTrainSummary(displayAutoTask);
  const searchResultSummary = buildSearchResultSummary(autoTrainSummary);
  const searchResultSummarySource = buildSearchResultSummarySource(displayAutoTask, autoTrainSummary);
  const searchSummaryLoadingText = buildSearchSummaryLoadingText(displayAutoTask, searchResultSummary);

  const currentTaskTitle =
    (mode === "compare" ? currentCompareTask?.title : displayAutoTask?.title) ??
    (mode === "search" && isSearchLaunchPending ? draftTaskTitle : null) ??
    (isDetachedSearchDraft ? buildSearchTaskTitle(formValues.dataset, formValues.modelName) : null) ??
    (selectedTaskId || selectedCompareCandidate ? draftTaskTitle : "Untitled task");
  const currentTaskStatus =
    (mode === "compare" ? compareDisplayStatus : displayAutoTask?.status) ??
    (mode === "search" && isSearchLaunchPending ? "running" : "draft");
  const compareSourceTaskId = displayAutoTask?.source_task_id ?? (isSearchDraftFromCompare ? selectedTaskId : null);
  const hasCompareBackLink = Boolean(
    controlMode === "search" && compareSourceTaskId && (displayAutoTask?.source_task_type === "model_compare" || isSearchDraftFromCompare)
  );
  const useModeSwitchBackToCompare = Boolean(controlMode === "search" && isSearchDraftFromCompare);

  const isCompareRunning = Boolean(compareTask && !isCompareTerminalStatus(compareDisplayStatus));
  const isSearchRunning = Boolean(displayAutoTask && !["stopped", "stopped_by_policy", "failed"].includes(displayAutoTask.status));
  const isPrimaryActionBusy =
    startAutoTrainMutation.isPending || startModelCompareMutation.isPending || (controlMode === "compare" ? isCompareRunning : isSearchRunning);
  const controlErrorMessage =
    datasetsQuery.error?.message ??
    parameterSpaceQuery.error?.message ??
    startAutoTrainMutation.error?.message ??
    startModelCompareMutation.error?.message ??
    stopAutoTrainMutation.error?.message ??
    stopModelCompareMutation.error?.message ??
    null;

  const currentSearchRunId = displayAutoTask?.run_id ?? null;
  const runDetailQuery = useRunDetail(currentSearchRunId);
  const runTrendQuery = useRunTrend(currentSearchRunId);
  const runDetail = runDetailQuery.data ?? null;

  const searchTrendSeries = useMemo(
    () =>
      buildTrendSeriesFromPayload(runTrendQuery.data),
    [runTrendQuery.data]
  );
  const persistedSearchTrendSeries = useMemo(() => buildPersistedSearchTrendSeries(autoTrainSummary), [autoTrainSummary]);
  const displayedSearchTrendSeries = searchTrendSeries.length > 0 ? searchTrendSeries : persistedSearchTrendSeries;

  const handleModeSelect = (nextMode: WorkspaceMode) => {
    setMode(nextMode);
    if (selectedTaskType && selectedTaskType !== toTaskType(nextMode) && selectedTaskType !== "model_compare") {
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
      const fallbackCompareTitle = buildTaskTitle("compare", formValues.dataset);
      const normalizedCurrentTitle = currentTaskTitle?.trim();
      const generatedTitle =
        selectedTaskId && normalizedCurrentTitle && normalizedCurrentTitle !== "Untitled task"
          ? normalizedCurrentTitle
          : fallbackCompareTitle;
      setDraftTaskTitle(generatedTitle);
      const response = await startModelCompareMutation.mutateAsync({
        title: generatedTitle,
        dataset: formValues.dataset,
        candidate_models: formValues.compareCandidateModels,
        config: buildExperimentConfig(formValues, baselineImageSize, parameterSpace.version, COMPARE_ANCHOR_MODEL)
      });
      await queryClient.invalidateQueries({ queryKey: ["task-history"] });
      selectTask(response.task_id, "model_compare");
      return;
    }

    const generatedTitle =
      displayAutoTask?.task_id && !isDetachedSearchDraft ? currentTaskTitle : buildSearchTaskTitle(formValues.dataset, formValues.modelName);
    setDraftTaskTitle(generatedTitle);
    const isSearchFromCompareContext = Boolean(selectedTaskType === "model_compare" && !displayAutoTask);
    const shouldStartFreshSearchRun = Boolean(isSearchFromCompareContext || isDetachedSearchDraft);
    setIsSearchLaunchPending(true);
    try {
      const response = await startAutoTrainMutation.mutateAsync({
        title: generatedTitle,
        run_id: shouldStartFreshSearchRun ? null : (displayAutoTask?.run_id ?? null),
        run_name: runDetail?.name ?? buildRunName(formValues.dataset, formValues.modelName),
        dataset: formValues.dataset,
        model_name: formValues.modelName,
        source_task_type: isSearchFromCompareContext ? "model_compare" : null,
        source_task_id: isSearchFromCompareContext ? selectedTaskId : null,
        source_task_title: isSearchFromCompareContext ? currentCompareTask?.title ?? null : null,
        source_model_name: isSearchFromCompareContext ? selectedCompareCandidate?.modelName ?? formValues.modelName : null,
        config: buildExperimentConfig(formValues, baselineImageSize, parameterSpace.version, undefined, {
          allowEpochSearch: formValues.allowEpochSearch
        }),
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
    setMode("search");
    setIsEditingTitle(false);
    setIsSearchLaunchPending(false);
    setDraftTaskTitle(buildTaskTitle("search", candidate.dataset, candidate.modelName));
    setFormValues((previous) => ({
      ...previous,
      dataset: candidate.dataset,
      modelName: candidate.modelName
    }));
  };

  function selectTask(taskId: string, taskType: TaskType) {
    setSelectedCompareCandidate(null);
    setIsSearchLaunchPending(false);
    setSelectedTaskId(taskId);
    setSelectedTaskType(taskType);
    setMode(taskType === "model_compare" ? "compare" : "search");
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
              <FormField
                label="Dataset"
                labelAccessory={
                  <span className="field-label-accessory">
                    <label className="inline-checkbox inline-checkbox-compact" title={DEMO_TOGGLE_TOOLTIP}>
                      <input
                        aria-label={DEMO_TOGGLE_TOOLTIP}
                        checked={formValues.useDemoMode}
                        onChange={(event) =>
                          setFormValues((previous) => ({
                            ...previous,
                            useDemoMode: event.target.checked
                          }))
                        }
                        type="checkbox"
                      />
                      <span>Demo</span>
                    </label>
                  </span>
                }
              >
                <select
                  disabled={!isDatasetSelectorReady}
                  value={formValues.dataset}
                  onChange={(event) => {
                    const nextDataset = event.target.value;
                    setFormValues((previous) => ({
                      ...previous,
                      dataset: nextDataset
                    }));
                  }}
                >
                  {!areDatasetsReady && !hasPersistedTaskContext ? (
                    <option value={formValues.dataset}>
                      {datasetsQuery.isPending
                        ? "Loading datasets..."
                        : datasetsQuery.isError
                          ? "Failed to load datasets"
                          : "No datasets found"}
                    </option>
                  ) : null}
                  {!areDatasetsReady && hasPersistedTaskContext ? (
                    <option value={formValues.dataset}>{formValues.dataset}</option>
                  ) : null}
                  {areDatasetsReady ? <option value={formValues.dataset}>{selectedDatasetOptionLabel}</option> : null}
                  {datasets.filter((dataset) => dataset.name !== selectedDatasetOptionLabel).map((dataset) => (
                    <option key={dataset.name} value={dataset.name}>
                      {dataset.name}
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
            {controlMode === "search" ? (
              <>
                {hasCompareBackLink ? (
                  <div className="form-field">
                    <div className="search-model-link-row">
                      <span className="workspace-label">Model</span>
                      <div className="search-model-link-actions">
                        <label className="inline-checkbox inline-checkbox-compact" title={EPOCH_SEARCH_TOOLTIP}>
                          <input
                            aria-label={EPOCH_SEARCH_TOOLTIP}
                            checked={formValues.allowEpochSearch}
                            onChange={(event) =>
                              setFormValues((previous) => ({
                                ...previous,
                                allowEpochSearch: event.target.checked
                              }))
                            }
                            type="checkbox"
                          />
                          <span>Search epochs</span>
                        </label>
                        {useModeSwitchBackToCompare ? (
                          <button
                            className="text-button"
                            onClick={() => handleModeSelect("compare")}
                            type="button"
                          >
                            Back to Compare
                          </button>
                        ) : (
                          <Link
                            className="text-button"
                            to={`/workspace?taskId=${compareSourceTaskId}&taskType=model_compare`}
                          >
                            Back to Compare
                          </Link>
                        )}
                      </div>
                    </div>
                    <div className="model-select-grid">
                      <select
                        aria-label="Model family"
                        value={selectedModelFamily}
                        onChange={(event) => {
                          const nextFamily = event.target.value as (typeof MODEL_FAMILY_ORDER)[number];
                          const nextModelName = availableSearchModels.find((modelName) => getModelFamily(modelName) === nextFamily);
                          if (!nextModelName) {
                            return;
                          }
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: nextModelName
                          }));
                        }}
                      >
                        {MODEL_FAMILY_ORDER.map((familyName) => (
                          <option key={familyName} value={familyName}>
                            {MODEL_FAMILY_LABELS[familyName]}
                          </option>
                        ))}
                      </select>
                      <select
                        aria-label="Model"
                        value={formValues.modelName}
                        onChange={(event) =>
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: event.target.value as SupportedModelName
                          }))
                        }
                      >
                        {visibleSearchModels.map((modelName) => (
                          <option key={modelName} value={modelName}>
                            {MODEL_LABELS[modelName]}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                ) : (
                  <FormField
                    label="Model"
                    labelAccessory={
                      <span className="field-label-accessory">
                        <label className="inline-checkbox inline-checkbox-compact" title={EPOCH_SEARCH_TOOLTIP}>
                          <input
                            aria-label={EPOCH_SEARCH_TOOLTIP}
                            checked={formValues.allowEpochSearch}
                            onChange={(event) =>
                              setFormValues((previous) => ({
                                ...previous,
                                allowEpochSearch: event.target.checked
                              }))
                            }
                            type="checkbox"
                          />
                          <span>Search epochs</span>
                        </label>
                      </span>
                    }
                  >
                    <div className="model-select-grid">
                      <select
                        aria-label="Model family"
                        value={selectedModelFamily}
                        onChange={(event) => {
                          const nextFamily = event.target.value as (typeof MODEL_FAMILY_ORDER)[number];
                          const nextModelName = availableSearchModels.find((modelName) => getModelFamily(modelName) === nextFamily);
                          if (!nextModelName) {
                            return;
                          }
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: nextModelName
                          }));
                        }}
                      >
                        {MODEL_FAMILY_ORDER.map((familyName) => (
                          <option key={familyName} value={familyName}>
                            {MODEL_FAMILY_LABELS[familyName]}
                          </option>
                        ))}
                      </select>
                      <select
                        aria-label="Model"
                        value={formValues.modelName}
                        onChange={(event) =>
                          setFormValues((previous) => ({
                            ...previous,
                            modelName: event.target.value as SupportedModelName
                          }))
                        }
                      >
                        {visibleSearchModels.map((modelName) => (
                          <option key={modelName} value={modelName}>
                            {MODEL_LABELS[modelName]}
                          </option>
                        ))}
                      </select>
                    </div>
                  </FormField>
                )}
              </>
            ) : null}
            {controlMode === "compare" ? (
              <div className="popover-panel mode-options-panel">
                <div className="section-heading">
                  <h3>Models</h3>
                  <div className="mode-options-actions">
                    <button
                      className="text-button mode-options-link"
                      onClick={() =>
                        setFormValues((previous) => ({
                          ...previous,
                          compareCandidateModels: areAllCompareModelsSelected ? [] : [...COMPARE_CANDIDATE_MODELS]
                        }))
                      }
                      type="button"
                    >
                      {areAllCompareModelsSelected ? "Reset" : "All"}
                    </button>
                    {COMPARE_CANDIDATE_MODELS.length > DEFAULT_COMPARE_MODEL_COUNT ? (
                      <button
                        className="text-button mode-options-link"
                        onClick={() => setShowAllCompareModels((previous) => !previous)}
                        type="button"
                      >
                        {showAllCompareModels ? "Less" : "More"}
                      </button>
                    ) : null}
                  </div>
                </div>
                <div className="option-stack">
                  {visibleCompareModels.map((modelName) => {
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
          </div>

          <div className="control-actions">
            <button
              className="primary-button"
              disabled={
                isPrimaryActionBusy ||
                !areDatasetsReady ||
                !parameterSpaceQuery.data ||
                (controlMode === "compare" && formValues.compareCandidateModels.length === 0)
              }
              onClick={() => void handleRun()}
              type="button"
            >
              {isPrimaryActionBusy ? "Running..." : controlMode === "compare" ? "Run Compare" : "Start Search"}
            </button>
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
          </div>
        </aside>

        <section className="display-panel">
          <div className="progress-panel progress-panel-plain">
            {mode === "compare" ? (
              <CompareProgressCard
                task={compareTask}
                completedCandidateCount={completedCandidateCount}
                displayStatus={compareDisplayStatus}
                externalErrorMessage={controlErrorMessage}
              />
            ) : (
              <SearchProgressCard task={displayAutoTask} externalErrorMessage={controlErrorMessage} />
            )}
          </div>

          <div className="display-header">
            <div>
              <h2>Results</h2>
            </div>
          </div>

          {mode === "compare" ? (
            <div className="display-stack">
              {compareResultSummary ? (
                <div className="result-summary-row">
                  {compareResultSummarySource ? <span className="result-summary-source">{compareResultSummarySource}</span> : null}
                  <p className="result-summary-copy">{compareResultSummary}</p>
                </div>
              ) : buildCompareSummaryLoadingText(compareTask) ? (
                <div className="result-summary-row result-summary-row-loading">
                  <span className="result-summary-source">{compareResultSummarySource ?? "Summary"}</span>
                  <p className="result-summary-copy result-summary-copy-loading">{buildCompareSummaryLoadingText(compareTask)}</p>
                </div>
              ) : null}
              <CompareScatterChart
                candidates={successfulCandidates}
                dataset={compareTask?.dataset ?? formValues.dataset}
                onSelectCandidate={(candidate) => prepareSearchFromCompareCandidate(candidate)}
                selectedCandidateModelName={selectedCompareCandidate?.modelName ?? null}
              />
            </div>
          ) : (
            <div className="display-stack">
              {searchResultSummary ? (
                <div className="result-summary-row">
                  {searchResultSummarySource ? <span className="result-summary-source">{searchResultSummarySource}</span> : null}
                  <p className="result-summary-copy">{searchResultSummary}</p>
                </div>
              ) : searchSummaryLoadingText ? (
                <div className="result-summary-row result-summary-row-loading">
                  <span className="result-summary-source">{searchResultSummarySource ?? "Summary"}</span>
                  <p className="result-summary-copy result-summary-copy-loading">{searchSummaryLoadingText}</p>
                </div>
              ) : null}
              <MetricTrendChart
                series={displayedSearchTrendSeries}
                showBestPathToggle={Boolean(displayAutoTask && isAutoTrainTerminalStatus(displayAutoTask.status))}
                summary={autoTrainSummary}
              />
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
  displayStatus,
  externalErrorMessage
}: {
  task: ModelCompareTask | null;
  completedCandidateCount: number;
  displayStatus: string;
  externalErrorMessage?: string | null;
}) {
  const totalModels = task?.total_models ?? task?.candidate_models?.length ?? 0;
  const progress = totalModels > 0 ? Math.min(100, Math.round((completedCandidateCount / totalModels) * 100)) : 0;
  const statusTone = task ? normalizeStatusTone(displayStatus) : "neutral";
  const footerText = buildTaskProgressFooter(task);
  const failureNotice = buildTaskFailureNotice(task, externalErrorMessage);

  return (
      <div>
      <div className="progress-activity-block">
        <div className="progress-activity-header">
          <h2>Status</h2>
        </div>
        <div className="progress-activity-divider" />
      </div>
      {task ? (
        <>
          <div className="compare-progress-row">
            <div className="compare-status-block">
              <span className={`compare-status-dot compare-status-dot-${statusTone}`} />
              <span className="compare-status-text">{formatStatusLabel(displayStatus)}</span>
            </div>
            <div className="compare-meta-inline">
              <span>{completedCandidateCount}/{totalModels || 0}</span>
              <span>{formatElapsed(task.elapsed_seconds)}</span>
            </div>
          </div>
        </>
      ) : null}
      {!task ? <div className="progress-empty-space" /> : null}
      {task ? (
        <>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          {footerText ? <p className="progress-footer-copy">{footerText}</p> : null}
          {failureNotice ? (
            <p className="progress-footer-copy progress-footer-copy-error">
              <ErrorNoticeIcon />
              <span>{failureNotice}</span>
            </p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function SearchProgressCard({ task, externalErrorMessage }: { task: AutoTrainTask | null; externalErrorMessage?: string | null }) {
  const summary = readAutoTrainSummary(task);
  const activeProposal = task && !isAutoTrainTerminalStatus(task.status) ? summary?.current_proposal ?? null : getActiveSearchProposal(summary);
  const suggestionText = getSearchSuggestionText(task, summary, activeProposal);
  const validatedChangeSummary = describeValidatedProposalChanges(activeProposal);
  const datasetSummaryText = buildTaskDatasetSummary(task);
  const startupMessage = buildTaskStartupMessage(task);
  const primaryStatusText = datasetSummaryText ?? startupMessage;
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

  const validatedChangeHeadline = validatedChangeSummary ? `Will apply ${validatedChangeSummary}` : null;
  const metaInlineText = buildSearchMetaInlineText(task, elapsedDisplay);
  const runLabel = buildSearchRunLabel(task);
  const gpuMemoryLabel = buildSearchGpuMemoryLabel(summary);
  const warningNotice = buildTaskWarningNotice(task);
  const failureNotice = buildTaskFailureNotice(task, externalErrorMessage);
  const roundFailureNotice = buildLatestExperimentFailureNotice(task);
  const hasInsightRows = Boolean(primaryStatusText || suggestionText || validatedChangeHeadline);

  return (
    <div>
      <div className="progress-activity-block">
        <div className="progress-activity-header">
          <h2>Status</h2>
          {runLabel ? <div className="search-meta-inline">{runLabel}</div> : null}
        </div>
        <div className="progress-activity-divider" />
        {metaInlineText || gpuMemoryLabel ? (
          <div className="search-meta-inline">
            {[metaInlineText, gpuMemoryLabel].filter(Boolean).join(" · ")}
          </div>
        ) : null}
      </div>
      {hasInsightRows ? (
        <div className="progress-insight-group">
          {primaryStatusText ? (
            <div className="progress-inline-note">
              {datasetSummaryText ? <DatasetSummaryIcon /> : <SearchFocusIcon />}
              <span>{primaryStatusText}</span>
            </div>
          ) : null}
          {suggestionText ? (
            <div className="progress-suggestion-row">
              <SuggestionIcon />
              <p className="progress-suggestion-copy">{suggestionText}</p>
            </div>
          ) : null}
          {validatedChangeHeadline ? (
            <div className="progress-inline-note">
              <SearchFocusIcon />
              <span>{validatedChangeHeadline}</span>
            </div>
          ) : null}
        </div>
      ) : null}
      {warningNotice ? (
        <p className="progress-footer-copy progress-footer-copy-warning">
          <WarningNoticeIcon />
          <span>{warningNotice}</span>
        </p>
      ) : null}
      {failureNotice ? (
        <p className="progress-footer-copy progress-footer-copy-error">
          <ErrorNoticeIcon />
          <span>{failureNotice}</span>
        </p>
      ) : null}
      {!failureNotice && roundFailureNotice ? (
        <p className="progress-footer-copy progress-footer-copy-error">
          <ErrorNoticeIcon />
          <span>{roundFailureNotice}</span>
        </p>
      ) : null}
      {!task ? <div className="progress-empty-space" /> : null}
    </div>
  );
}

function DatasetSummaryIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon" viewBox="0 0 16 16">
      <path
        d="M3.25 4.25h9.5v7.5h-9.5z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
      <path
        d="M5.5 7.25h5M5.5 9.5h2.75"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeWidth="1.5"
      />
    </svg>
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

function ErrorNoticeIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon" viewBox="0 0 16 16">
      <circle cx="8" cy="8" fill="none" r="6.25" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 4.75v4.25M8 11.5h.01" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
    </svg>
  );
}

function WarningNoticeIcon() {
  return (
    <svg aria-hidden="true" className="progress-inline-icon" viewBox="0 0 16 16">
      <path
        d="M8 2.1 14 13.2c.2.36-.06.8-.47.8H2.47c-.41 0-.67-.44-.47-.8L8 2.1Z"
        fill="none"
        stroke="currentColor"
        strokeLinejoin="round"
        strokeWidth="1.5"
      />
      <path d="M8 6v3.7M8 11.9h.01" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="1.5" />
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
    const label = MODEL_LABELS[candidate.model_name as SupportedModelName] ?? candidate.model_name;
    return {
      candidate,
      label,
      x,
      y
    };
  });
  const labelPlacements = buildScatterLabelPlacements(points, width, height, padding);
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
            <text
              className="chart-label"
              textAnchor={labelPlacements[point.candidate.model_name]?.textAnchor ?? "start"}
              x={labelPlacements[point.candidate.model_name]?.x ?? point.x + 12}
              y={labelPlacements[point.candidate.model_name]?.y ?? point.y - 10}
            >
              {point.label}
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

type ScatterLabelPlacement = {
  textAnchor: "start" | "middle" | "end";
  x: number;
  y: number;
};

function buildScatterLabelPlacements(
  points: Array<{ candidate: ModelCompareCandidateResult; label: string; x: number; y: number }>,
  width: number,
  height: number,
  padding: number
): Record<string, ScatterLabelPlacement> {
  const fontSize = 10;
  const lineHeight = 12;
  const characterWidth = 6.4;
  const pointRadius = 9;
  const horizontalGap = 7;
  const verticalGap = 6;
  const overlapPadding = 4;
  const pointCollisionPadding = 5;
  const bounds = {
    left: padding + 4,
    right: width - padding - 4,
    top: padding + lineHeight,
    bottom: height - padding - 4
  };
  const occupiedRects: Array<{ left: number; right: number; top: number; bottom: number }> = [];
  const placements: Record<string, ScatterLabelPlacement> = {};
  const sortedPoints = [...points].sort((left, right) => left.y - right.y || left.x - right.x);
  const pointRects = points.map((point) => ({
    modelName: point.candidate.model_name,
    rect: {
      left: point.x - pointRadius - pointCollisionPadding,
      right: point.x + pointRadius + pointCollisionPadding,
      top: point.y - pointRadius - pointCollisionPadding,
      bottom: point.y + pointRadius + pointCollisionPadding
    }
  }));

  for (const point of sortedPoints) {
    const labelWidth = Math.max(point.label.length * characterWidth, fontSize * 4);
    const candidates = [
      { textAnchor: "start" as const, x: point.x + pointRadius + horizontalGap, y: point.y - verticalGap },
      { textAnchor: "end" as const, x: point.x - pointRadius - horizontalGap, y: point.y - verticalGap },
      { textAnchor: "middle" as const, x: point.x, y: point.y - pointRadius - verticalGap },
      { textAnchor: "middle" as const, x: point.x, y: point.y + pointRadius + lineHeight },
      { textAnchor: "start" as const, x: point.x + pointRadius + horizontalGap, y: point.y + lineHeight * 0.45 },
      { textAnchor: "end" as const, x: point.x - pointRadius - horizontalGap, y: point.y + lineHeight * 0.45 },
      { textAnchor: "start" as const, x: point.x + pointRadius + horizontalGap + 8, y: point.y - verticalGap - 8 },
      { textAnchor: "end" as const, x: point.x - pointRadius - horizontalGap - 8, y: point.y - verticalGap - 8 }
    ];

    let bestPlacement: ScatterLabelPlacement | null = null;
    let bestRect: { left: number; right: number; top: number; bottom: number } | null = null;
    let bestScore = Number.POSITIVE_INFINITY;

    for (const candidate of candidates) {
      const rect = getScatterLabelRect(candidate, labelWidth, lineHeight);
      const clampedPlacement = clampScatterLabelPlacement(candidate, rect, bounds);
      const clampedRect = getScatterLabelRect(clampedPlacement, labelWidth, lineHeight);
      const overlapPenalty = occupiedRects.reduce((totalPenalty, occupiedRect) => {
        if (!doRectsOverlap(expandRect(occupiedRect, overlapPadding), expandRect(clampedRect, overlapPadding))) {
          return totalPenalty;
        }
        return totalPenalty + getRectOverlapArea(occupiedRect, clampedRect);
      }, 0);
      const pointPenalty = pointRects.reduce((totalPenalty, pointRect) => {
        if (!doRectsOverlap(pointRect.rect, clampedRect)) {
          return totalPenalty;
        }
        const collisionPenalty = pointRect.modelName === point.candidate.model_name ? 160 : 240;
        return totalPenalty + collisionPenalty + getRectOverlapArea(pointRect.rect, clampedRect) * 3;
      }, 0);
      const distancePenalty = Math.abs(clampedPlacement.x - point.x) * 0.35 + Math.abs(clampedPlacement.y - point.y) * 0.85;
      const score = overlapPenalty * 100 + pointPenalty * 100 + distancePenalty;
      if (score < bestScore) {
        bestPlacement = clampedPlacement;
        bestRect = clampedRect;
        bestScore = score;
      }
    }

    const finalPlacement = bestPlacement ?? { textAnchor: "start", x: point.x + horizontalGap, y: point.y - verticalGap };
    occupiedRects.push(bestRect ?? getScatterLabelRect(finalPlacement, labelWidth, lineHeight));
    placements[point.candidate.model_name] = finalPlacement;
  }

  return placements;
}

function getScatterLabelRect(placement: ScatterLabelPlacement, labelWidth: number, lineHeight: number) {
  const left =
    placement.textAnchor === "start"
      ? placement.x
      : placement.textAnchor === "end"
        ? placement.x - labelWidth
        : placement.x - labelWidth / 2;
  return {
    left,
    right: left + labelWidth,
    top: placement.y - lineHeight + 2,
    bottom: placement.y + 2
  };
}

function clampScatterLabelPlacement(
  placement: ScatterLabelPlacement,
  rect: { left: number; right: number; top: number; bottom: number },
  bounds: { left: number; right: number; top: number; bottom: number }
): ScatterLabelPlacement {
  let x = placement.x;
  let y = placement.y;
  if (rect.left < bounds.left) {
    x += bounds.left - rect.left;
  }
  if (rect.right > bounds.right) {
    x -= rect.right - bounds.right;
  }
  if (rect.top < bounds.top) {
    y += bounds.top - rect.top;
  }
  if (rect.bottom > bounds.bottom) {
    y -= rect.bottom - bounds.bottom;
  }
  return {
    textAnchor: placement.textAnchor,
    x,
    y
  };
}

function doRectsOverlap(
  left: { left: number; right: number; top: number; bottom: number },
  right: { left: number; right: number; top: number; bottom: number }
) {
  return !(
    left.right < right.left ||
    left.left > right.right ||
    left.bottom < right.top ||
    left.top > right.bottom
  );
}

function expandRect(rect: { left: number; right: number; top: number; bottom: number }, padding: number) {
  return {
    left: rect.left - padding,
    right: rect.right + padding,
    top: rect.top - padding,
    bottom: rect.bottom + padding
  };
}

function getRectOverlapArea(
  left: { left: number; right: number; top: number; bottom: number },
  right: { left: number; right: number; top: number; bottom: number }
) {
  const overlapWidth = Math.max(0, Math.min(left.right, right.right) - Math.max(left.left, right.left));
  const overlapHeight = Math.max(0, Math.min(left.bottom, right.bottom) - Math.max(left.top, right.top));
  return overlapWidth * overlapHeight;
}

type TrendMetricPoint = {
  experiment_id: string;
  experiment_index: number;
  metric_name: string;
  metric_value: number;
};

type TrendMetricSeries = {
  metricName: string;
  label: string;
  color: string;
  axis: "left" | "right";
  family: "accuracy" | "loss" | "latency" | "epoch";
  defaultVisible: boolean;
  points: TrendMetricPoint[];
};

type BestPathNode = {
  experimentId: string;
  experimentIndex: number;
  proposal: ProposalSnapshot | null;
  isBaseline: boolean;
  isCurrentBest: boolean;
  changeSummary: string | null;
};

function MetricTrendChart({
  series,
  showBestPathToggle,
  summary
}: {
  series: TrendMetricSeries[];
  showBestPathToggle: boolean;
  summary: AutoTrainSummarySnapshot | null;
}) {
  if (!series.length) {
    return <EmptyStateCard copy="Start or continue a search task to populate the trend chart." />;
  }

  const width = 760;
  const height = 380;
  const svgRef = useRef<SVGSVGElement | null>(null);
  const padding = {
    top: 30,
    right: 68,
    bottom: 52,
    left: 68
  };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const [chartMode, setChartMode] = useState<"all" | "best">("all");
  const [visibleMetricNames, setVisibleMetricNames] = useState<string[]>(() =>
    series.filter((item) => item.defaultVisible).map((item) => item.metricName)
  );
  const [zoomRange, setZoomRange] = useState<{ startExperimentIndex: number; endExperimentIndex: number } | null>(null);
  const [dragSelection, setDragSelection] = useState<{ startX: number; currentX: number } | null>(null);
  const [activeTooltipPoint, setActiveTooltipPoint] = useState<{
    experimentIndex: number;
    experimentId: string;
    x: number;
    y: number;
  } | null>(null);
  const availableMetricNames = series.map((item) => item.metricName);

  useEffect(() => {
    setVisibleMetricNames((previous) => {
      const availableNameSet = new Set(availableMetricNames);
      const hasVisibleAvailableMetric = previous.some((metricName) => availableNameSet.has(metricName));
      if (hasVisibleAvailableMetric) {
        return previous;
      }
      const defaultVisible = series.filter((item) => item.defaultVisible).map((item) => item.metricName);
      return defaultVisible.length > 0 ? defaultVisible : availableMetricNames.slice(0, 1);
    });
  }, [availableMetricNames, series]);

  const bestPathNodes = useMemo(() => buildBestPathNodes(series, summary), [series, summary]);
  useEffect(() => {
    if (chartMode === "best" && bestPathNodes.length === 0) {
      setChartMode("all");
    }
  }, [bestPathNodes.length, chartMode]);
  useEffect(() => {
    if (!showBestPathToggle && chartMode === "best") {
      setChartMode("all");
    }
  }, [chartMode, showBestPathToggle]);

  const sourceSeries =
    chartMode === "best" && bestPathNodes.length > 0 ? buildBestPathSeries(series, bestPathNodes) : series;
  const visibleSeries = sourceSeries.filter((item) => visibleMetricNames.includes(item.metricName));
  if (!visibleSeries.length) {
    return <EmptyStateCard copy="Select at least one metric to display the trend chart." />;
  }

  const allSeriesByMetricName = new Map(sourceSeries.map((item) => [item.metricName, item]));
  const baselineMetricValues = new Map(
    sourceSeries.flatMap((item) => {
      const firstPoint = [...item.points].sort((left, right) => left.experiment_index - right.experiment_index)[0];
      if (!firstPoint) {
        return [];
      }
      return [[item.metricName, firstPoint.metric_value] as const];
    })
  );
  const displayedVisibleSeries = visibleSeries;
  const zoomedVisibleSeries = displayedVisibleSeries.map((item) => ({
    ...item,
    points: zoomRange
      ? item.points.filter(
          (point) =>
            point.experiment_index >= zoomRange.startExperimentIndex &&
            point.experiment_index <= zoomRange.endExperimentIndex
        )
      : item.points
  }));

  const experimentIndexes = Array.from(
    new Set(zoomedVisibleSeries.flatMap((item) => item.points.map((point) => point.experiment_index)))
  ).sort((left, right) => left - right);
  const firstExperimentIndex = experimentIndexes[0] ?? 1;
  const lastExperimentIndex = experimentIndexes[experimentIndexes.length - 1] ?? firstExperimentIndex;
  const xTickIndexes =
    experimentIndexes.length <= 6
      ? experimentIndexes
      : Array.from(new Set([firstExperimentIndex, ...sampleTickIndexes(experimentIndexes, 6), lastExperimentIndex]));

  const leftValues = zoomedVisibleSeries
    .filter((item) => item.axis === "left")
    .flatMap((item) => item.points.map((point) => point.metric_value));
  const rightFamilies = Array.from(new Set(zoomedVisibleSeries.filter((item) => item.axis === "right").map((item) => item.family)));
  const rightFamily = (rightFamilies[0] as "loss" | "latency" | "epoch" | undefined) ?? "loss";
  const rightValues = zoomedVisibleSeries
    .filter((item) => item.axis === "right")
    .flatMap((item) => item.points.map((point) => point.metric_value));
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

  const chartSeries = zoomedVisibleSeries.map((item) => {
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
  const bestPathNodeByExperimentId = new Map(bestPathNodes.map((node) => [node.experimentId, node]));
  const labelAnchorSeries = chartSeries.find((item) => item.metricName === "top1_acc") ?? chartSeries[0] ?? null;
  const bestPathLabelEntries =
    chartMode === "best" && labelAnchorSeries
      ? labelAnchorSeries.plottedPoints.flatMap((point, index) => {
          const node = bestPathNodeByExperimentId.get(point.experiment_id);
          if (!node || node.isBaseline || !node.changeSummary) {
            return [];
          }
          const lines = wrapBestPathAnnotation(node.isCurrentBest ? `best | ${node.changeSummary}` : node.changeSummary);
          const labelWidth = Math.min(220, Math.max(...lines.map((line) => line.length), 0) * 5.5 + 18);
          const labelHeight = lines.length * 14 + 12;
          const preferredY = point.y + (index % 2 === 0 ? -labelHeight - 14 : 14);
          const clampedX = Math.min(
            Math.max(point.x - labelWidth / 2, padding.left + 6),
            width - padding.right - labelWidth - 6
          );
          const clampedY = Math.min(
            Math.max(preferredY, padding.top + 6),
            height - padding.bottom - labelHeight - 6
          );
          return [
            {
              experimentId: point.experiment_id,
              x: clampedX,
              y: clampedY,
              width: labelWidth,
              height: labelHeight,
              lines,
              isCurrentBest: node.isCurrentBest
            }
          ];
        })
      : [];
  const activeRoundEntries =
    activeTooltipPoint === null
      ? []
      : SEARCH_TREND_METRICS.flatMap((metric) => {
          const sourceSeries = allSeriesByMetricName.get(metric.metricName);
          const activePoint = sourceSeries?.points.find(
            (point) => point.experiment_index === activeTooltipPoint.experimentIndex
          );
          if (!sourceSeries || !activePoint) {
            return [];
          }
          return [
            {
              metricName: metric.metricName,
              label: metric.label,
              color: metric.color,
              family: metric.family,
              experimentId: activePoint.experiment_id,
              metricValue: activePoint.metric_value,
              baselineMetricValue: baselineMetricValues.get(metric.metricName) ?? null,
            }
          ];
        });
  const activeAnchorPoint = activeTooltipPoint;
  const activeBestPathNode =
    activeAnchorPoint && chartMode === "best" ? bestPathNodeByExperimentId.get(activeAnchorPoint.experimentId) ?? null : null;
  const tooltipChangeLines = activeBestPathNode?.changeSummary ? wrapBestPathAnnotation(activeBestPathNode.changeSummary) : [];
  const tooltipWidth = 188;
  const tooltipHeaderHeight = activeRoundEntries[0]?.experimentId ? 50 : 36;
  const tooltipHeight = tooltipHeaderHeight + activeRoundEntries.length * 15 + tooltipChangeLines.length * 13 + 10;
  const tooltipOffset = 14;
  const tooltipX = activeAnchorPoint
    ? Math.min(
        Math.max(
          activeAnchorPoint.x + (activeAnchorPoint.x > width - padding.right - tooltipWidth ? -tooltipWidth - tooltipOffset : tooltipOffset),
          padding.left + 6
        ),
        width - padding.right - tooltipWidth - 6
      )
    : 0;
  const tooltipY = activeAnchorPoint
    ? Math.min(
        Math.max(
          activeAnchorPoint.y + (activeAnchorPoint.y < padding.top + tooltipHeight ? tooltipOffset : -tooltipHeight - tooltipOffset),
          padding.top + 6
        ),
        height - padding.bottom - tooltipHeight - 6
      )
    : 0;
  const selectionStartX = dragSelection ? Math.min(dragSelection.startX, dragSelection.currentX) : null;
  const selectionWidth = dragSelection ? Math.abs(dragSelection.currentX - dragSelection.startX) : 0;

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

  const clampChartX = (value: number) => Math.min(Math.max(value, padding.left), width - padding.right);

  const getSvgXFromClientX = (clientX: number) => {
    const svgElement = svgRef.current;
    if (!svgElement) {
      return null;
    }
    const rect = svgElement.getBoundingClientRect();
    if (rect.width <= 0) {
      return null;
    }
    const relativeX = ((clientX - rect.left) / rect.width) * width;
    return clampChartX(relativeX);
  };

  const handleChartMouseDown = (event: ReactMouseEvent<SVGSVGElement>) => {
    const svgX = getSvgXFromClientX(event.clientX);
    if (svgX === null) {
      return;
    }
    setActiveTooltipPoint(null);
    setDragSelection({ startX: svgX, currentX: svgX });
  };

  const handleChartMouseMove = (event: ReactMouseEvent<SVGSVGElement>) => {
    if (!dragSelection) {
      return;
    }
    const svgX = getSvgXFromClientX(event.clientX);
    if (svgX === null) {
      return;
    }
    setDragSelection((previous) => (previous ? { ...previous, currentX: svgX } : previous));
  };

  const finishChartZoom = () => {
    if (!dragSelection) {
      return;
    }
    const minX = Math.min(dragSelection.startX, dragSelection.currentX);
    const maxX = Math.max(dragSelection.startX, dragSelection.currentX);
    setDragSelection(null);
    if (maxX - minX < 18) {
      return;
    }
    const selectedExperimentIndexes = experimentIndexes.filter((experimentIndex) => {
      const x = getX(experimentIndex);
      return x >= minX && x <= maxX;
    });
    if (selectedExperimentIndexes.length < 2) {
      return;
    }
    setZoomRange({
      startExperimentIndex: selectedExperimentIndexes[0],
      endExperimentIndex: selectedExperimentIndexes[selectedExperimentIndexes.length - 1]
    });
  };

  return (
    <div className="chart-card">
      <div className="chart-header">
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
        <div className="chart-actions">
          {showBestPathToggle ? (
            <button
              className="text-button chart-reset-button"
              disabled={bestPathNodes.length === 0}
              onClick={() => setChartMode((previous) => (previous === "all" ? "best" : "all"))}
              type="button"
            >
              {chartMode === "all" ? "Path" : "All"}
            </button>
          ) : null}
          {zoomRange ? (
            <button className="text-button chart-reset-button" onClick={() => setZoomRange(null)} type="button">
              Reset Zoom
            </button>
          ) : null}
        </div>
      </div>
      <svg
        aria-label="Metric trend chart"
        className="chart-svg"
        onMouseDown={handleChartMouseDown}
        onMouseMove={handleChartMouseMove}
        onMouseUp={finishChartZoom}
        onMouseLeave={finishChartZoom}
        ref={svgRef}
        role="img"
        viewBox={`0 0 ${width} ${height}`}
      >
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
        {selectionStartX !== null && selectionWidth > 0 ? (
          <rect
            className="chart-zoom-selection"
            height={chartHeight}
            rx={10}
            ry={10}
            width={selectionWidth}
            x={selectionStartX}
            y={padding.top}
          />
        ) : null}
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
              <g
                key={`${item.metricName}-${point.experiment_id}`}
                onMouseEnter={() =>
                  setActiveTooltipPoint({
                    experimentIndex: point.experiment_index,
                    experimentId: point.experiment_id,
                    x: point.x,
                    y: point.y
                  })
                }
                onMouseLeave={() =>
                  setActiveTooltipPoint((previous) =>
                    previous?.experimentIndex === point.experiment_index ? null : previous
                  )
                }
              >
                <circle
                  className="chart-dot trend-chart-dot"
                  cx={point.x}
                  cy={point.y}
                  r={activeTooltipPoint?.experimentIndex === point.experiment_index ? 5 : 3.25}
                  style={{ fill: item.color, stroke: item.color }}
                />
              </g>
            ))}
          </g>
        ))}
        {bestPathLabelEntries.map((entry) => (
          <g key={`annotation-${entry.experimentId}`} transform={`translate(${entry.x} ${entry.y})`}>
            <rect
              className={`chart-annotation-box${entry.isCurrentBest ? " chart-annotation-box-current" : ""}`}
              height={entry.height}
              rx={10}
              ry={10}
              width={entry.width}
              x={0}
              y={0}
            />
            {entry.lines.map((line, index) => (
              <text
                className={`chart-annotation-text${entry.isCurrentBest && index === 0 ? " chart-annotation-text-current" : ""}`}
                key={`${entry.experimentId}-${line}-${index}`}
                x={10}
                y={20 + index * 14}
              >
                {line}
              </text>
            ))}
          </g>
        ))}
        {activeAnchorPoint ? (
          <g transform={`translate(${tooltipX} ${tooltipY})`}>
            <rect className="chart-tooltip-box" height={tooltipHeight} rx={14} ry={14} width={tooltipWidth} x={0} y={0} />
            <text className="chart-tooltip-title" x={14} y={20}>
              {`Round ${activeTooltipPoint?.experimentIndex}`}
            </text>
            {activeRoundEntries[0]?.experimentId ? (
              <text className="chart-tooltip-line" x={14} y={34}>
                {activeRoundEntries[0].experimentId}
              </text>
            ) : null}
            {activeRoundEntries.map((entry, index) => (
              <g key={`tooltip-${entry.metricName}`} transform={`translate(14 ${tooltipHeaderHeight + index * 15})`}>
                <circle cx={4} cy={0} fill={entry.color} r={3.5} />
                <text className="chart-tooltip-line" x={14} y={4}>
                  {`${entry.label} ${formatTrendMetricValue(entry.metricValue, entry.family)}${formatTooltipBaselineDelta(
                    entry.metricValue,
                    entry.baselineMetricValue,
                    entry.family
                  )}`}
                </text>
              </g>
            ))}
            {tooltipChangeLines.map((line, index) => (
              <text
                className="chart-tooltip-line chart-tooltip-line-annotation"
                key={`tooltip-change-${line}-${index}`}
                x={14}
                y={tooltipHeaderHeight + activeRoundEntries.length * 15 + 12 + index * 13}
              >
                {line}
              </text>
            ))}
          </g>
        ) : null}
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
  status?: string;
  decision?: string | null;
  decision_reason?: string | null;
  metrics?: Record<string, number>;
  summary?: string;
};

type RoundSnapshot = {
  round_index?: number;
  proposal?: ProposalSnapshot;
  result?: SummarySnapshot;
};

type AutoTrainSummarySnapshot = {
  baseline?: SummarySnapshot;
  rounds?: RoundSnapshot[];
  current_proposal?: ProposalSnapshot | null;
  ai_summary?: string | null;
  ai_summary_error?: string | null;
  stop_reason?: string | null;
};

function readAutoTrainSummary(task: AutoTrainTask | null): AutoTrainSummarySnapshot | null {
  if (!task?.summary || typeof task.summary !== "object") {
    return null;
  }
  return task.summary as AutoTrainSummarySnapshot;
}

function buildPersistedSearchTrendSeries(summary: AutoTrainSummarySnapshot | null) {
  if (!summary) {
    return [];
  }

  const resultRows: Array<{ experiment_id: string; experiment_index: number; metrics: Record<string, number> }> = [];
  const baseline = summary.baseline;
  if (baseline?.experiment_id && baseline.metrics) {
    resultRows.push({
      experiment_id: baseline.experiment_id,
      experiment_index: 1,
      metrics: baseline.metrics
    });
  }

  (summary.rounds ?? []).forEach((round, index) => {
    const result = round.result;
    if (!result?.experiment_id || !result.metrics) {
      return;
    }
    resultRows.push({
      experiment_id: result.experiment_id,
      experiment_index: baseline?.experiment_id ? index + 2 : index + 1,
      metrics: result.metrics
    });
  });

  return SEARCH_TREND_METRICS.map((metric) => ({
    ...metric,
    points: resultRows.flatMap((row) => {
      const metricValue = row.metrics[metric.metricName];
      if (typeof metricValue !== "number") {
        return [];
      }
      return [
        {
          experiment_id: row.experiment_id,
          experiment_index: row.experiment_index,
          metric_name: metric.metricName,
          metric_value: metricValue
        }
      ];
    })
  })).filter((series) => series.points.length > 0);
}

function buildTrendSeriesFromPayload(payload: RunTrendPayload | undefined) {
  if (!payload) {
    return [];
  }

  const pointsByMetricName = new Map(payload.series.map((item) => [item.metric_name, item.points]));
  return SEARCH_TREND_METRICS.map((metric) => ({
    ...metric,
    points: pointsByMetricName.get(metric.metricName) ?? []
  })).filter((series) => series.points.length > 0);
}

function buildBestPathNodes(
  series: TrendMetricSeries[],
  summary: AutoTrainSummarySnapshot | null
): BestPathNode[] {
  if (!summary) {
    return [];
  }

  const experimentIndexById = new Map<string, number>();
  series.forEach((item) => {
    item.points.forEach((point) => {
      if (!experimentIndexById.has(point.experiment_id)) {
        experimentIndexById.set(point.experiment_id, point.experiment_index);
      }
    });
  });

  const nodes: BestPathNode[] = [];
  const baselineExperimentId = summary.baseline?.experiment_id;
  if (baselineExperimentId) {
    nodes.push({
      experimentId: baselineExperimentId,
      experimentIndex: experimentIndexById.get(baselineExperimentId) ?? 1,
      proposal: null,
      isBaseline: true,
      isCurrentBest: false,
      changeSummary: null
    });
  }

  (summary.rounds ?? []).forEach((round, index) => {
    const result = round.result;
    if (!result?.experiment_id || result.decision !== "keep") {
      return;
    }
    nodes.push({
      experimentId: result.experiment_id,
      experimentIndex:
        experimentIndexById.get(result.experiment_id) ??
        (baselineExperimentId ? index + 2 : index + 1),
      proposal: round.proposal ?? null,
      isBaseline: false,
      isCurrentBest: false,
      changeSummary: describeCompactProposalChanges(round.proposal ?? null)
    });
  });

  const deduplicatedNodes = Array.from(
    new Map(nodes.map((node) => [node.experimentId, node])).values()
  ).sort((left, right) => left.experimentIndex - right.experimentIndex);
  if (deduplicatedNodes.length > 0) {
    deduplicatedNodes[deduplicatedNodes.length - 1] = {
      ...deduplicatedNodes[deduplicatedNodes.length - 1],
      isCurrentBest: true
    };
  }
  return deduplicatedNodes;
}

function buildBestPathSeries(series: TrendMetricSeries[], bestPathNodes: BestPathNode[]) {
  const bestPathExperimentIds = new Set(bestPathNodes.map((node) => node.experimentId));
  return series
    .map((item) => ({
      ...item,
      points: item.points.filter((point) => bestPathExperimentIds.has(point.experiment_id))
    }))
    .filter((item) => item.points.length > 0);
}

function wrapBestPathAnnotation(text: string, maxLineLength = 34) {
  const segments = text
    .split(" | ")
    .map((segment) => segment.trim())
    .filter(Boolean);
  if (segments.length === 0) {
    return [];
  }
  const lines: string[] = [];
  let currentLine = "";
  segments.forEach((segment) => {
    const nextLine = currentLine ? `${currentLine} | ${segment}` : segment;
    if (nextLine.length <= maxLineLength || !currentLine) {
      currentLine = nextLine;
      return;
    }
    lines.push(currentLine);
    currentLine = segment;
  });
  if (currentLine) {
    lines.push(currentLine);
  }
  return lines;
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

function buildCompareResultSummary(task: ModelCompareTask | null) {
  const aiSummary = task?.summary?.ai_summary?.trim();
  if (aiSummary) {
    return aiSummary;
  }
  const aiSummaryError = task?.summary?.ai_summary_error?.trim();
  if (!aiSummaryError) {
    return null;
  }
  return `AI summary failed: ${aiSummaryError}`;
}

function buildCompareResultSummarySource(task: ModelCompareTask | null) {
  const aiSummary = task?.summary?.ai_summary?.trim();
  const aiModelLabel = formatAiModelLabel(task?.ai_model_name);
  if (aiSummary) {
    return aiModelLabel ? `Summary from ${aiModelLabel}` : "Summary";
  }
  const aiSummaryError = task?.summary?.ai_summary_error?.trim();
  if (!aiSummaryError) {
    return null;
  }
  return aiModelLabel ? `Summary failed from ${aiModelLabel}` : "Summary failed";
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

function buildSearchTaskTitle(dataset: string, modelName: SupportedModelName) {
  return `Search - ${MODEL_LABELS[modelName] ?? modelName} - ${dataset}`;
}

function buildSearchMetaInlineText(task: AutoTrainTask | null, elapsedSeconds: number) {
  if (!task) {
    return null;
  }

  const parts: string[] = [];
  if (task.current_round) {
    parts.push(`Round ${task.current_round}`);
  }
  parts.push(formatElapsed(elapsedSeconds));
  if ((task.provider_total_tokens_total ?? 0) > 0) {
    parts.push(`Tokens ${formatCompactInteger(task.provider_total_tokens_total)}`);
  }
  return parts.length ? parts.join(" · ") : null;
}

function buildSearchRunLabel(task: AutoTrainTask | null) {
  const runId = task?.run_id?.trim();
  if (!runId) {
    return null;
  }
  return `Run ${runId}`;
}

function buildSearchGpuMemoryLabel(summary: AutoTrainSummarySnapshot | null) {
  const latestSuccessfulResult =
    [...(summary?.rounds ?? [])].reverse().find((round) => round.result?.status === "success")?.result ??
    summary?.baseline ??
    null;
  const gpuMemoryMb = latestSuccessfulResult?.resource?.gpu_memory_mb;
  if (typeof gpuMemoryMb !== "number" || gpuMemoryMb <= 0) {
    return null;
  }
  return `Peak GPU ${formatGpuMemory(gpuMemoryMb)}`;
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
    return (
      summary?.current_proposal?.hypothesis?.trim() ??
      [...(summary?.rounds ?? [])].reverse().find((round) => round.proposal?.hypothesis?.trim())?.proposal?.hypothesis?.trim() ??
      null
    );
  }
  return null;
}

function buildSearchResultSummary(summary: AutoTrainSummarySnapshot | null) {
  const aiSummary = summary?.ai_summary?.trim();
  if (aiSummary) {
    return aiSummary;
  }
  const aiSummaryError = summary?.ai_summary_error?.trim();
  if (!aiSummaryError) {
    return null;
  }
  return `AI summary failed: ${aiSummaryError}`;
}

function buildSearchResultSummarySource(
  task: AutoTrainTask | null,
  summary: AutoTrainSummarySnapshot | null
) {
  const aiSummary = summary?.ai_summary?.trim();
  const aiModelLabel = formatAiModelLabel(task?.ai_model_name);
  if (aiSummary) {
    return aiModelLabel ? `Summary from ${aiModelLabel}` : "Summary";
  }
  const aiSummaryError = summary?.ai_summary_error?.trim();
  if (!aiSummaryError) {
    return null;
  }
  return aiModelLabel ? `Summary failed from ${aiModelLabel}` : "Summary failed";
}

function buildSearchSummaryLoadingText(task: AutoTrainTask | null, summaryText: string | null) {
  if (!task || summaryText) {
    return null;
  }
  const activityMessage = task.activity_message?.trim() ?? "";
  if (/generating search summary/i.test(activityMessage)) {
    return "Generating AI summary...";
  }
  return null;
}

function buildCompareSummaryLoadingText(task: ModelCompareTask | null) {
  if (!task || buildCompareResultSummary(task)) {
    return null;
  }
  const activityMessage = task.activity_message?.trim() ?? "";
  if (/generating compare summary/i.test(activityMessage)) {
    return "Generating AI summary...";
  }
  return null;
}

function formatAiModelLabel(modelName: string | null | undefined) {
  if (!modelName) {
    return null;
  }
  const normalizedModelName = modelName.trim();
  if (!normalizedModelName) {
    return null;
  }
  const pathSegments = normalizedModelName.split("/").filter(Boolean);
  return pathSegments[pathSegments.length - 1] ?? normalizedModelName;
}

function formatGpuMemory(memoryMb: number) {
  if (memoryMb >= 1024) {
    return `${Number((memoryMb / 1024).toFixed(1))} GB`;
  }
  return `${Math.round(memoryMb)} MB`;
}

function buildTaskProgressFooter(task: AutoTrainTask | ModelCompareTask | null) {
  if (!task) {
    return null;
  }
  return buildTaskStartupMessage(task) ?? buildTaskDatasetSummary(task) ?? null;
}

function buildTaskStartupMessage(task: AutoTrainTask | ModelCompareTask | null) {
  if (!task) {
    return null;
  }
  const activityMessage = task.activity_message?.trim();
  if (activityMessage && /(validating|creating|starting|preparing|loading|checking|queued)/i.test(activityMessage)) {
    return activityMessage;
  }
  return null;
}

function buildTaskDatasetSummary(task: AutoTrainTask | ModelCompareTask | null) {
  if (!task) {
    return null;
  }
  const summary = task.dataset_summary?.trim() ?? null;
  const trainingImageSize = Number(task.training_image_size ?? 0);
  if (!summary) {
    return trainingImageSize > 0 ? `train size ${trainingImageSize}x${trainingImageSize}` : null;
  }
  return summary;
}

function isProviderOverloadedText(text: string | null | undefined) {
  if (!text) {
    return false;
  }
  return /(status=529|overloaded_error|high load|currently under high load)/i.test(text);
}

function isProviderOverloadedTaskFailure(task: AutoTrainTask | ModelCompareTask | null) {
  if (!task || task.status !== "failed") {
    return false;
  }
  return isProviderOverloadedText(task.error) || isProviderOverloadedText(task.activity_message);
}

function buildTaskFailureNotice(task: AutoTrainTask | ModelCompareTask | null, externalErrorMessage?: string | null) {
  const normalizedExternalError = externalErrorMessage?.trim() ?? null;
  if (normalizedExternalError) {
    if (isProviderOverloadedText(normalizedExternalError)) {
      return "Provider is under high load (529). Retry after a short wait.";
    }
    return normalizedExternalError;
  }

  if (!task || task.status !== "failed") {
    return null;
  }

  if (isProviderOverloadedTaskFailure(task)) {
    return "Provider is under high load (529). Retry after a short wait.";
  }
  return task.error?.trim() ?? null;
}

function buildTaskWarningNotice(task: AutoTrainTask | null) {
  if (!task || task.status === "failed") {
    return null;
  }
  const proposalWarning = task.proposal_warning?.trim() ?? "";
  if (proposalWarning) {
    return proposalWarning;
  }
  const activityMessage = task.activity_message?.trim() ?? "";
  if (!/retry/i.test(activityMessage)) {
    return null;
  }
  const latestRetryLog = [...task.logs].reverse().find((entry) => /retrying in \d+s/i.test(entry));
  if (!latestRetryLog) {
    return null;
  }
  const normalizedLog = latestRetryLog.trim();
  const attemptMatch = normalizedLog.match(/\((\d+)\/(\d+)\)/);
  const delayMatch = normalizedLog.match(/retrying in (\d+)s/i);
  const attemptText = attemptMatch ? ` (${attemptMatch[1]}/${attemptMatch[2]})` : "";
  const delayText = delayMatch ? `${delayMatch[1]}s` : "a moment";
  const normalizedLower = normalizedLog.toLowerCase();
  if (normalizedLower.includes("read timed out") || normalizedLower.includes("connect timeout")) {
    return `Provider timeout. Retrying in ${delayText}${attemptText}.`;
  }
  if (normalizedLower.includes("status=429") || normalizedLower.includes("high load") || normalizedLower.includes("overloaded")) {
    return `Provider is busy. Retrying in ${delayText}${attemptText}.`;
  }
  if (normalizedLower.includes("round ") || normalizedLower.includes("baseline")) {
    return `Experiment failed. Retrying in ${delayText}${attemptText}.`;
  }
  return `Request failed. Retrying in ${delayText}${attemptText}.`;
}

function buildLatestExperimentFailureNotice(task: AutoTrainTask | null) {
  if (!task || task.status === "failed") {
    return null;
  }
  const latestFailureLog = [...task.logs].reverse().find((entry) => {
    const normalizedEntry = entry.toLowerCase();
    return (
      normalizedEntry.includes("training failed:") ||
      normalizedEntry.includes("retry budget exhausted without a valid result") ||
      normalizedEntry.includes("proposal cannot build a valid follow-up config")
    );
  });
  if (!latestFailureLog) {
    return null;
  }

  const normalizedLog = latestFailureLog.trim();
  const trainingFailureMatch = normalizedLog.match(/training failed:\s*(.+)$/i);
  if (trainingFailureMatch?.[1]) {
    return `Last round failed: ${trainingFailureMatch[1].trim()}`;
  }
  const invalidConfigMatch = normalizedLog.match(/Proposal cannot build a valid follow-up config:\s*(.+)$/i);
  if (invalidConfigMatch?.[1]) {
    return `Last proposal was rejected: ${invalidConfigMatch[1].trim()}`;
  }
  if (/retry budget exhausted without a valid result/i.test(normalizedLog)) {
    return "Last round failed repeatedly and did not produce a valid result.";
  }
  return null;
}

const PROPOSAL_CHANGE_SHORT_VALUE_LABELS: Record<string, string> = {
  cross_entropy: "ce",
  cross_entropy_with_label_smoothing: "ce+ls",
  focal_loss: "focal",
  native_classifier: "native",
  dropout_linear: "drop+lin",
  linear: "linear",
  avg_pool: "avg",
  gem_pool: "gem",
  cosine: "cos",
  step: "step",
  none: "none",
  basic: "basic",
  adamw: "adamw",
  adam: "adam",
  sgd: "sgd",
  true: "on",
  false: "off"
};

function describeProposalChangeSummary(proposal: ProposalSnapshot | null) {
  if (!proposal) {
    return null;
  }
  const changedFields = collectStructuredProposalChangeEntries(proposal).map((entry) => entry.path);
  if (!changedFields.length) {
    return null;
  }
  return changedFields.join(" / ");
}

function describeValidatedProposalChanges(proposal: ProposalSnapshot | null) {
  if (!proposal) {
    return null;
  }
  const changedFields = Object.entries(proposal.changes ?? {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([fieldName, value]) => `${fieldName} ${formatCompactProposalValue(value)}`);
  if (!changedFields.length) {
    return null;
  }
  return changedFields.join(" | ");
}

function describeCompactProposalChanges(proposal: ProposalSnapshot | null) {
  if (!proposal) {
    return null;
  }
  const changedFields = collectStructuredProposalChangeEntries(proposal).map((entry) => {
    return `${entry.path} ${formatCompactProposalValue(entry.value)}`;
  });
  if (!changedFields.length) {
    return null;
  }
  return changedFields.join(" | ");
}

function collectStructuredProposalChangeEntries(proposal: ProposalSnapshot) {
  const structuredEntries = [
    ...flattenChangeTree(proposal.train_hyp_changes, ["train_hyp"]),
    ...flattenChangeTree(proposal.recipe_changes, ["model_recipe"])
  ];
  if (structuredEntries.length > 0) {
    return structuredEntries;
  }
  return Object.entries(proposal.changes ?? {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([fieldName, value]) => ({ path: fieldName, value }));
}

function flattenChangeTree(changeTree: unknown, pathParts: string[] = []): Array<{ path: string; value: unknown }> {
  if (!isPlainObject(changeTree)) {
    return [];
  }
  const flattenedEntries: Array<{ path: string; value: unknown }> = [];
  for (const [fieldName, value] of Object.entries(changeTree)) {
    if (value === null || value === undefined) {
      continue;
    }
    const nextPathParts = [...pathParts, fieldName];
    if (isPlainObject(value)) {
      const nestedEntries = flattenChangeTree(value, nextPathParts);
      if (nestedEntries.length > 0) {
        flattenedEntries.push(...nestedEntries);
      }
      continue;
    }
    flattenedEntries.push({ path: nextPathParts.join("."), value });
  }
  return flattenedEntries;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function formatCompactProposalValue(value: unknown) {
  if (typeof value === "number") {
    if (Number.isInteger(value)) {
      return `${value}`;
    }
    if (Math.abs(value) >= 1 || value === 0) {
      return Number(value.toFixed(3)).toString();
    }
    return value.toExponential(1).replace("e-0", "e-").replace("e+0", "e+");
  }
  if (typeof value === "boolean") {
    return value ? "on" : "off";
  }
  if (typeof value === "string") {
    return PROPOSAL_CHANGE_SHORT_VALUE_LABELS[value] ?? value.replace(/_/g, " ");
  }
  return String(value);
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

function formatTrendMetricValue(value: number, family: "accuracy" | "loss" | "latency" | "epoch") {
  if (family === "accuracy") {
    return formatAccuracy(value);
  }
  if (family === "latency") {
    return formatLatency(value);
  }
  if (family === "epoch") {
    return `${Math.round(value)}`;
  }
  return formatLossTick(value);
}

function formatSignedAccuracyDelta(value: number) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${Math.abs(value * 100).toFixed(2)}%`;
}

function formatTooltipBaselineDelta(
  metricValue: number,
  baselineMetricValue: number | null,
  family: "accuracy" | "loss" | "latency" | "epoch"
) {
  if (baselineMetricValue === null) {
    return "";
  }
  const delta = metricValue - baselineMetricValue;
  if (family === "accuracy") {
    return ` (${formatSignedAccuracyDelta(delta)})`;
  }
  if (family === "latency") {
    return ` (${formatSignedNumberDelta(delta, 2)} ms)`;
  }
  if (family === "epoch") {
    return ` (${formatSignedNumberDelta(delta, 0)})`;
  }
  return ` (${formatSignedNumberDelta(delta, 4)})`;
}

function formatSignedNumberDelta(value: number, digits: number) {
  const sign = value > 0 ? "+" : value < 0 ? "-" : "";
  return `${sign}${Math.abs(value).toFixed(digits)}`;
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
    case "mobilenet_v3_large":
      return "#6a5cff";
    case "efficientnet_b0":
      return "#0f8b8d";
    case "efficientnet_b1":
      return "#1b9aaa";
    case "googlenet":
      return "#11a36a";
    case "resnet18":
      return "#d9485f";
    case "resnet34":
      return "#c83f57";
    case "resnet50":
      return "#b5314e";
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

function toTaskType(mode: WorkspaceMode): TaskType {
  return mode === "compare" ? "model_compare" : "auto_train";
}

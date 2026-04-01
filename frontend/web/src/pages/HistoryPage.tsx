import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate, useOutletContext } from "react-router-dom";

import { EMPTY_APP_SHELL_HEADER_CONTENT, type AppShellOutletContext } from "../app/AppShell";
import { useDeleteTask, useRenameTaskTitle, useTaskHistory } from "../features/api/hooks";
import type { TaskHistoryItem } from "../types/domain";

export function HistoryPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { setHeaderContent } = useOutletContext<AppShellOutletContext>();
  const taskHistoryQuery = useTaskHistory();
  const deleteTaskMutation = useDeleteTask();
  const renameTaskTitleMutation = useRenameTaskTitle();
  const taskHistory = taskHistoryQuery.data ?? [];
  const [openMenuTaskId, setOpenMenuTaskId] = useState<string | null>(null);

  useEffect(() => {
    setHeaderContent({
      title: null,
      actions: null
    });
    return () => setHeaderContent(EMPTY_APP_SHELL_HEADER_CONTENT);
  }, [setHeaderContent]);

  return (
    <div className="history-page">
      <section className="history-panel">
        <div className="section-heading">
          <h2>My Tasks</h2>
        </div>
        {taskHistoryQuery.error ? <p className="error-copy">{taskHistoryQuery.error.message}</p> : null}
        <div className="history-list">
          <Link className="history-card history-card-new" to="/workspace?new=1">
            <div className="history-card-top">
              <span className="history-new-icon" aria-hidden="true">
                +
              </span>
            </div>
            <div className="history-copy">
              <h3>New Task</h3>
              <p>Start a new compare or search task.</p>
            </div>
          </Link>
          {taskHistory.length === 0 ? (
            <div className="history-card empty-state-card">
              <p>No task history yet. Start a compare or search task from the workspace.</p>
            </div>
          ) : (
            taskHistory.map((taskItem) => (
              <div
                className="history-card history-card-clickable"
                key={taskItem.task_id}
                onClick={() => navigate(`/workspace?taskId=${taskItem.task_id}&taskType=${taskItem.task_type}`)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    navigate(`/workspace?taskId=${taskItem.task_id}&taskType=${taskItem.task_type}`);
                  }
                }}
                role="link"
                tabIndex={0}
              >
                <div className="history-card-top">
                  <div className="history-chip-row">
                    <span className={`history-type-chip history-type-chip-${taskItem.task_type}`}>{formatTaskTypeLabel(taskItem.task_type)}</span>
                    {taskItem.source_task_type === "model_compare" && taskItem.source_task_id ? (
                      <button
                        className="history-origin-chip history-origin-chip-action"
                        onClick={(event) => {
                          event.stopPropagation();
                          navigate(`/workspace?taskId=${taskItem.source_task_id}&taskType=model_compare`);
                        }}
                        type="button"
                      >
                        From Compare
                      </button>
                    ) : taskItem.source_task_type === "model_compare" ? (
                      <span className="history-origin-chip">From Compare</span>
                    ) : null}
                  </div>
                  <div
                    className="history-card-actions"
                    onClick={(event) => event.stopPropagation()}
                    onKeyDown={(event) => event.stopPropagation()}
                  >
                    <span className="history-time-label">{formatHistoryTime(taskItem.updated_at)}</span>
                    <div className="history-menu">
                      <button
                        aria-expanded={openMenuTaskId === taskItem.task_id}
                        className="history-menu-button"
                        onClick={() => setOpenMenuTaskId((previous) => (previous === taskItem.task_id ? null : taskItem.task_id))}
                        type="button"
                      >
                        ⋮
                      </button>
                      {openMenuTaskId === taskItem.task_id ? (
                        <div className="history-menu-popover">
                          <button
                            className="history-menu-item"
                            disabled={renameTaskTitleMutation.isPending}
                            onClick={async () => {
                              const nextTitle = window.prompt("Rename task", taskItem.title);
                              if (nextTitle === null) {
                                return;
                              }
                              const normalizedTitle = nextTitle.trim();
                              if (!normalizedTitle || normalizedTitle === taskItem.title) {
                                setOpenMenuTaskId(null);
                                return;
                              }
                              try {
                                await renameTaskTitleMutation.mutateAsync({
                                  taskId: taskItem.task_id,
                                  taskType: taskItem.task_type,
                                  title: normalizedTitle
                                });
                                await Promise.all([
                                  queryClient.invalidateQueries({ queryKey: ["task-history"] }),
                                  queryClient.invalidateQueries({ queryKey: ["auto-train-task"] }),
                                  queryClient.invalidateQueries({ queryKey: ["model-compare-task"] })
                                ]);
                                setOpenMenuTaskId(null);
                              } catch (error) {
                                const message = error instanceof Error ? error.message : "Rename task failed.";
                                window.alert(message);
                              }
                            }}
                            type="button"
                          >
                            Rename
                          </button>
                          <button
                            className="history-menu-item history-menu-item-danger"
                            disabled={deleteTaskMutation.isPending}
                            onClick={async () => {
                              if (!window.confirm(buildDeleteConfirmation(taskItem))) {
                                return;
                              }
                              try {
                                await deleteTaskMutation.mutateAsync({
                                  taskId: taskItem.task_id,
                                  taskType: taskItem.task_type
                                });
                                queryClient.setQueryData<TaskHistoryItem[]>(["task-history"], (previous) =>
                                  (previous ?? []).filter((item) => item.task_id !== taskItem.task_id)
                                );
                                await Promise.all([
                                  queryClient.invalidateQueries({ queryKey: ["task-history"] }),
                                  queryClient.invalidateQueries({ queryKey: ["runs"] }),
                                  queryClient.invalidateQueries({ queryKey: ["run-detail"] }),
                                  queryClient.invalidateQueries({ queryKey: ["run-summary"] }),
                                  queryClient.invalidateQueries({ queryKey: ["run-metrics"] }),
                                  queryClient.invalidateQueries({ queryKey: ["auto-train-task"] }),
                                  queryClient.invalidateQueries({ queryKey: ["model-compare-task"] })
                                ]);
                                setOpenMenuTaskId(null);
                              } catch (error) {
                                const message = error instanceof Error ? error.message : "Delete task failed.";
                                window.alert(message);
                              }
                            }}
                            type="button"
                          >
                            Delete
                          </button>
                        </div>
                      ) : null}
                    </div>
                  </div>
                </div>
                <div className="history-copy">
                  <h3>{taskItem.title}</h3>
                  <p>{taskItem.summary ?? buildTaskSummary(taskItem)}</p>
                </div>
                <div className="history-meta">
                  <span className={`history-status-pill history-status-pill-${normalizeStatusTone(taskItem.status)}`}>
                    {formatStatusLabel(taskItem.status)}
                  </span>
                  <span>{taskItem.dataset ?? "Unknown dataset"}</span>
                  <span>{taskItem.model_name ?? "Multi-model"}</span>
                </div>
              </div>
            ))
          )}
        </div>
      </section>
    </div>
  );
}

function buildTaskSummary(taskItem: {
  task_type: "auto_train" | "model_compare";
  dataset?: string | null;
  model_name?: string | null;
  candidate_models?: string[];
  source_task_type?: "model_compare" | null;
  source_model_name?: string | null;
}) {
  if (taskItem.task_type === "auto_train") {
    if (taskItem.source_task_type === "model_compare" && taskItem.source_model_name) {
      return `Search from compare using ${taskItem.source_model_name} on ${taskItem.dataset ?? "dataset"}.`;
    }
    return `Search on ${taskItem.dataset ?? "dataset"} with ${taskItem.model_name ?? "model"}.`;
  }
  const candidateCount = taskItem.candidate_models?.length ?? 0;
  return `Compare ${candidateCount || "multiple"} models on ${taskItem.dataset ?? "dataset"}.`;
}

function formatTaskTypeLabel(taskType: "auto_train" | "model_compare") {
  return taskType === "auto_train" ? "Search" : "Compare";
}

function formatStatusLabel(status: string) {
  return status.replace(/_/g, " ");
}

function normalizeStatusTone(status: string) {
  if (status === "failed") {
    return "danger";
  }
  if (status === "success") {
    return "success";
  }
  if (status === "stopping") {
    return "warning";
  }
  if (["running", "queued"].includes(status)) {
    return "active";
  }
  if (["stopped", "stopped_by_policy"].includes(status)) {
    return "neutral";
  }
  return "neutral";
}

function formatHistoryTime(value: string | null | undefined) {
  if (!value) {
    return "Unknown time";
  }
  const parsedDate = new Date(value);
  if (Number.isNaN(parsedDate.getTime())) {
    return "Unknown time";
  }
  return parsedDate.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function buildDeleteConfirmation(taskItem: TaskHistoryItem) {
  if (taskItem.task_type === "model_compare") {
    return "Delete this compare task?\n\nThis will also delete any search tasks started from this compare.";
  }
  return "Delete this search task?\n\nAny run created only for this search will be removed too, including weights, logs, and related artifacts.";
}

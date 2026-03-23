"""Streamlit demo UI for the autonomous training platform."""

from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_MODELS = {
    "mobilenet_v2": {
        "model_family": "mobilenet",
        "parameter_space_version": "mobilenet_v2@v1",
        "parameter_space": {
            "model_name": "mobilenet_v2",
            "version": "mobilenet_v2@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_level": {"type": "enum", "choices": ["low", "medium", "high"]},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            },
        },
    },
    "googlenet": {
        "model_family": "googlenet",
        "parameter_space_version": "googlenet@v1",
        "parameter_space": {
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
                "augmentation_level": {"type": "enum", "choices": ["low", "medium", "high"]},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
                "aux_logits": {"type": "enum", "choices": [True, False]},
            },
        },
    },
}

LOG_LIMIT = 60
LIVE_LOG_CONTAINER: Any | None = None
LIVE_AI_PANEL_CONTAINER: Any | None = None
AI_BLOCKED_CHANGE_FIELDS = {"epochs"}


def request_json(path: str, fallback: Any) -> Any:
    """Fetch JSON from the backend and fall back to local demo data."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=0.8)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return fallback


def post_json(path: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    """Post JSON to the backend and return success flag with payload or error."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, error.response.json()
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def post_without_body(path: str) -> tuple[bool, Any]:
    """Post without a request body."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", timeout=3600)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, error.response.json()
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def find_running_experiment_id(runs: list[dict[str, Any]]) -> str | None:
    """Return the currently running experiment id if any."""
    for run in runs:
        run_detail = request_json(f'/runs/{run["id"]}', {})
        for experiment in run_detail.get("experiments", []):
            if experiment.get("status") == "running":
                return experiment.get("id")
    return None


def generate_aihubmix_proposal_request(run_id: str) -> tuple[bool, Any]:
    """Request one real proposal from the backend."""
    try:
        response = requests.post(f"{API_BASE_URL}/runs/{run_id}/proposal", timeout=60)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, error.response.json()
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def start_auto_train_task_request(payload: dict[str, Any], selected_run_id: str, rounds: int) -> tuple[bool, Any]:
    """Start one backend auto-train task."""
    request_payload = {
        "run_id": None if selected_run_id == "__all__" else selected_run_id,
        "run_name": payload["name"],
        "dataset": payload["dataset"],
        "model_name": payload["model_name"],
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "rounds": rounds,
    }
    return post_json("/runs/auto-train", request_payload)


def load_auto_train_task(task_id: str) -> tuple[bool, Any]:
    """Load one backend auto-train task."""
    return True, request_json(f"/runs/auto-train/{task_id}", {"detail": "Auto train task not found"})


def stop_auto_train_task_request(task_id: str) -> tuple[bool, Any]:
    """Stop one backend auto-train task."""
    return post_without_body(f"/runs/auto-train/{task_id}/stop")


def wait_for_experiment_completion(experiment_id: str, timeout_seconds: int = 3600) -> tuple[bool, dict[str, Any]]:
    """Poll one experiment until it reaches a terminal state."""
    started_at = time.time()
    while time.time() - started_at < timeout_seconds:
        experiment_detail = load_experiment_detail(experiment_id)
        if experiment_detail.get("status") in {"success", "failed", "discarded"}:
            return True, experiment_detail
        time.sleep(2)
    return False, {"detail": f"Experiment {experiment_id} did not finish within timeout."}


def load_runs() -> list[dict[str, Any]]:
    """Return all runs."""
    return request_json(
        "/runs",
        [
            {
                "id": "run_demo_001",
                "name": "CIFAR-10 MobileNet baseline tuning",
                "dataset": "cifar10",
                "model_name": "mobilenet_v2",
                "status": "active",
                "experiment_count": 3,
            }
        ],
    )


def has_running_experiment(runs: list[dict[str, Any]]) -> bool:
    """Return whether any run has a running experiment."""
    for run in runs:
        run_detail = request_json(f'/runs/{run["id"]}', {})
        for experiment in run_detail.get("experiments", []):
            if experiment.get("status") == "running":
                return True
    return False


def load_run_detail(run_id: str) -> dict[str, Any]:
    """Return one run detail."""
    return request_json(
        f"/runs/{run_id}",
        {
            "id": run_id,
            "name": "CIFAR-10 MobileNet baseline tuning",
            "dataset": "cifar10",
            "model_name": "mobilenet_v2",
            "status": "active",
            "notes": "Using Streamlit fallback demo data because backend is not reachable.",
            "baseline_experiment_id": "exp_demo_001",
            "best_experiment_id": "exp_demo_002",
            "frontier_experiment_id": "exp_demo_002",
            "experiments": [
                {"id": "exp_demo_001", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "keep", "is_best_so_far": False},
                {"id": "exp_demo_002", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "keep", "is_best_so_far": True},
                {"id": "exp_demo_003", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "discard", "is_best_so_far": False},
            ],
        },
    )


def load_run_summary(run_id: str) -> dict[str, Any]:
    """Return one run-level summary."""
    return request_json(
        f"/runs/{run_id}/summary",
        {
            "run_id": run_id,
            "baseline_experiment_id": "exp_demo_001",
            "best_experiment_id": "exp_demo_002",
            "frontier_experiment_id": "exp_demo_002",
            "keep_count": 2,
            "discard_count": 1,
            "crash_count": 0,
            "timeout_count": 0,
        },
    )


def load_run_metrics(run_id: str, metric_name: str) -> dict[str, Any]:
    """Return trend metrics for a run."""
    return request_json(
        f"/runs/{run_id}/metrics?metric_name={metric_name}",
        {
            "run_id": run_id,
            "metric_name": metric_name,
            "available_metrics": ["top1_acc", "val_loss", "train_loss"],
            "points": [
                {"experiment_id": "exp_demo_001", "experiment_index": 1, "metric_name": metric_name, "metric_value": 0.71},
                {"experiment_id": "exp_demo_002", "experiment_index": 2, "metric_name": metric_name, "metric_value": 0.78},
                {"experiment_id": "exp_demo_003", "experiment_index": 3, "metric_name": metric_name, "metric_value": 0.80},
            ],
        },
    )


def load_experiment_detail(experiment_id: str) -> dict[str, Any]:
    """Return one experiment detail."""
    return request_json(
        f"/experiments/{experiment_id}",
        {
            "id": experiment_id,
            "run_id": "run_demo_001",
            "status": "success",
            "decision": "keep",
            "decision_reason": "当前 run 下的最佳实验。",
            "baseline_experiment_id": "exp_demo_001",
            "is_best_so_far": experiment_id == "exp_demo_002",
            "config": {
                "task_type": "classification",
                "dataset": "cifar10",
                "model_family": "mobilenet",
                "model_name": "mobilenet_v2",
                "parameter_space_version": "mobilenet_v2@v1",
                "participates_in_ranking": True,
                "params": {
                    "optimizer": "adamw",
                    "learning_rate": 0.004,
                    "batch_size": 128,
                    "image_size": 64,
                    "epochs": 30,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "augmentation_level": "medium",
                    "label_smoothing": 0.08,
                    "aux_logits": None,
                },
            },
            "parameter_space": {
                "model_name": "mobilenet_v2",
                "version": "mobilenet_v2@v1",
                "editable_params": {},
            },
            "proposal": {
                "task_type": "classification",
                "model_name": "mobilenet_v2",
                "based_on_experiment_ids": ["exp_demo_001"],
                "hypothesis": "Slightly higher learning rate may improve early convergence.",
                "changes": {"learning_rate": 0.004, "label_smoothing": 0.08},
                "reason": "The previous baseline is still underfitting in early epochs.",
                "risk": "low",
            },
            "result": {
                "status": "success",
                "metrics": {"train_loss": 0.42, "val_loss": 0.51, "top1_acc": 0.78, "best_epoch": 24},
                "resource": {"gpu_memory_mb": 2100, "training_seconds": 320},
                "params": {
                    "optimizer": "adamw",
                    "learning_rate": 0.004,
                    "batch_size": 128,
                    "image_size": 64,
                    "epochs": 30,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "augmentation_level": "medium",
                    "label_smoothing": 0.08,
                    "aux_logits": None,
                },
                "artifacts": {
                    "log_path": "artifacts/logs/exp_demo_002.log",
                    "checkpoint_path": "artifacts/checkpoints/exp_demo_002.pt",
                },
            },
            "reflection": {
                "outcome": "improved",
                "analysis": "Higher learning rate improved convergence without obvious instability.",
                "confidence": 0.78,
                "next_action": "Explore nearby learning rates and keep cosine scheduler.",
                "recommended_changes": {"learning_rate": 0.005},
            },
        },
    )


def append_activity_log(message: str) -> None:
    """Append one timestamped activity message to session state."""
    activity_logs = st.session_state.setdefault("activity_logs", [])
    activity_logs.append(f'[{time.strftime("%H:%M:%S")}] {message}')
    st.session_state["activity_logs"] = activity_logs[-LOG_LIMIT:]
    refresh_activity_log_view()


def refresh_activity_log_view() -> None:
    """Refresh the live activity log area when available."""
    global LIVE_LOG_CONTAINER
    if LIVE_LOG_CONTAINER is None:
        return
    log_lines = st.session_state.get("activity_logs", [])
    log_text = "\n".join(log_lines[-20:]) if log_lines else "Logs will appear here once training starts."
    LIVE_LOG_CONTAINER.code(log_text, language=None, wrap_lines=True, height=220)


def format_metric_summary(experiment_detail: dict[str, Any]) -> str:
    """Build a compact metric summary for logs."""
    metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
    top1_acc = metrics.get("top1_acc")
    val_loss = metrics.get("val_loss")
    train_loss = metrics.get("train_loss")
    best_epoch = metrics.get("best_epoch")
    metric_parts = []
    if top1_acc is not None:
        metric_parts.append(f"top1_acc={top1_acc}")
    if val_loss is not None:
        metric_parts.append(f"val_loss={val_loss}")
    if train_loss is not None:
        metric_parts.append(f"train_loss={train_loss}")
    if best_epoch is not None:
        metric_parts.append(f"best_epoch={best_epoch}")
    return ", ".join(metric_parts) if metric_parts else "no metrics returned"


def format_proposal_changes(changes: dict[str, Any]) -> str:
    """Format one proposal change payload for display."""
    visible_changes = [
        f"{key}={value}"
        for key, value in changes.items()
        if key not in AI_BLOCKED_CHANGE_FIELDS and value is not None
    ]
    return ", ".join(visible_changes) if visible_changes else "无参数变更"


def get_best_experiment_id(run_id: str) -> str | None:
    """Return the best experiment id for one run."""
    best_experiment = get_best_experiment_detail(run_id)
    if best_experiment is None:
        return None
    return best_experiment.get("id")


def sanitize_ai_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Drop blocked AI changes before executing auto tuning."""
    return {
        key: value
        for key, value in changes.items()
        if key not in AI_BLOCKED_CHANGE_FIELDS and value is not None
    }


def clear_database_records() -> tuple[bool, Any]:
    """Clear all backend records."""
    return post_without_body("/runs/reset")


def clear_selected_run_records(run_id: str) -> tuple[bool, Any]:
    """Clear the currently selected run and all of its records."""
    return post_without_body(f"/runs/{run_id}/reset")


def stop_current_experiment(experiment_id: str) -> tuple[bool, Any]:
    """Stop the current running experiment immediately."""
    return post_without_body(f"/experiments/{experiment_id}/stop")


def reset_frontend_state_after_clear() -> None:
    """Clear session state that mirrors backend records."""
    preserved_keys = {
        "run_name",
        "dataset",
        "model_name",
        "optimizer",
        "learning_rate",
        "batch_size",
        "image_size",
        "epochs",
        "weight_decay",
        "scheduler",
        "augmentation_level",
        "label_smoothing",
        "aux_logits",
        "ai_test_rounds",
    }
    preserved_values = {key: st.session_state.get(key) for key in preserved_keys if key in st.session_state}
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.update(preserved_values)
    st.session_state["selected_run_id"] = "__all__"
    st.session_state["activity_logs"] = []


def store_ai_panel(payload: dict[str, Any]) -> None:
    """Store the latest AI panel payload."""
    st.session_state["ai_panel"] = payload
    refresh_ai_panel_view()


def build_result_snapshot(experiment_id: str, experiment_detail: dict[str, Any]) -> dict[str, Any]:
    """Build one compact result snapshot."""
    metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
    return {
        "experiment_id": experiment_id,
        "status": experiment_detail.get("status"),
        "metrics": metrics,
        "summary": format_metric_summary(experiment_detail),
    }


def store_manual_ai_suggestion(
    run_id: str,
    experiment_id: str,
    experiment_detail: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    """Store one post-train suggestion for manual train mode."""
    store_ai_panel(
        {
            "mode": "manual",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "result": build_result_snapshot(experiment_id, experiment_detail),
            "proposal": proposal,
        }
    )


def start_auto_train_summary(run_id: str, experiment_id: str, experiment_detail: dict[str, Any]) -> None:
    """Initialize auto-train summary."""
    st.session_state["auto_train_summary"] = {
        "mode": "auto",
        "run_id": run_id,
        "baseline": build_result_snapshot(experiment_id, experiment_detail),
        "rounds": [],
    }
    refresh_ai_panel_view()


def record_auto_train_round(
    round_index: int,
    experiment_id: str,
    proposal: dict[str, Any],
    experiment_detail: dict[str, Any],
) -> None:
    """Record one auto-train round."""
    summary = st.session_state.setdefault("auto_train_summary", {"mode": "auto", "rounds": []})
    summary.setdefault("rounds", []).append(
        {
            "round_index": round_index,
            "proposal": proposal,
            "result": build_result_snapshot(experiment_id, experiment_detail),
        }
    )
    st.session_state["auto_train_summary"] = summary
    refresh_ai_panel_view()


def finalize_auto_train_summary(final_proposal: dict[str, Any]) -> None:
    """Publish the auto-train summary to the AI panel."""
    summary = st.session_state.get("auto_train_summary")
    if not summary:
        return
    panel_payload = dict(summary)
    panel_payload["final_proposal"] = final_proposal
    store_ai_panel(panel_payload)


def refresh_ai_panel_view() -> None:
    """Refresh the live AI panel when available."""
    global LIVE_AI_PANEL_CONTAINER
    if LIVE_AI_PANEL_CONTAINER is None:
        return
    suggestion_payload = st.session_state.get("ai_panel") or st.session_state.get("auto_train_summary")
    with LIVE_AI_PANEL_CONTAINER.container():
        with st.container(border=True):
            if not suggestion_payload:
                st.caption("AI 建议和调优趋势会在训练过程中或结束后显示在这里。")
                st.code("等待训练输出...", language=None, wrap_lines=True, height=180)
                return
            if suggestion_payload.get("mode") == "auto":
                final_proposal = suggestion_payload.get("final_proposal")
                baseline = suggestion_payload.get("baseline", {})
                rounds = suggestion_payload.get("rounds", [])
                progress = st.session_state.get("auto_task_progress") or {}
                st.markdown("**自动训练总结**")
                if progress and progress.get("status") in {"queued", "running", "stopping"}:
                    completed_rounds = len(rounds)
                    current_round = progress.get("current_round", 0)
                    total_rounds = progress.get("total_rounds", 0)
                    current_experiment_id = progress.get("current_experiment_id") or "-"
                    st.caption(
                        f"进度：已完成 {completed_rounds}/{total_rounds} 轮，"
                        f"当前轮次 {current_round}/{total_rounds}，"
                        f"实验 {current_experiment_id}。"
                    )
                st.markdown(f"基线实验：`{baseline.get('experiment_id', '-')}`  ")
                st.caption(baseline.get("summary", ""))
                trend_rows = []
                baseline_metrics = baseline.get("metrics", {})
                trend_rows.append(
                    {
                        "round_index": 0,
                        "top1_acc": baseline_metrics.get("top1_acc"),
                        "val_loss": baseline_metrics.get("val_loss"),
                        "train_loss": baseline_metrics.get("train_loss"),
                    }
                )
                for round_info in rounds:
                    round_metrics = round_info["result"].get("metrics", {})
                    st.markdown(
                        f"第 {round_info['round_index']} 轮："
                        f"{format_proposal_changes(round_info['proposal']['changes'])}"
                    )
                    st.caption(
                        f"{round_info['result']['experiment_id']} | {round_info['result']['summary']}"
                    )
                    trend_rows.append(
                        {
                            "round_index": int(round_info["round_index"]),
                            "top1_acc": round_metrics.get("top1_acc"),
                            "val_loss": round_metrics.get("val_loss"),
                            "train_loss": round_metrics.get("train_loss"),
                        }
                    )
                if len(trend_rows) > 0:
                    st.markdown("**调优趋势**")
                    st.line_chart(
                        trend_rows,
                        x="round_index",
                        y=["top1_acc", "val_loss", "train_loss"],
                        use_container_width=True,
                    )
                if final_proposal:
                    st.markdown("**下一步建议**")
                    st.markdown(final_proposal["hypothesis"])
                    st.caption(final_proposal["reason"])
                    st.markdown(f"`{format_proposal_changes(final_proposal['changes'])}`")
                else:
                    st.caption("自动训练进行中。每完成一轮后，这里的趋势会自动更新。")
                return

            proposal = suggestion_payload["proposal"]
            result = suggestion_payload["result"]
            st.markdown("**单次训练建议**")
            st.caption(f"实验 {result['experiment_id']} | {result['summary']}")
            st.markdown(f"**建议**  \n{proposal['hypothesis']}")
            st.caption(proposal["reason"])
            st.markdown(f"**建议修改**  \n{format_proposal_changes(proposal['changes'])}")


def generate_and_store_ai_suggestion(
    run_id: str,
    experiment_id: str,
    experiment_detail: dict[str, Any],
    log_prefix: str,
) -> tuple[bool, dict[str, Any] | str]:
    """Generate one AI suggestion for a completed experiment and store it."""
    ok, proposal_response = generate_aihubmix_proposal_request(run_id)
    if not ok:
        error_message = f'AI suggestion failed: {proposal_response.get("detail", proposal_response)}'
        append_activity_log(error_message)
        return False, error_message
    append_activity_log(f"{log_prefix} AI suggestion: {proposal_response['hypothesis']}")
    append_activity_log(f"{log_prefix} Proposed changes: {format_proposal_changes(proposal_response['changes'])}")
    store_manual_ai_suggestion(run_id, experiment_id, experiment_detail, proposal_response)
    return True, proposal_response


def set_ui_locked(is_locked: bool) -> None:
    """Lock or unlock all train actions."""
    st.session_state["ui_locked"] = is_locked


def set_post_action_notice(message: str, level: str = "success") -> None:
    """Store one notice to be shown after the next rerun."""
    st.session_state["post_action_notice"] = {"message": message, "level": level}


def clear_training_state() -> None:
    """Clear all transient frontend training state after completion."""
    st.session_state["ui_locked"] = False
    st.session_state["active_train_control"] = None
    st.session_state["training_experiment_id"] = None
    st.session_state["last_running_experiment_id"] = None
    st.session_state["current_auto_task_id"] = None
    st.session_state["auto_task_progress"] = None
    st.session_state["skip_auto_poll_once"] = False


def sync_auto_train_task_state() -> str | None:
    """Sync frontend state from one backend auto-train task snapshot."""
    current_auto_task_id = st.session_state.get("current_auto_task_id")
    if not current_auto_task_id:
        return None

    st.session_state["active_train_control"] = "auto"
    ok, auto_task_response = load_auto_train_task(current_auto_task_id)
    if not ok or "task_id" not in auto_task_response:
        return None

    st.session_state["activity_logs"] = auto_task_response.get("logs", [])
    summary = auto_task_response.get("summary")
    if summary is not None:
        st.session_state["ai_panel"] = summary
    if auto_task_response.get("run_id"):
        st.session_state["selected_run_id"] = auto_task_response["run_id"]
    st.session_state["training_experiment_id"] = auto_task_response.get("current_experiment_id")
    st.session_state["auto_task_progress"] = {
        "status": auto_task_response.get("status"),
        "current_round": auto_task_response.get("current_round", 0),
        "total_rounds": auto_task_response.get("total_rounds", 0),
        "current_experiment_id": auto_task_response.get("current_experiment_id"),
    }

    auto_status = auto_task_response.get("status")
    if auto_status in {"queued", "running", "stopping"}:
        st.session_state["ui_locked"] = True
    return auto_status


@st.fragment(run_every="2s")
def render_live_training_monitor() -> None:
    """Keep the log and AI panel updated while training is active."""
    auto_task_id = st.session_state.get("current_auto_task_id")
    experiment_id = st.session_state.get("training_experiment_id")
    if not auto_task_id and not experiment_id:
        return

    if auto_task_id:
        auto_status = sync_auto_train_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()
        if auto_status in {"completed", "stopped", "failed"}:
            if auto_status == "completed":
                set_post_action_notice(
                    f"Auto Train finished for run {st.session_state.get('selected_run_id', '-')}.",
                    "success",
                )
            elif auto_status == "stopped":
                set_post_action_notice("Auto Train stopped and discarded the current experiment.", "success")
            elif auto_status == "failed":
                task_snapshot = request_json(
                    f"/runs/auto-train/{auto_task_id}",
                    {"error": "unknown error"},
                )
                set_post_action_notice(
                    f"Auto Train failed: {task_snapshot.get('error', 'unknown error')}",
                    "error",
                )
            clear_training_state()
            st.rerun()
        return

    experiment_detail = load_experiment_detail(experiment_id)
    if experiment_detail.get("status") in {"success", "failed", "discarded"}:
        st.rerun()


def queue_train_request(mode: str, payload: dict[str, Any], rounds: int = 1) -> None:
    """Queue one train request and force a rerun with locked UI."""
    if st.session_state.get("ui_locked") or st.session_state.get("pending_train_request"):
        return
    st.session_state["active_train_control"] = "manual" if mode == "manual" else "auto"
    st.session_state["pending_train_request"] = {
        "mode": mode,
        "payload": payload,
        "rounds": rounds,
        "selected_run_id": st.session_state.get("selected_run_id", "__all__"),
    }
    set_ui_locked(True)
    st.rerun()


def process_pending_train_request() -> None:
    """Execute one queued train request."""
    pending_request = st.session_state.get("pending_train_request")
    if not pending_request:
        return

    st.session_state["pending_train_request"] = None
    mode = pending_request["mode"]
    payload = pending_request["payload"]
    ai_rounds = int(pending_request.get("rounds", 1))
    selected_run_id = pending_request.get("selected_run_id", "__all__")

    if mode == "manual":
        append_activity_log("Manual train requested from the current parameter panel.")
        if selected_run_id == "__all__":
            ok, message = create_run_and_first_experiment(payload)
        else:
            payload["based_on_experiment_ids"] = [get_latest_experiment_detail(selected_run_id)["id"]] if get_latest_experiment_detail(selected_run_id) else []
            payload["proposal_hypothesis"] = "Manual follow-up experiment under the selected run."
            payload["proposal_reason"] = "Use the current parameter panel as the next structured experiment."
            ok, message = append_experiment_to_run(selected_run_id, payload)
        if ok:
            st.session_state["auto_train_mode"] = False
            set_post_action_notice(message, "success")
            st.rerun()
        else:
            set_ui_locked(False)
            append_activity_log(message)
            st.error(message)
        return

    append_activity_log(f"Auto Train requested for {ai_rounds} AI rounds.")
    st.session_state["auto_train_mode"] = True
    if selected_run_id == "__all__":
        ok, message = create_run_and_first_experiment(payload)
    else:
        payload["based_on_experiment_ids"] = [get_latest_experiment_detail(selected_run_id)["id"]] if get_latest_experiment_detail(selected_run_id) else []
        payload["proposal_hypothesis"] = "Auto Train baseline appended to the selected run."
        payload["proposal_reason"] = "Use the current parameter panel as the baseline before AI follow-up rounds."
        ok, message = append_experiment_to_run(selected_run_id, payload)
    if not ok:
        set_ui_locked(False)
        append_activity_log(message)
        st.error(message)
        return

    latest_run_id = st.session_state.get("selected_run_id", "")
    current_experiment_id = st.session_state.get("training_experiment_id")
    done, experiment_detail = wait_for_experiment_completion(current_experiment_id)
    if not done or experiment_detail.get("status") != "success":
        set_ui_locked(False)
        append_activity_log(f"Baseline experiment {current_experiment_id} failed.")
        st.error(experiment_detail.get("detail", f"Experiment {current_experiment_id} failed"))
        return

    start_auto_train_summary(latest_run_id, current_experiment_id, experiment_detail)
    append_activity_log(
        f"Baseline experiment {current_experiment_id} finished successfully with "
        f"{format_metric_summary(experiment_detail)}."
    )

    for round_index in range(ai_rounds):
        ok, proposal_response = generate_aihubmix_proposal_request(latest_run_id)
        if not ok:
            set_ui_locked(False)
            append_activity_log(f"Round {round_index + 1}: AI suggestion failed.")
            st.error(f'AIHubMix proposal failed: {proposal_response.get("detail", proposal_response)}')
            return
        append_activity_log(
            f"Round {round_index + 1}: AI suggested {proposal_response['hypothesis']}"
        )
        sanitized_changes = sanitize_ai_changes(proposal_response["changes"])
        append_activity_log(
            f"Round {round_index + 1}: applied changes {format_proposal_changes(sanitized_changes)}."
        )
        latest_experiment = get_latest_experiment_detail(latest_run_id)
        if latest_experiment is None:
            set_ui_locked(False)
            st.error("No experiment is available in the selected run for proposal generation.")
            return

        proposal = dict(proposal_response)
        proposal["config"] = {
            "task_type": latest_experiment["config"]["task_type"],
            "dataset": latest_experiment["config"]["dataset"],
            "model_family": latest_experiment["config"]["model_family"],
            "model_name": latest_experiment["config"]["model_name"],
            "parameter_space_version": latest_experiment["config"]["parameter_space_version"],
            "params": {
                **latest_experiment["config"]["params"],
                **sanitized_changes,
            },
        }
        round_payload = build_payload_from_form(
            {
                "run_name": st.session_state["run_name"],
                "dataset": proposal["config"]["dataset"],
                "model_name": proposal["config"]["model_name"],
                **proposal["config"]["params"],
                "proposal_hypothesis": proposal["hypothesis"],
                "proposal_reason": proposal["reason"],
                "based_on_experiment_ids": proposal["based_on_experiment_ids"],
            }
        )
        round_payload["proposal_changes"] = sanitized_changes
        ok, message = append_experiment_to_run(latest_run_id, round_payload)
        if not ok:
            set_ui_locked(False)
            append_activity_log(message)
            st.error(message)
            return

        current_experiment_id = st.session_state.get("training_experiment_id")
        done, experiment_detail = wait_for_experiment_completion(current_experiment_id)
        if not done or experiment_detail.get("status") != "success":
            set_ui_locked(False)
            append_activity_log(f"Round {round_index + 1}: experiment {current_experiment_id} failed.")
            st.error(experiment_detail.get("detail", f"Experiment {current_experiment_id} failed"))
            return

        record_auto_train_round(round_index + 1, current_experiment_id, proposal_response, experiment_detail)
        append_activity_log(
            f"Round {round_index + 1}: experiment {current_experiment_id} finished with "
            f"{format_metric_summary(experiment_detail)}."
        )

    ok, suggestion_message = generate_and_store_ai_suggestion(
        latest_run_id,
        current_experiment_id,
        experiment_detail,
        "Final",
    )
    if not ok:
        append_activity_log(suggestion_message)
    else:
        finalize_auto_train_summary(suggestion_message)

    st.session_state["selected_experiment_id"] = current_experiment_id
    st.session_state["last_finished_experiment_id"] = current_experiment_id
    st.session_state["training_experiment_id"] = None
    st.session_state["last_running_experiment_id"] = None
    set_ui_locked(False)
    set_post_action_notice(f"Auto Train finished for run {latest_run_id}.", "success")
    st.rerun()


def build_experiment_comparison_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build comparison rows for one run."""
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        detail = load_experiment_detail(experiment["id"])
        result = detail.get("result") or {}
        metrics = result.get("metrics") or {}
        params = detail.get("config", {}).get("params", {})
        rows.append(
            {
                "selected": True,
                "experiment_id": experiment["id"],
                "decision": detail.get("decision"),
                "anchor": "",
                "status": experiment["status"],
                "top1_acc": metrics.get("top1_acc"),
                "val_loss": metrics.get("val_loss"),
                "train_loss": metrics.get("train_loss"),
                "best_epoch": metrics.get("best_epoch"),
                "optimizer": params.get("optimizer"),
                "learning_rate": params.get("learning_rate"),
                "batch_size": params.get("batch_size"),
                "image_size": params.get("image_size"),
                "epochs": params.get("epochs"),
                "scheduler": params.get("scheduler"),
            }
        )
    return rows


def build_all_training_records(runs: list[dict[str, Any]], selected_run_id: str) -> list[dict[str, Any]]:
    """Build training records for one run or all runs."""
    records: list[dict[str, Any]] = []
    candidate_runs = runs if selected_run_id == "__all__" else [run for run in runs if run["id"] == selected_run_id]
    for run in candidate_runs:
        run_detail = load_run_detail(run["id"])
        run_summary = load_run_summary(run["id"])
        for experiment in run_detail.get("experiments", []):
            detail = load_experiment_detail(experiment["id"])
            result = detail.get("result") or {}
            metrics = result.get("metrics") or {}
            params = detail.get("config", {}).get("params", {})
            anchor_labels = []
            if experiment["id"] == run_summary.get("baseline_experiment_id"):
                anchor_labels.append("baseline")
            if experiment["id"] == run_summary.get("best_experiment_id"):
                anchor_labels.append("best")
            if experiment["id"] == run_summary.get("frontier_experiment_id"):
                anchor_labels.append("frontier")
            records.append(
                {
                    "selected": True,
                    "run_id": run["id"],
                    "run_name": run["name"],
                    "experiment_id": experiment["id"],
                    "anchor": "/".join(anchor_labels),
                    "decision": detail.get("decision"),
                    "status": experiment["status"],
                    "model_name": run["model_name"],
                    "dataset": run["dataset"],
                    "top1_acc": metrics.get("top1_acc"),
                    "val_loss": metrics.get("val_loss"),
                    "train_loss": metrics.get("train_loss"),
                    "best_epoch": metrics.get("best_epoch"),
                    "optimizer": params.get("optimizer"),
                    "learning_rate": params.get("learning_rate"),
                    "batch_size": params.get("batch_size"),
                    "image_size": params.get("image_size"),
                    "epochs": params.get("epochs"),
                    "scheduler": params.get("scheduler"),
                }
            )
    return records


def get_latest_experiment_detail(run_id: str) -> dict[str, Any] | None:
    """Return the latest experiment detail for one run."""
    run_detail = load_run_detail(run_id)
    experiments = run_detail.get("experiments", [])
    if not experiments:
        return None
    latest_experiment = experiments[-1]
    return load_experiment_detail(latest_experiment["id"])


def get_best_experiment_detail(run_id: str) -> dict[str, Any] | None:
    """Return the best finished experiment detail for one run."""
    run_summary = load_run_summary(run_id)
    best_experiment_id = run_summary.get("best_experiment_id")
    if best_experiment_id:
        return load_experiment_detail(best_experiment_id)

    run_detail = load_run_detail(run_id)
    experiments = run_detail.get("experiments", [])
    if not experiments:
        return None

    best_detail: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    for index, experiment in enumerate(experiments):
        detail = load_experiment_detail(experiment["id"])
        if detail.get("status") != "success":
            continue
        metrics = (detail.get("result") or {}).get("metrics") or {}
        top1_acc = metrics.get("top1_acc")
        val_loss = metrics.get("val_loss")
        ranking_key = (
            float(top1_acc) if top1_acc is not None else float("-inf"),
            -float(val_loss) if val_loss is not None else float("-inf"),
            float(index),
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_detail = detail

    return best_detail or get_latest_experiment_detail(run_id)


def generate_fake_llm_proposal(run_id: str) -> dict[str, Any] | None:
    """Generate a deterministic fake proposal from the latest experiment."""
    latest_experiment = get_latest_experiment_detail(run_id)
    if latest_experiment is None:
        return None

    config = latest_experiment["config"]
    params = dict(config["params"])
    latest_experiment_id = latest_experiment["id"]
    updated_params = dict(params)

    current_learning_rate = float(params["learning_rate"])
    next_learning_rate = round(min(current_learning_rate * 1.2, 0.01), 4)
    updated_params["learning_rate"] = next_learning_rate

    current_label_smoothing = float(params["label_smoothing"])
    next_label_smoothing = round(min(current_label_smoothing + 0.02, 0.2), 2)
    updated_params["label_smoothing"] = next_label_smoothing

    proposal = {
        "task_type": "classification",
        "model_name": config["model_name"],
        "based_on_experiment_ids": [latest_experiment_id],
        "hypothesis": "适度提高学习率并增加标签平滑，可能改善早期收敛。",
        "changes": {
            "learning_rate": next_learning_rate,
            "label_smoothing": next_label_smoothing,
        },
        "reason": "基于上一轮实验结果，继续围绕收敛速度和泛化能力做小步调整。",
        "risk": "low",
        "config": {
            "task_type": config["task_type"],
            "dataset": config["dataset"],
            "model_family": config["model_family"],
            "model_name": config["model_name"],
            "parameter_space_version": config["parameter_space_version"],
            "participates_in_ranking": config.get("participates_in_ranking", True),
            "params": updated_params,
        },
    }
    return proposal


def apply_generated_proposal(run_id: str, proposal: dict[str, Any]) -> None:
    """Apply one generated proposal into the append-experiment form."""
    generated_params = proposal["config"]["params"]
    st.session_state["append_run_id"] = run_id
    st.session_state["model_name"] = proposal["config"]["model_name"]
    st.session_state["optimizer"] = generated_params["optimizer"]
    st.session_state["learning_rate"] = generated_params["learning_rate"]
    st.session_state["batch_size"] = generated_params["batch_size"]
    st.session_state["image_size"] = generated_params["image_size"]
    st.session_state["epochs"] = generated_params["epochs"]
    st.session_state["weight_decay"] = generated_params["weight_decay"]
    st.session_state["scheduler"] = generated_params["scheduler"]
    st.session_state["augmentation_level"] = generated_params["augmentation_level"]
    st.session_state["label_smoothing"] = generated_params["label_smoothing"]
    st.session_state["aux_logits"] = generated_params["aux_logits"] if generated_params["aux_logits"] is not None else False
    st.session_state["participates_in_ranking"] = proposal["config"].get("participates_in_ranking", True)
    st.session_state["proposal_hypothesis"] = proposal["hypothesis"]
    st.session_state["proposal_reason"] = proposal["reason"]
    st.session_state["proposal_changes"] = proposal["changes"]
    st.session_state["based_on_experiment_ids"] = proposal["based_on_experiment_ids"]
    st.session_state["action_mode"] = "append_experiment"


def load_reference_config(selected_run_id: str) -> dict[str, Any]:
    """Load the latest config for the selected run or return defaults."""
    default_config = {
        "run_name": "CIFAR-10 baseline study",
        "dataset": "cifar10",
        "model_name": "mobilenet_v2",
        "participates_in_ranking": True,
        "params": {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 128,
            "image_size": 64,
            "epochs": 10,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "augmentation_level": "medium",
            "label_smoothing": 0.1,
            "aux_logits": False,
        },
    }
    if selected_run_id == "__all__":
        return default_config
    latest_experiment = get_latest_experiment_detail(selected_run_id)
    if latest_experiment is None:
        return default_config
    run_detail = load_run_detail(selected_run_id)
    return {
        "run_name": run_detail["name"],
        "dataset": latest_experiment["config"]["dataset"],
        "model_name": latest_experiment["config"]["model_name"],
        "participates_in_ranking": latest_experiment["config"].get("participates_in_ranking", True),
        "params": latest_experiment["config"]["params"],
    }


def build_payload_from_form(form_values: dict[str, Any]) -> dict[str, Any]:
    """Build backend payload from current form values."""
    model_name = form_values["model_name"]
    model_config = SUPPORTED_MODELS[model_name]
    params = {
        "optimizer": form_values["optimizer"],
        "learning_rate": float(form_values["learning_rate"]),
        "batch_size": int(form_values["batch_size"]),
        "image_size": int(form_values["image_size"]),
        "epochs": int(form_values["epochs"]),
        "weight_decay": float(form_values["weight_decay"]),
        "scheduler": form_values["scheduler"],
        "augmentation_level": form_values["augmentation_level"],
        "label_smoothing": float(form_values["label_smoothing"]),
        "aux_logits": bool(form_values["aux_logits"]) if model_name == "googlenet" else None,
    }
    return {
        "name": form_values["run_name"],
        "dataset": form_values["dataset"],
        "model_name": model_name,
        "notes": form_values.get("notes", ""),
        "proposal_hypothesis": form_values.get("proposal_hypothesis", ""),
        "proposal_reason": form_values.get("proposal_reason", ""),
        "based_on_experiment_ids": form_values.get("based_on_experiment_ids", []),
        "proposal_changes": params,
        "config": {
            "task_type": "classification",
            "dataset": form_values["dataset"],
            "model_family": model_config["model_family"],
            "model_name": model_name,
            "parameter_space_version": model_config["parameter_space_version"],
            "participates_in_ranking": bool(form_values.get("participates_in_ranking", True)),
            "params": params,
        },
        "parameter_space": model_config["parameter_space"],
    }


def create_run_and_first_experiment(payload: dict[str, Any]) -> tuple[bool, str]:
    """Create a run, then create and train its first experiment."""
    run_request = {
        "name": payload["name"],
        "dataset": payload["dataset"],
        "model_name": payload["model_name"],
        "base_config": payload["config"],
        "notes": payload["notes"],
    }
    ok, run_response = post_json("/runs", run_request)
    if not ok:
        return False, f'Create run failed: {run_response.get("detail", run_response)}'
    append_activity_log(f"Created run {run_response['id']} ({payload['model_name']} on {payload['dataset']}).")

    experiment_request = {
        "run_id": run_response["id"],
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "proposal": None,
    }
    ok, experiment_response = post_json("/experiments", experiment_request)
    if not ok:
        return False, f'Create experiment failed: {experiment_response.get("detail", experiment_response)}'
    append_activity_log(f"Created experiment {experiment_response['id']} as the baseline trial.")

    ok, train_response = post_without_body(f'/experiments/{experiment_response["id"]}/train')
    if not ok:
        return False, f'Train experiment failed: {train_response.get("detail", train_response)}'
    append_activity_log(f"Started training experiment {experiment_response['id']}.")

    st.session_state["selected_run_id"] = run_response["id"]
    st.session_state["selected_experiment_id"] = experiment_response["id"]
    st.session_state["training_experiment_id"] = experiment_response["id"]
    return True, f'Started training for run {run_response["id"]} and experiment {experiment_response["id"]}.'


def append_experiment_to_run(run_id: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Append one experiment to an existing run and start training."""
    experiment_request = {
        "run_id": run_id,
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "proposal": {
            "task_type": "classification",
            "model_name": payload["model_name"],
            "based_on_experiment_ids": payload["based_on_experiment_ids"],
            "hypothesis": payload["proposal_hypothesis"],
            "changes": payload["proposal_changes"],
            "reason": payload["proposal_reason"],
            "risk": "low",
        },
    }
    ok, experiment_response = post_json("/experiments", experiment_request)
    if not ok:
        return False, f'Append experiment failed: {experiment_response.get("detail", experiment_response)}'
    append_activity_log(
        f"Appended experiment {experiment_response['id']} to run {run_id} with changes: "
        f"{format_proposal_changes(payload['proposal_changes'])}."
    )

    ok, train_response = post_without_body(f'/experiments/{experiment_response["id"]}/train')
    if not ok:
        return False, f'Train experiment failed: {train_response.get("detail", train_response)}'
    append_activity_log(f"Started training experiment {experiment_response['id']}.")

    st.session_state["selected_run_id"] = run_id
    st.session_state["selected_experiment_id"] = experiment_response["id"]
    st.session_state["training_experiment_id"] = experiment_response["id"]
    return True, f'Appended and started experiment {experiment_response["id"]} under run {run_id}.'


def render_control_panel(runs: list[dict[str, Any]], selected_run_id: str, is_training_active: bool) -> None:
    """Render the left-side parameter and action panel."""
    st.subheader("Train")
    st.caption("左侧用于训练。`Train` 会按当前参数启动一次实验并在结束后自动生成建议；`Auto Train` 会自动连续训练、自动采纳 AI 建议，不再人工确认。")
    if is_training_active:
        st.warning("A training job is running. Actions are temporarily locked.")

    reference_config = load_reference_config(selected_run_id)
    if st.session_state.get("form_reference_run") != selected_run_id:
        st.session_state["run_name"] = reference_config["run_name"]
        st.session_state["dataset"] = reference_config["dataset"]
        st.session_state["model_name"] = reference_config["model_name"]
        st.session_state["optimizer"] = reference_config["params"]["optimizer"]
        st.session_state["learning_rate"] = reference_config["params"]["learning_rate"]
        st.session_state["batch_size"] = reference_config["params"]["batch_size"]
        st.session_state["image_size"] = reference_config["params"]["image_size"]
        st.session_state["epochs"] = reference_config["params"]["epochs"]
        st.session_state["weight_decay"] = reference_config["params"]["weight_decay"]
        st.session_state["scheduler"] = reference_config["params"]["scheduler"]
        st.session_state["augmentation_level"] = reference_config["params"]["augmentation_level"]
        st.session_state["label_smoothing"] = reference_config["params"]["label_smoothing"]
        st.session_state["aux_logits"] = bool(reference_config["params"].get("aux_logits") or False)
        st.session_state["participates_in_ranking"] = reference_config.get("participates_in_ranking", True)
        st.session_state["based_on_experiment_ids"] = []
        st.session_state["form_reference_run"] = selected_run_id

    with st.container(border=True):
        st.markdown("**Training Setup**")
        st.caption("选择 run 后，这里的参数会自动带入该 run 最新一次实验，便于继续迭代。")
        top_left, top_mid, top_right = st.columns(3)
        with top_left:
            st.text_input("Run Name", key="run_name")
        with top_mid:
            st.selectbox("Dataset", options=["cifar10"], index=0, key="dataset")
        with top_right:
            model_name = st.selectbox("Model", options=["mobilenet_v2", "googlenet"], key="model_name")

        row_one = st.columns(3)
        with row_one[0]:
            st.selectbox("Optimizer", options=["sgd", "adam", "adamw"], key="optimizer")
        with row_one[1]:
            st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, step=0.0001, format="%.4f", key="learning_rate")
        with row_one[2]:
            st.selectbox("Batch Size", options=[32, 64, 128, 256], key="batch_size")

        row_two = st.columns(3)
        with row_two[0]:
            st.selectbox("Image Size", options=[32, 64, 96], key="image_size")
        with row_two[1]:
            st.selectbox("Epochs", options=[10, 20, 30, 50], key="epochs")
        with row_two[2]:
            st.number_input("Weight Decay", min_value=0.0, max_value=0.01, step=0.0001, format="%.4f", key="weight_decay")

        row_three = st.columns(3)
        with row_three[0]:
            st.selectbox("Scheduler", options=["none", "step", "cosine"], key="scheduler")
        with row_three[1]:
            st.selectbox("Augmentation Level", options=["low", "medium", "high"], key="augmentation_level")
        with row_three[2]:
            st.number_input("Label Smoothing", min_value=0.0, max_value=0.2, step=0.01, format="%.2f", key="label_smoothing")

        footer_left, footer_right = st.columns([1, 1])
        with footer_left:
            st.checkbox("Enable aux_logits", disabled=model_name != "googlenet", key="aux_logits")
        with footer_right:
            ai_rounds = st.number_input("Auto Train Rounds", min_value=1, max_value=20, value=10, step=1, key="ai_test_rounds")
        st.checkbox("Rank This Experiment", key="participates_in_ranking")

    payload = build_payload_from_form(
        {
            "run_name": st.session_state["run_name"],
            "dataset": st.session_state["dataset"],
            "model_name": st.session_state["model_name"],
            "optimizer": st.session_state["optimizer"],
            "learning_rate": st.session_state["learning_rate"],
            "batch_size": st.session_state["batch_size"],
            "image_size": st.session_state["image_size"],
            "epochs": st.session_state["epochs"],
            "weight_decay": st.session_state["weight_decay"],
            "scheduler": st.session_state["scheduler"],
            "augmentation_level": st.session_state["augmentation_level"],
            "label_smoothing": st.session_state["label_smoothing"],
            "aux_logits": st.session_state["aux_logits"],
            "participates_in_ranking": st.session_state["participates_in_ranking"],
            "based_on_experiment_ids": st.session_state.get("based_on_experiment_ids", []),
        }
    )

    is_appending_to_existing_run = selected_run_id != "__all__"
    train_button_label = "Append Train" if is_appending_to_existing_run else "Train"
    auto_train_button_label = "Append Auto Train" if is_appending_to_existing_run else "Auto Train"
    active_train_control = st.session_state.get("active_train_control")
    action_left, action_right = st.columns(2)
    with action_left:
        left_label = "Stop Training" if is_training_active and active_train_control == "manual" else train_button_label
        left_disabled = is_training_active and active_train_control != "manual"
        if st.button(left_label, disabled=left_disabled, use_container_width=True):
            if is_training_active and active_train_control == "manual":
                running_experiment_id = st.session_state.get("training_experiment_id") or find_running_experiment_id(runs)
                if running_experiment_id:
                    ok, response = stop_current_experiment(running_experiment_id)
                    if ok:
                        clear_training_state()
                        append_activity_log(f"Experiment {running_experiment_id} was stopped and discarded.")
                        set_post_action_notice(f"Stopped and discarded experiment {running_experiment_id}.", "success")
                        st.rerun()
                    else:
                        st.error(f'Stop training failed: {response.get("detail", response)}')
            else:
                queue_train_request("manual", payload, rounds=1)
    with action_right:
        right_label = "Stop Training" if is_training_active and active_train_control == "auto" else auto_train_button_label
        right_disabled = is_training_active and active_train_control != "auto"
        if st.button(right_label, disabled=right_disabled, use_container_width=True):
            if is_training_active and active_train_control == "auto":
                current_auto_task_id = st.session_state.get("current_auto_task_id")
                if current_auto_task_id:
                    ok, response = stop_auto_train_task_request(current_auto_task_id)
                    if ok:
                        append_activity_log(f"Stopping auto train task {current_auto_task_id}...")
                        st.rerun()
                    else:
                        st.error(f'Stop training failed: {response.get("detail", response)}')
            else:
                st.session_state["active_train_control"] = "auto"
                ok, response = start_auto_train_task_request(payload, selected_run_id, int(ai_rounds))
                if ok:
                    st.session_state["current_auto_task_id"] = response["task_id"]
                    st.session_state["ui_locked"] = True
                    st.session_state["skip_auto_poll_once"] = True
                    append_activity_log(f"Auto Train task started: {response['task_id']}")
                    st.rerun()
                else:
                    st.session_state["active_train_control"] = None
                    st.error(f'Auto Train failed to start: {response.get("detail", response)}')


def render_activity_log() -> None:
    """Render the training activity log."""
    global LIVE_LOG_CONTAINER
    st.subheader("Execution Log")
    LIVE_LOG_CONTAINER = st.empty()
    refresh_activity_log_view()


def render_ai_suggestion_panel() -> None:
    """Render the latest AI suggestion card."""
    global LIVE_AI_PANEL_CONTAINER
    st.subheader("AI 建议")
    LIVE_AI_PANEL_CONTAINER = st.empty()
    refresh_ai_panel_view()


def render_run_list(runs: list[dict[str, Any]]) -> str:
    """Render run list and return selected run id."""
    st.subheader("Run Selector")
    clear_disabled = bool(st.session_state.get("ui_locked"))
    if not runs:
        if st.button("Clear All", type="secondary", disabled=clear_disabled):
            ok, response = clear_database_records()
            if ok:
                reset_frontend_state_after_clear()
                st.success(
                    "Database cleared: "
                    f"runs={response.get('deleted_runs', 0)}, "
                    f"experiments={response.get('deleted_experiments', 0)}, "
                    f"results={response.get('deleted_results', 0)}"
                )
                st.rerun()
            else:
                st.error(f'Clear database failed: {response.get("detail", response)}')
        st.info("No runs found. You can still start a new training run from the left panel.")
        st.session_state["selected_run_id"] = "__all__"
        return "__all__"

    run_options = ["__all__"] + [run["id"] for run in runs]
    default_run_id = st.session_state.get("selected_run_id", runs[0]["id"])
    selected_run_index = next((index for index, run_id in enumerate(run_options) if run_id == default_run_id), 0)
    selector_column, clear_column = st.columns([6, 1])
    with selector_column:
        selected_run_id = st.selectbox(
            "Select Run",
            options=run_options,
            index=selected_run_index,
            format_func=lambda run_id: "All Runs" if run_id == "__all__" else next(run["name"] for run in runs if run["id"] == run_id),
        )
    with clear_column:
        st.caption(" ")
        if selected_run_id != "__all__":
            if st.button("Clear", type="secondary", disabled=clear_disabled):
                ok, response = clear_selected_run_records(selected_run_id)
                if ok:
                    reset_frontend_state_after_clear()
                    st.success(
                        "Run cleared: "
                        f"experiments={response.get('deleted_experiments', 0)}, "
                        f"results={response.get('deleted_results', 0)}"
                    )
                    st.rerun()
                else:
                    st.error(f'Clear run failed: {response.get("detail", response)}')

    st.session_state["selected_run_id"] = selected_run_id
    return selected_run_id


def render_training_records_workspace(runs: list[dict[str, Any]], selected_run_id: str) -> None:
    """Render a single-table training records workspace."""
    st.subheader("Training Records")
    all_rows = build_all_training_records(runs, selected_run_id)
    if not all_rows:
        st.info("No training records yet.")
        return

    best_experiment_ids: set[str] = set()
    baseline_experiment_ids: set[str] = set()
    frontier_experiment_ids: set[str] = set()
    if selected_run_id == "__all__":
        for run in runs:
            summary = load_run_summary(run["id"])
            if summary.get("best_experiment_id"):
                best_experiment_ids.add(summary["best_experiment_id"])
            if summary.get("baseline_experiment_id"):
                baseline_experiment_ids.add(summary["baseline_experiment_id"])
            if summary.get("frontier_experiment_id"):
                frontier_experiment_ids.add(summary["frontier_experiment_id"])
    else:
        summary = load_run_summary(selected_run_id)
        if summary.get("best_experiment_id"):
            best_experiment_ids.add(summary["best_experiment_id"])
        if summary.get("baseline_experiment_id"):
            baseline_experiment_ids.add(summary["baseline_experiment_id"])
        if summary.get("frontier_experiment_id"):
            frontier_experiment_ids.add(summary["frontier_experiment_id"])

    selection_key = "training_records_selected_experiments"
    selected_experiment_ids = st.session_state.get(selection_key, [row["experiment_id"] for row in all_rows])
    editor_rows = []
    for row in all_rows:
        editor_row = dict(row)
        editor_row["selected"] = row["experiment_id"] in selected_experiment_ids
        editor_row["best"] = "最佳" if row["experiment_id"] in best_experiment_ids else ""
        if row["experiment_id"] in baseline_experiment_ids:
            editor_row["anchor"] = f"{editor_row.get('anchor', '')}/baseline".strip("/")
        if row["experiment_id"] in frontier_experiment_ids:
            editor_row["anchor"] = f"{editor_row.get('anchor', '')}/frontier".strip("/")
        editor_rows.append(editor_row)

    edited_rows = st.data_editor(
        editor_rows,
        use_container_width=True,
        hide_index=True,
        key="training_records_editor",
        column_config={
            "selected": st.column_config.CheckboxColumn("Compare", help="Include this experiment in the chart comparison"),
            "best": st.column_config.TextColumn("最佳"),
            "anchor": st.column_config.TextColumn("锚点"),
            "decision": st.column_config.TextColumn("决策"),
        },
        disabled=[
            "best",
            "anchor",
            "decision",
            "run_id",
            "run_name",
            "experiment_id",
            "status",
            "model_name",
            "dataset",
            "top1_acc",
            "val_loss",
            "train_loss",
            "best_epoch",
            "optimizer",
            "learning_rate",
            "batch_size",
            "image_size",
            "epochs",
            "scheduler",
        ],
    )
    selected_experiment_ids = [row["experiment_id"] for row in edited_rows if row["selected"]]
    st.session_state[selection_key] = selected_experiment_ids
    if not selected_experiment_ids:
        st.warning("Select at least one training record to compare in the chart.")
        return

    if selected_run_id != "__all__":
        st.markdown("**Metrics Trend**")
        metric_seed = load_run_metrics(selected_run_id, "top1_acc")
        available_metrics = metric_seed.get("available_metrics", ["top1_acc", "val_loss", "train_loss"])
        metric_name = st.selectbox("Metric", available_metrics)
        metrics = metric_seed if metric_name == metric_seed.get("metric_name") else load_run_metrics(selected_run_id, metric_name)
        filtered_points = [
            point for point in metrics["points"] if point["experiment_id"] in selected_experiment_ids
        ]
        if filtered_points:
            st.line_chart(
                [
                    {
                        "experiment_index": point["experiment_index"],
                        metric_name: point["metric_value"],
                    }
                    for point in filtered_points
                ],
                x="experiment_index",
                y=metric_name,
                use_container_width=True,
            )
        else:
            st.info("The selected training records do not have visible metric points yet.")
    else:
        st.caption("Metrics trend is shown only when a single run is selected.")


def render_result_workspace(runs: list[dict[str, Any]], selected_run_id: str) -> None:
    """Render the right-side result workspace."""
    st.subheader("Results")
    selected_experiment_id = st.session_state.get("selected_experiment_id")
    best_experiment_detail = None
    run_summary = None
    if selected_run_id not in {"", "__all__"}:
        run_summary = load_run_summary(selected_run_id)
        best_experiment_detail = get_best_experiment_detail(selected_run_id)
        if not selected_experiment_id and best_experiment_detail is not None:
            selected_experiment_id = best_experiment_detail["id"]
            st.session_state["selected_experiment_id"] = selected_experiment_id

    inspected_experiment_detail = load_experiment_detail(selected_experiment_id) if selected_experiment_id else None
    summary_experiment_detail = best_experiment_detail or inspected_experiment_detail

    if summary_experiment_detail:
        result = summary_experiment_detail.get("result") or {}
        metrics = result.get("metrics") or {}
        metric_columns = st.columns(4)
        metric_columns[0].metric("Status", summary_experiment_detail.get("status", "-"))
        metric_columns[1].metric("Top1 Acc", metrics.get("top1_acc", "-"))
        metric_columns[2].metric("Val Loss", metrics.get("val_loss", "-"))
        metric_columns[3].metric("Train Loss", metrics.get("train_loss", "-"))
        if run_summary is not None:
            st.caption(
                " / ".join(
                    [
                        f"baseline={run_summary.get('baseline_experiment_id') or '-'}",
                        f"best={run_summary.get('best_experiment_id') or '-'}",
                        f"frontier={run_summary.get('frontier_experiment_id') or '-'}",
                    ]
                )
            )

        with st.container(border=True):
            title = "最佳实验详情" if best_experiment_detail is not None else "实验详情"
            st.markdown(f"**{title}**")
            st.json(
                {
                    "experiment_id": summary_experiment_detail.get("id"),
                    "run_id": summary_experiment_detail.get("run_id"),
                    "decision": summary_experiment_detail.get("decision"),
                    "decision_reason": summary_experiment_detail.get("decision_reason"),
                    "baseline_experiment_id": summary_experiment_detail.get("baseline_experiment_id"),
                    "is_best_so_far": summary_experiment_detail.get("is_best_so_far"),
                    "config": summary_experiment_detail.get("config"),
                    "result": summary_experiment_detail.get("result"),
                    "reflection": summary_experiment_detail.get("reflection"),
                },
                expanded=False,
            )
    else:
        st.caption("Select or create a run to inspect results.")

    render_ai_suggestion_panel()
    render_training_records_workspace(runs, selected_run_id)


def main() -> None:
    """Render the Streamlit demo UI."""
    st.set_page_config(page_title="AutoVisionLab Demo", layout="wide")
    st.title("AutoVisionLab Demo")
    st.caption("左侧训练，右侧看结果。`Train` 单次执行后给建议；`Auto Train` 自动连续调优并总结本轮优化结果。")
    post_action_notice = st.session_state.pop("post_action_notice", None)
    if post_action_notice:
        if post_action_notice["level"] == "success":
            st.success(post_action_notice["message"])
        else:
            st.error(post_action_notice["message"])

    sync_auto_train_task_state()

    runs = load_runs()
    is_training_active = has_running_experiment(runs) or bool(st.session_state.get("ui_locked"))
    if is_training_active and st.session_state.get("active_train_control") is None:
        st.session_state["active_train_control"] = "manual"
    left_column, right_column = st.columns([1, 1.45], gap="large")

    with left_column:
        selected_run_id = render_run_list(runs)
        render_control_panel(runs, selected_run_id, is_training_active)
        render_activity_log()

    with right_column:
        render_result_workspace(runs, selected_run_id)

    render_live_training_monitor()
    process_pending_train_request()

    training_experiment_id = st.session_state.get("training_experiment_id")
    if training_experiment_id:
        training_experiment_detail = load_experiment_detail(training_experiment_id)
        training_status = training_experiment_detail.get("status")
        if training_status == "running":
            if st.session_state.get("last_running_experiment_id") != training_experiment_id:
                append_activity_log(f"Experiment {training_experiment_id} is running.")
                st.session_state["last_running_experiment_id"] = training_experiment_id
        elif st.session_state.get("last_finished_experiment_id") != training_experiment_id:
            st.session_state["last_finished_experiment_id"] = training_experiment_id
            st.session_state["selected_experiment_id"] = training_experiment_id
            clear_training_state()
            if training_status == "discarded":
                append_activity_log(f"Experiment {training_experiment_id} was discarded.")
                st.warning(f"Training stopped and discarded: {training_experiment_id}")
                return
            append_activity_log(
                f"Experiment {training_experiment_id} finished with status {training_status} "
                f"and {format_metric_summary(training_experiment_detail)}."
            )
            completed_run_id = training_experiment_detail.get("run_id")
            if completed_run_id:
                ok, suggestion_message = generate_and_store_ai_suggestion(
                    completed_run_id,
                    training_experiment_id,
                    training_experiment_detail,
                    "Post-train",
                )
                if not ok:
                    append_activity_log(suggestion_message)
            st.success(f"Training finished with status: {training_status}")


if __name__ == "__main__":
    main()

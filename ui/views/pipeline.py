"""
Pipeline — the actions no existing tool provides (trigger data versioning,
training, CI gate). Viewing/inspection lives in MLflow / MinIO / Grafana / W&B
(see the Links page).

Uses a segmented_control (not st.tabs) to pick the section so ONLY the selected
section's code runs. st.tabs renders every tab's code each run, which would make
the Training auto-stream loop rerun the whole page even while viewing Data/CI-CD.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse

import streamlit as st

from api import api_get, api_post

# Streamlit puts ui/ on sys.path (for `from api import ...`); add the repo root too
# so the PDF-ingest path can import the client-side pipeline (src.data.pdf_versioning).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_PROG_RE = re.compile(r"\d+%\|")  # tqdm progress-bar marker


def _api_host() -> str:
    return urlparse(st.session_state.get("api_base", "")).hostname or "localhost"


def _run_pdf_ingest(uploaded, inf_url, n, dpi) -> None:
    """Client-side: PDFs → inference server → data version(s). Runs in-process
    (Streamlit is local), so it needs MinIO/MLflow env + pymupdf installed here.
    Version is auto-resolved by the pipeline counter; benchmark leakage guard is
    always on."""
    from dotenv import load_dotenv
    load_dotenv()
    host = _api_host()
    os.environ.setdefault("MINIO_ENDPOINT", f"http://{host}:9000")
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"http://{host}:5000")
    missing = [k for k in ("MINIO_ACCESS_KEY", "MINIO_SECRET_KEY") if not os.environ.get(k)]
    if missing:
        st.error(f"Thiếu trong .env: {', '.join(missing)} — cần để ghi MinIO.")
        return
    try:
        from src.data.pdf_versioning import create_version_from_pdfs
    except ImportError as exc:  # noqa: BLE001
        st.error(f"Thiếu dependency: {exc}. Chạy: `pip install pymupdf`")
        return

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for uf in uploaded:
            p = os.path.join(tmp, uf.name)
            with open(p, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(p)
        with st.spinner(f"Đang xử lý {len(paths)} PDF qua {inf_url} … "
                        f"(tiến trình chi tiết in ở terminal chạy streamlit)"):
            try:
                written = create_version_from_pdfs(
                    pdf_paths=paths, inference_url=inf_url, max_samples=n, dpi=dpi,
                    version=None,          # auto by counter
                    exclude_versions=None,  # always exclude benchmark
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Lỗi ingest: {exc}")
                return

    if not written:
        st.warning("Không có sample mới nào được thêm (toàn trùng / benchmark / inference lỗi).")
        return
    st.success(f"Đã ghi {len(written)} version.")
    st.dataframe(
        [{"version": m["version"], "count": m["count"], "target": m["target_per_version"],
          "trạng thái": "🟢 COMPLETE" if m["complete"] else "🟡 OPEN",
          "loại do benchmark": m["filter_stats"].get("rejected_benchmark", 0),
          "run": m["mlflow_run_id"][:8]} for m in written],
        hide_index=True, use_container_width=True,
    )


def _clean_log(text: str) -> str:
    """
    Collapse tqdm progress spam: tqdm updates a bar with \\r, but in a captured
    (non-tty) log each update becomes a new line → hundreds of near-duplicate
    lines. Keep only the latest of any run of consecutive progress lines so the
    box shows one tidy, updating bar instead of a flood.
    """
    out: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        ln = raw.rstrip()
        if not ln:
            continue
        if _PROG_RE.search(ln) and out and _PROG_RE.search(out[-1]):
            out[-1] = ln          # replace previous progress line with newest
        else:
            out.append(ln)
    return "\n".join(out)


def _ssh_tail(host: str, port: str, lines: int = 400) -> tuple[bool, str]:
    """Run `tail -n <lines>` on the instance via SSH (local ssh client + Vast key)."""
    try:
        out = subprocess.run(
            ["ssh", "-p", str(port),
             "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=8",
             f"root@{host}", f"tail -n {lines} /var/log/onstart.log"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=15,
        )
        return (True, out.stdout) if out.returncode == 0 else (False, out.stderr.strip() or "ssh failed")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


st.title("🎛️ Pipeline Actions")

_DATA, _TRAIN, _GATE, _SERVE = "🗂️ Data", "🏋️ Training", "✅ CI/CD Gate", "🚀 Serving"
section = st.segmented_control(
    "Section", [_DATA, _TRAIN, _GATE, _SERVE], default=_TRAIN, label_visibility="collapsed"
)

# ── Data: create a version ────────────────────────────────────────────────────
if section == _DATA:
    st.subheader("Tạo data version")
    st.caption("Duyệt các version đã có ở MinIO console (xem trang Links).")

    # Self-labeling từ PDF (chạy client-side trong tiến trình Streamlit).
    # (Mode "Từ HF Hub" tạm ẩn — endpoint /data/versions/create vẫn còn nếu cần.)
    st.caption("Client tách PDF → ảnh từng trang → inference server → DocTags → gom đủ N "
               "thành 1 version. Version tự đánh số theo counter (resume version OPEN nếu "
               "có, không thì lấy số kế tiếp); đủ N thì đóng, phần dư tràn sang version mới. "
               "Trang trùng benchmark luôn bị loại tự động (chống leakage).")
    with st.form("pdf_version"):
        c1, c2 = st.columns(2)
        n = c1.number_input("Sample / version (N)", min_value=1, value=200, step=10)
        dpi = c2.number_input("Render DPI", min_value=72, value=200, step=10)
        inf_url = st.text_input("Inference server URL", value=f"http://{_api_host()}",
                                help="LB serving (OpenAI-compatible). Pool phải đang Deploy.")
        pdfs = st.file_uploader("PDF tài liệu", type=["pdf"], accept_multiple_files=True)
        go = st.form_submit_button("🚀 Tạo / append version từ PDF", type="primary")

    if go:
        if not pdfs:
            st.error("Cần upload ít nhất 1 PDF.")
        else:
            _run_pdf_ingest(pdfs, inf_url.rstrip("/"), int(n), int(dpi))

# ── Training: trigger + track + live logs ─────────────────────────────────────
elif section == _TRAIN:
    st.session_state.setdefault("training_jobs", [])

    ok_d, data = api_get("/data/versions")
    ok_m, models = api_get("/models/versions")
    data_versions = data.get("versions", []) if ok_d else []
    model_versions = sorted(
        (str(v["version"]) for v in models.get("versions", [])), key=int, reverse=True
    ) if ok_m else []

    # Mask-mode selector tạm ẩn (loc-only/bbox đang điều tra — gây overfit, không
    # cải thiện structure). Training mặc định full-sequence "all". Khôi phục sau khi
    # xác minh nguyên nhân (truncation/độ phân giải) bằng cách bỏ comment phần này
    # + ô selectbox + nhánh body["mask_mode"] bên dưới.

    with st.form("trigger_training"):
        st.subheader("Trigger a training job")
        dv = st.selectbox("Data version", data_versions) if data_versions else st.text_input("Data version")
        init_v = st.selectbox("Warm-start from model version", ["latest (default)"] + model_versions)
        gpu_offer = st.text_input("GPU offer ID (để trống = env / auto-select)", placeholder="vd 39903270")
        c1, c2, c3, c4 = st.columns(4)
        epochs = c1.number_input("num_epochs", min_value=1, value=None, step=1)
        batch = c2.number_input("batch_size", min_value=1, value=None, step=1)
        grad = c3.number_input("grad_accum", min_value=1, value=None, step=1)
        lr = c4.number_input("learning_rate", min_value=0.0, value=None, format="%.6f")
        go = st.form_submit_button("🚀 Trigger")

    if go:
        body = {"data_version": dv}
        if init_v and not init_v.startswith("latest"):
            body["init_adapter_version"] = init_v
        if gpu_offer.strip():
            body["gpu_template_id"] = gpu_offer.strip()
        if epochs is not None: body["num_epochs"] = int(epochs)
        if batch is not None:  body["batch_size"] = int(batch)
        if grad is not None:   body["grad_accum"] = int(grad)
        if lr:                 body["learning_rate"] = float(lr)
        ok, res = api_post("/training/trigger", json=body)
        if ok:
            job_id = res["job_id"]
            if job_id not in st.session_state.training_jobs:
                st.session_state.training_jobs.insert(0, job_id)
            st.success(f"Triggered · job_id = {job_id}")
        else:
            st.error(res)

    st.divider()
    st.subheader("Track a job")
    ok_j, jobs_resp = api_get("/training/jobs")
    jobs = jobs_resp.get("jobs", []) if ok_j else []
    badge = {"running": "🟡", "completed": "🟢", "failed": "🔴"}
    options = {
        f"{badge.get(j['status'], '⚪')} {j['status']} · {j['job_id'][:8]} · "
        f"{j.get('data_version') or ''}"
        f"{' · 📦 bbox' if j.get('mask_mode') == 'bbox' else ''}":
            j["job_id"]
        for j in jobs
    }
    if options:
        label = st.selectbox("Job (mới nhất ở trên)", list(options.keys()))
        job_id = options.get(label)
    else:
        job_id = st.text_input("Job id (chưa có job nào — dán job_id)")

    if job_id:
        ok, status = api_get(f"/training/{job_id}/status")
        if not ok:
            st.error(status)
        else:
            badge = {"running": "🟡", "completed": "🟢", "failed": "🔴"}.get(status.get("status"), "⚪")
            st.write(f"{badge} **{status.get('status')}** · instance: {status.get('vast_instance_id') or '—'}")
            links = []
            if status.get("mlflow_run_url"): links.append(f"[MLflow]({status['mlflow_run_url']})")
            if status.get("wandb_run_url"):  links.append(f"[W&B]({status['wandb_run_url']})")
            if links:
                st.markdown(" · ".join(links))
            if status.get("metrics"):
                st.json(status["metrics"])

            with st.expander("🛟 Recover (instance chết / job kẹt)"):
                st.caption("Resume từ đúng stage (tự quyết): REGISTER_ONLY / FINALIZE / "
                           "RESUME_TRAIN. Để trống offer ID = env/auto; điền nếu auto-select "
                           "không tìm được máy.")
                rec_offer = st.text_input("GPU offer ID", key="recover_offer",
                                          placeholder="vd 39903270 (tùy chọn)")
                if st.button("🛟 Recover job"):
                    body = {}
                    if rec_offer.strip():
                        body["gpu_template_id"] = rec_offer.strip()
                    ok_r, res_r = api_post(f"/training/{job_id}/recover", json=body)
                    if ok_r:
                        prov = "đã thuê instance" if res_r["provisioned"] else "xử lý tại server (không cần GPU)"
                        st.success(f"action = {res_r['action']} · {prov}")
                    else:
                        st.error(res_r)

            st.subheader("Live logs")
            ssh_host, ssh_port = status.get("ssh_host"), status.get("ssh_port")
            if ssh_host and ssh_port:
                running = status.get("status") == "running"
                st.caption("🔴 Live (cập nhật tại chỗ mỗi 2s, không nhấp nháy)."
                           if running else "Job đã kết thúc — log cố định.")
                # Placeholder created ONCE; the loop only updates its content
                # in-place (no st.rerun → no DOM rebuild → no flicker). Any widget
                # interaction (section/sidebar/nav) interrupts this loop.
                ph = st.container(height=420).empty()
                while True:
                    ok_log, text = _ssh_tail(ssh_host, ssh_port)
                    if ok_log:
                        ph.code(_clean_log(text) or "(empty)")
                    else:
                        ph.warning(f"SSH chưa kết nối được (đang thử lại): {text}")
                    if not running:
                        break
                    time.sleep(2)
                    ok_s, s2 = api_get(f"/training/{job_id}/status")
                    running = ok_s and s2.get("status") == "running"
                    # Re-read ssh target each poll so logs follow a NEW instance
                    # after a recover re-provisions (tags get overwritten).
                    if ok_s and s2.get("ssh_host") and s2.get("ssh_port"):
                        ssh_host, ssh_port = s2["ssh_host"], s2["ssh_port"]
            else:
                st.caption("Instance chưa provisioned xong — log sẽ stream khi sẵn sàng.")

        with st.expander("Snapshot logs (qua Vast API, ~1 phút lag)"):
            tail = st.slider("tail lines", 50, 1000, 200, step=50)
            if st.button("Fetch snapshot"):
                with st.spinner("Fetching logs…"):
                    ok, logs = api_get(f"/training/{job_id}/logs", params={"tail": tail}, timeout=60)
                if ok:
                    st.code(logs.get("logs", "") or "(empty)")
                else:
                    st.error(logs)

# ── CI/CD gate ────────────────────────────────────────────────────────────────
elif section == _GATE:
    st.subheader("Run CI/CD gate")
    st.caption("Reads Staging CER, compares to threshold + Production, transitions stage.")
    with st.form("run_gate"):
        c1, c2 = st.columns(2)
        threshold = c1.number_input("cer_threshold", min_value=0.0, value=0.15, format="%.4f")
        tolerance = c2.number_input("regression_tolerance", min_value=1.0, value=1.05, format="%.2f")
        run = st.form_submit_button("Run gate")
    if run:
        ok, res = api_post("/cicd/gate", json={
            "cer_threshold": float(threshold), "regression_tolerance": float(tolerance),
        })
        if not ok:
            st.error(res)
        else:
            result = res.get("result")
            msg = f"{result.upper()} → {res.get('new_stage')}"
            if result == "pass":
                st.success(msg, icon="🟢")
            else:
                st.error(msg, icon="🔴")
            c1, c2, c3 = st.columns(3)
            c1.metric("Staging version", res.get("staging_version"))
            c2.metric("Staging CER", f"{res.get('staging_cer'):.4f}")
            prod = res.get("production_cer")
            c3.metric("Production CER", f"{prod:.4f}" if prod is not None else "—")
            # eval_loss drives the regression check
            d1, d2 = st.columns(2)
            sl = res.get("staging_loss")
            pl = res.get("production_loss")
            d1.metric("Staging eval_loss", f"{sl:.4f}" if sl is not None else "—")
            d2.metric("Production eval_loss", f"{pl:.4f}" if pl is not None else "—")

# ── Serving: deploy / teardown the Colab pool + status ────────────────────────
elif section == _SERVE:
    st.subheader("Serving (Colab pool)")
    st.caption("Deploy bật pool (min 1, tự co giãn 1↔2 theo tải, tự ngủ sau 10' rảnh). "
               "Inference gọi qua LB URL bên dưới (OpenAI-compatible /v1/chat/completions).")

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🚀 Deploy", use_container_width=True):
        ok, res = api_post("/deploy/trigger")
        if ok:
            st.success("Deploy requested — pool đang khởi động (~1.5-2 phút).")
        else:
            st.error(res)
    if c2.button("🛑 Teardown", use_container_width=True):
        ok, res = api_post("/deploy/teardown")
        if ok:
            st.warning("Teardown requested — đang stop toàn bộ instance.")
        else:
            st.error(res)
    auto = c3.toggle("Tự refresh trạng thái (3s)", value=False)

    ok_s, status = api_get("/deploy/status")
    if not ok_s:
        st.error(status)
    else:
        badge = {"active": "🟢", "starting": "🟡", "deploying": "🟡", "backoff": "🟡",
                 "sleeping": "⚪", "tearing_down": "🟠", "launch_failed": "🔴"}
        sstat = status.get("status", "unknown")
        floor = status.get("desired_floor", 0)
        st.write(f"{badge.get(sstat, '⚪')} **{sstat}** · desired_floor={floor}")

        lb = status.get("lb_url")
        if lb:
            st.markdown(f"**LB endpoint:** `{lb}/v1/chat/completions`  ·  [{lb}]({lb})")

        instances = status.get("instances", [])
        if instances:
            st.dataframe(
                [{"name": i["name"],
                  "ready": "✅" if i.get("ready") else "⏳",
                  "adapter": i.get("adapter") or "—",
                  "uptime (s)": i.get("uptime_s"),
                  "tunnel": i.get("tunnel_url") or "—"}
                 for i in instances],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Chưa có instance nào (pool đang ngủ hoặc khởi động).")

    if auto:
        time.sleep(3)
        st.rerun()

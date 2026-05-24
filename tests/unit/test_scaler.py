import json
import time
import pytest
from unittest.mock import MagicMock, patch

from src.serving.scaler import AutoScaler, ScalerConfig, ScalerState


@pytest.fixture
def config(tmp_path):
    return ScalerConfig(
        scale_up_threshold=5.0,
        scale_down_threshold=1.5,
        min_instances=1,
        max_instances=5,
        cooldown_seconds=180,
        poll_interval=30,
        nginx_upstream_conf=str(tmp_path / "upstream.conf"),
        state_file=str(tmp_path / "scaler_state.json"),
        prometheus_targets_file=str(tmp_path / "targets.json"),
        nginx_container_name="infra-nginx-1",
        vast_api_key="fake-key",
        gpu_template_id="template-123",
    )


@pytest.fixture
def scaler_one_instance(config):
    s = AutoScaler(config)
    s.state = ScalerState(
        instances=[{"id": "inst-1", "address": "10.0.0.1:8000"}],
        last_scale_time=0.0,
    )
    return s


@pytest.fixture
def scaler_two_instances(config):
    s = AutoScaler(config)
    s.state = ScalerState(
        instances=[
            {"id": "inst-1", "address": "10.0.0.1:8000"},
            {"id": "inst-2", "address": "10.0.0.2:8000"},
        ],
        last_scale_time=0.0,
    )
    return s


class TestScaleUpDecision:
    def test_true_above_threshold(self, scaler_one_instance):
        assert scaler_one_instance.should_scale_up(p95_latency=6.0) is True

    def test_false_below_threshold(self, scaler_one_instance):
        assert scaler_one_instance.should_scale_up(p95_latency=4.0) is False

    def test_false_at_exact_threshold(self, scaler_one_instance):
        assert scaler_one_instance.should_scale_up(p95_latency=5.0) is False

    def test_true_just_above_threshold(self, scaler_one_instance):
        assert scaler_one_instance.should_scale_up(p95_latency=5.001) is True

    def test_returns_bool(self, scaler_one_instance):
        assert isinstance(scaler_one_instance.should_scale_up(6.0), bool)


class TestScaleDownDecision:
    def test_true_below_threshold(self, scaler_two_instances):
        assert scaler_two_instances.should_scale_down(p95_latency=1.0) is True

    def test_false_above_threshold(self, scaler_two_instances):
        assert scaler_two_instances.should_scale_down(p95_latency=2.0) is False

    def test_false_at_exact_threshold(self, scaler_two_instances):
        assert scaler_two_instances.should_scale_down(p95_latency=1.5) is False

    def test_true_just_below_threshold(self, scaler_two_instances):
        assert scaler_two_instances.should_scale_down(p95_latency=1.499) is True

    def test_returns_bool(self, scaler_two_instances):
        assert isinstance(scaler_two_instances.should_scale_down(1.0), bool)


class TestCooldown:
    def test_active_within_window(self, scaler_one_instance):
        scaler_one_instance.state.last_scale_time = time.time() - 60  # 60s ago < 180s
        assert scaler_one_instance.is_cooldown_active() is True

    def test_expired_after_window(self, scaler_one_instance):
        scaler_one_instance.state.last_scale_time = time.time() - 200  # 200s > 180s
        assert scaler_one_instance.is_cooldown_active() is False

    def test_initial_zero_time_no_cooldown(self, config):
        s = AutoScaler(config)
        s.state = ScalerState(instances=[], last_scale_time=0.0)
        assert s.is_cooldown_active() is False

    def test_exactly_at_boundary_not_active(self, scaler_one_instance):
        scaler_one_instance.state.last_scale_time = time.time() - 180
        assert scaler_one_instance.is_cooldown_active() is False


class TestInstanceLimits:
    def test_cannot_scale_up_at_max(self, config):
        s = AutoScaler(config)
        s.state = ScalerState(
            instances=[{"id": f"inst-{i}", "address": f"10.0.0.{i}:8000"} for i in range(5)],
            last_scale_time=0.0,
        )
        assert s.can_scale_up() is False

    def test_can_scale_up_below_max(self, scaler_one_instance):
        assert scaler_one_instance.can_scale_up() is True

    def test_cannot_scale_down_at_min(self, scaler_one_instance):
        assert scaler_one_instance.can_scale_down() is False

    def test_can_scale_down_above_min(self, scaler_two_instances):
        assert scaler_two_instances.can_scale_down() is True

    def test_returns_bool_scale_up(self, scaler_one_instance):
        assert isinstance(scaler_one_instance.can_scale_up(), bool)

    def test_returns_bool_scale_down(self, scaler_one_instance):
        assert isinstance(scaler_one_instance.can_scale_down(), bool)


class TestNginxReload:
    def test_calls_docker_exec_with_container_name(self, scaler_one_instance):
        with patch("subprocess.run") as mock_run:
            scaler_one_instance._reload_nginx()
        cmd = mock_run.call_args[0][0]
        assert cmd == ["docker", "exec", "infra-nginx-1", "nginx", "-s", "reload"]

    def test_calls_subprocess_run_with_check_true(self, scaler_one_instance):
        with patch("subprocess.run") as mock_run:
            scaler_one_instance._reload_nginx()
        mock_run.assert_called_once_with(
            ["docker", "exec", "infra-nginx-1", "nginx", "-s", "reload"],
            check=True,
        )


class TestNginxUpstreamWrite:
    def test_conf_contains_all_addresses(self, scaler_two_instances):
        with patch.object(scaler_two_instances, "_reload_nginx"):
            scaler_two_instances._write_nginx_upstream()
        content = open(scaler_two_instances.config.nginx_upstream_conf).read()
        assert "10.0.0.1:8000" in content
        assert "10.0.0.2:8000" in content

    def test_nginx_reload_called_once(self, scaler_two_instances):
        with patch.object(scaler_two_instances, "_reload_nginx") as mock_reload:
            scaler_two_instances._write_nginx_upstream()
        mock_reload.assert_called_once()

    def test_removed_instance_not_in_conf(self, scaler_one_instance):
        with patch.object(scaler_one_instance, "_reload_nginx"):
            scaler_one_instance._write_nginx_upstream()
        content = open(scaler_one_instance.config.nginx_upstream_conf).read()
        assert "10.0.0.2:8000" not in content
        assert "10.0.0.1:8000" in content

    def test_conf_file_is_created(self, scaler_one_instance):
        import os
        conf = scaler_one_instance.config.nginx_upstream_conf
        if os.path.exists(conf):
            os.remove(conf)
        with patch.object(scaler_one_instance, "_reload_nginx"):
            scaler_one_instance._write_nginx_upstream()
        assert os.path.exists(conf)


class TestStatePersistence:
    def test_save_creates_file(self, scaler_one_instance):
        import os
        path = scaler_one_instance.config.state_file
        if os.path.exists(path):
            os.remove(path)
        scaler_one_instance.save_state()
        assert os.path.exists(path)

    def test_saved_state_has_required_keys(self, scaler_one_instance):
        scaler_one_instance.save_state()
        data = json.loads(open(scaler_one_instance.config.state_file).read())
        assert "instances" in data
        assert "last_scale_time" in data

    def test_saved_instances_match_state(self, scaler_one_instance):
        scaler_one_instance.save_state()
        data = json.loads(open(scaler_one_instance.config.state_file).read())
        assert len(data["instances"]) == 1
        assert data["instances"][0]["id"] == "inst-1"

    def test_load_restores_instances(self, config):
        state_data = {"instances": [{"id": "inst-99", "address": "10.0.0.99:8000"}], "last_scale_time": 1234567890.0}
        with open(config.state_file, "w") as f:
            json.dump(state_data, f)
        s = AutoScaler(config)
        s.load_state()
        assert len(s.state.instances) == 1
        assert s.state.instances[0]["id"] == "inst-99"

    def test_load_restores_last_scale_time(self, config):
        state_data = {"instances": [], "last_scale_time": 9999999.0}
        with open(config.state_file, "w") as f:
            json.dump(state_data, f)
        s = AutoScaler(config)
        s.load_state()
        assert s.state.last_scale_time == pytest.approx(9999999.0)

    def test_load_missing_file_initializes_empty(self, config):
        import os
        if os.path.exists(config.state_file):
            os.remove(config.state_file)
        s = AutoScaler(config)
        s.load_state()
        assert s.state.instances == []
        assert s.state.last_scale_time == 0.0


class TestScaleUpFlow:
    def test_appends_new_instance_to_state(self, scaler_one_instance):
        new_instance = {"id": "inst-new", "address": "10.0.0.99:8000"}
        with (
            patch.object(scaler_one_instance, "_create_vast_instance", return_value=new_instance),
            patch.object(scaler_one_instance, "_write_nginx_upstream"),
            patch.object(scaler_one_instance, "_write_prometheus_targets"),
            patch.object(scaler_one_instance, "save_state"),
        ):
            scaler_one_instance.scale_up()
        assert len(scaler_one_instance.state.instances) == 2
        assert any(inst["id"] == "inst-new" for inst in scaler_one_instance.state.instances)

    def test_updates_last_scale_time(self, scaler_one_instance):
        new_instance = {"id": "inst-new", "address": "10.0.0.99:8000"}
        before = time.time()
        with (
            patch.object(scaler_one_instance, "_create_vast_instance", return_value=new_instance),
            patch.object(scaler_one_instance, "_write_nginx_upstream"),
            patch.object(scaler_one_instance, "_write_prometheus_targets"),
            patch.object(scaler_one_instance, "save_state"),
        ):
            scaler_one_instance.scale_up()
        assert scaler_one_instance.state.last_scale_time >= before

    def test_prometheus_targets_called_on_scale_up(self, scaler_one_instance):
        new_instance = {"id": "inst-new", "address": "10.0.0.99:8000"}
        with (
            patch.object(scaler_one_instance, "_create_vast_instance", return_value=new_instance),
            patch.object(scaler_one_instance, "_write_nginx_upstream"),
            patch.object(scaler_one_instance, "_write_prometheus_targets") as mock_prom,
            patch.object(scaler_one_instance, "save_state"),
        ):
            scaler_one_instance.scale_up()
        mock_prom.assert_called_once()


class TestScaleDownFlow:
    def test_removes_one_instance_from_state(self, scaler_two_instances):
        with (
            patch.object(scaler_two_instances, "_destroy_vast_instance"),
            patch.object(scaler_two_instances, "_write_nginx_upstream"),
            patch.object(scaler_two_instances, "_write_prometheus_targets"),
            patch.object(scaler_two_instances, "save_state"),
        ):
            scaler_two_instances.scale_down()
        assert len(scaler_two_instances.state.instances) == 1

    def test_updates_last_scale_time(self, scaler_two_instances):
        before = time.time()
        with (
            patch.object(scaler_two_instances, "_destroy_vast_instance"),
            patch.object(scaler_two_instances, "_write_nginx_upstream"),
            patch.object(scaler_two_instances, "_write_prometheus_targets"),
            patch.object(scaler_two_instances, "save_state"),
        ):
            scaler_two_instances.scale_down()
        assert scaler_two_instances.state.last_scale_time >= before

    def test_prometheus_targets_called_on_scale_down(self, scaler_two_instances):
        with (
            patch.object(scaler_two_instances, "_destroy_vast_instance"),
            patch.object(scaler_two_instances, "_write_nginx_upstream"),
            patch.object(scaler_two_instances, "_write_prometheus_targets") as mock_prom,
            patch.object(scaler_two_instances, "save_state"),
        ):
            scaler_two_instances.scale_down()
        mock_prom.assert_called_once()


class TestPrometheusTargetsWrite:
    def test_creates_targets_file(self, scaler_one_instance):
        import os
        path = scaler_one_instance.config.prometheus_targets_file
        if os.path.exists(path):
            os.remove(path)
        scaler_one_instance._write_prometheus_targets()
        assert os.path.exists(path)

    def test_targets_contain_instance_addresses(self, scaler_two_instances):
        scaler_two_instances._write_prometheus_targets()
        content = json.loads(open(scaler_two_instances.config.prometheus_targets_file).read())
        targets = content[0]["targets"]
        assert "10.0.0.1:8000" in targets
        assert "10.0.0.2:8000" in targets

    def test_targets_empty_when_no_instances(self, config):
        s = AutoScaler(config)
        s.state = ScalerState(instances=[], last_scale_time=0.0)
        s._write_prometheus_targets()
        content = json.loads(open(config.prometheus_targets_file).read())
        assert content[0]["targets"] == []

    def test_removed_instance_not_in_targets(self, scaler_one_instance):
        scaler_one_instance._write_prometheus_targets()
        content = json.loads(open(scaler_one_instance.config.prometheus_targets_file).read())
        assert "10.0.0.2:8000" not in content[0]["targets"]

    def test_targets_job_label_is_vllm(self, scaler_one_instance):
        scaler_one_instance._write_prometheus_targets()
        content = json.loads(open(scaler_one_instance.config.prometheus_targets_file).read())
        assert content[0]["labels"]["job"] == "vllm"

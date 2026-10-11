"""Opt-in profiles for the two Windows initial-loading failures."""

import json
import re
import sys
import time
from pathlib import Path

import pytest

_TESTS = {
    "test_restart_timeout_can_reconnect_without_resubmitting_keys",
    "test_onboarding_errors_retry_independently",
}

_SCRIPT = """(() => {
  const data = window.startupDiagnostics = {tasks: [], fetches: [], lags: [], mutations: []};
  performance.setResourceTimingBufferSize(2000);
  new PerformanceObserver(list => {
    data.tasks.push(...list.getEntries().map(e => ({start: e.startTime, duration: e.duration})));
  }).observe({type: 'longtask', buffered: true});
  let last = performance.now();
  setInterval(() => {
    const now = performance.now();
    if (now - last > 150) data.lags.push({start: last, duration: now - last});
    last = now;
  }, 50);
  const original = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const row = {url: String(args[0]), start: performance.now()};
    data.fetches.push(row);
    try {
      const response = await original(...args);
      row.headers = performance.now();
      row.status = response.status;
      const json = response.json.bind(response);
      response.json = async (...opts) => {
        row.jsonStart = performance.now();
        const result = await json(...opts);
        row.jsonEnd = performance.now();
        return result;
      };
      return response;
    } catch (error) {
      row.error = error.message;
      row.end = performance.now();
      throw error;
    }
  };
  document.addEventListener('DOMContentLoaded', () => {
    for (const id of ['messageArea', 'openClaudeOnboarding']) {
      const element = document.getElementById(id);
      if (element) new MutationObserver(() => {
        data.mutations.push({id, time: performance.now(), text: element.textContent});
      }).observe(element, {subtree: true, childList: true, characterData: true});
    }
  });
})();"""


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    if sys.platform != "win32" or item.originalname not in _TESTS:
        return (yield)
    page = item.funcargs["page"]
    session = page.context.new_cdp_session(page)
    session.send("Performance.enable")
    session.send("Profiler.enable")
    session.send("Profiler.start")
    session.send("Network.enable")
    network = []

    def record(event, data):
        row = {
            "event": event,
            "request_id": data["requestId"],
            "browser_time": data["timestamp"],
            "received_at": time.monotonic(),
        }
        if event == "requestWillBeSent":
            row["url"] = data["request"]["url"]
        elif event == "responseReceived":
            row["status"] = data["response"]["status"]
            row["connection"] = data["response"].get("connectionId")
            row["timing"] = data["response"].get("timing")
        elif event == "loadingFailed":
            row["error"] = data["errorText"]
        network.append(row)

    for event in (
        "requestWillBeSent",
        "responseReceived",
        "loadingFinished",
        "loadingFailed",
    ):
        session.on("Network." + event, lambda data, event=event: record(event, data))
    page.add_init_script(_SCRIPT)
    try:
        return (yield)
    finally:
        # Capture before the shared fixture navigates away during teardown.
        profile = session.send("Profiler.stop")["profile"]
        metrics = session.send("Performance.getMetrics")["metrics"]
        timing = page.evaluate(
            """({...window.startupDiagnostics, timeOrigin: performance.timeOrigin,
              resources: performance.getEntriesByType('resource').map(e => e.toJSON())})"""
        )
        output = Path(item.config.getoption("--output")) / "startup-diagnostics"
        output.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^a-zA-Z0-9_.-]", "_", item.name)
        (output / f"{name}.cpuprofile").write_text(
            json.dumps(profile), encoding="utf-8"
        )
        (output / f"{name}.json").write_text(
            json.dumps({"timing": timing, "metrics": metrics, "network": network}),
            encoding="utf-8",
        )
        session.detach()

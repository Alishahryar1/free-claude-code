(() => {
  "use strict";
  let api, active = false, busy = false, generation = 0;
  const root = document.getElementById("integrationsRoot");
  const message = document.getElementById("integrationMessage");
  const refreshButton = document.getElementById("refreshIntegrations");
  const statuses = { not_configured: "Not configured", configured: "Configured", needs_attention: "Needs attention" };
  const disconnectInstructions = {
    "claude-vscode": [
      "In the user settings file above, remove FCC's ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN, and CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY entries from claudeCode.environmentVariables. Keep unrelated environment entries.",
      "Remove claudeCode.disableLoginPrompt or set it to false to resume normal login. Other optional flags from the setup snippet can be removed if no longer wanted. Reload VS Code.",
      "These steps remove this card's routing settings. Separate global/project overrides still need manual correction if present.",
    ],
    codex: [
      "In the shared TOML file above, remove model_provider only when it selects fcc. Remove the FCC entry under model_providers, including its auth settings. Remove model_catalog_json only when it points to FCC's catalog.",
      "Remove or replace the FCC model selection with a model for your normal provider. These steps apply to both table-header and inline TOML layouts. Preserve other providers and settings.",
      "Restart affected Codex App, VS Code, and CLI sessions, then use the client's normal authentication.",
    ],
    "claude-jetbrains": [
      'In the ACP file above, remove only agent_servers["Claude Code (FCC)"]. This removes the named agent, including any custom fields inside it. Preserve other agents.',
      "Restart the IDE and select your normal agent.",
    ],
  };

  function node(tag, text = "", className = "") {
    const value = document.createElement(tag);
    value.textContent = text;
    value.className = className;
    return value;
  }
  function button(label, action, className = "secondary-button") {
    const value = node("button", label, className);
    value.type = "button";
    value.addEventListener("click", action);
    return value;
  }
  function notice(text, error = false) {
    message.textContent = text;
    message.classList.toggle("error", error);
  }
  function setBusy(value) {
    busy = value;
    refreshButton.disabled = value;
    for (const control of root.querySelectorAll("button")) control.disabled = value;
  }
  function render(item) {
    const card = node("article", "", "integration-card");
    card.dataset.integration = item.id;
    const heading = node("div", "", "integration-heading");
    heading.append(node("h3", item.title));
    heading.append(node("span", statuses[item.status] || "Needs attention", `integration-status ${item.status}`));
    card.append(heading);
    const badges = node("div", "", "integration-badges");
    for (const badge of item.badges) badges.append(node("span", badge));
    card.append(badges, node("p", item.path, "integration-path"));
    if (item.id === "codex") card.append(node("p", "One shared configuration for Codex App, the VS Code extension, and normal Codex CLI use."));
    if (item.id === "claude-login") card.append(node("p", "Apply the shared Claude Code first-run setting when it still asks you to log in."));
    if (item.message) card.append(node("p", item.message, "integration-guidance"));
    for (const missing of item.missing) card.append(node("p", missing, "integration-guidance"));
    if (item.status === "configured" && item.id !== "claude-login") card.append(node("p", "Configuration saved. Restart the client to use it."));
    const actions = node("div", "", "integration-actions");
    if (item.can_apply) {
      const control = button(item.id === "claude-login" ? "Fix" : "Apply FCC settings", () => preview(item, control), "primary-button");
      actions.append(control);
    }
    const manual = node("a", "Manual setup");
    manual.href = item.documentation_url;
    manual.target = "_blank";
    manual.rel = "noopener noreferrer";
    actions.append(manual);
    card.append(actions);
    if (disconnectInstructions[item.id]) {
      const help = node("details", "", "integration-disconnect");
      help.append(node("summary", "How to disconnect"));
      for (const instruction of disconnectInstructions[item.id]) help.append(node("p", instruction));
      card.append(help);
    }
    return card;
  }
  async function refresh() {
    if (!api || busy) return;
    const request = ++generation;
    refreshButton.disabled = true;
    try {
      const result = await api("/admin/api/integrations");
      if (request !== generation) return;
      root.replaceChildren(...result.items.map(render));
      document.getElementById("integrationScope").textContent = result.scope;
      document.getElementById("integrationServerNotice").textContent = result.notice || "";
    } catch (error) {
      if (request === generation) notice(`Could not inspect integrations. ${error.message}`, true);
    } finally {
      if (request === generation) refreshButton.disabled = false;
    }
  }
  async function preview(item, source) {
    if (busy) return;
    ++generation;
    setBusy(true);
    notice("");
    try {
      const result = await api(`/admin/api/integrations/${item.id}/preview`, {
        method: "POST", body: JSON.stringify({}),
      });
      if (active) showPreview(result, source);
    } catch (error) {
      notice(error.message, true);
    } finally {
      setBusy(false);
    }
  }
  function showPreview(preview, source) {
    const dialog = node("dialog", "", "integration-dialog");
    const heading = node("h3", `${preview.id === "claude-login" ? "Fix" : "Apply FCC settings"}: ${preview.title}`);
    heading.id = "integrationDialogTitle";
    dialog.setAttribute("aria-labelledby", heading.id);
    const content = node("div", "", "integration-dialog-content");
    dialog.append(heading, content);
    content.append(node("p", preview.writes_file ? "FCC will update this file:" : "Configuration file:"), node("p", preview.path, "integration-path"));
    content.append(node("p", preview.summary, "integration-guidance"));
    const actions = node("div", "", "integration-actions");
    let committing = false;
    function close() {
      if (committing) return;
      dialog.close();
      dialog.remove();
      if (source.isConnected) source.focus();
    }
    const cancel = button("Cancel", close);
    cancel.autofocus = true;
    const confirm = button("Confirm", async () => {
      if (committing) return;
      committing = true;
      setBusy(true);
      cancel.disabled = confirm.disabled = true;
      confirm.textContent = "Saving…";
      try {
        const result = await api(`/admin/api/integrations/${preview.id}/apply`, {
          method: "POST", body: JSON.stringify({ revision: preview.revision }),
        });
        notice(`${result.message || "No client settings changed."} ${result.instructions}`);
      } catch (error) {
        notice(`${error.message} Current files will be checked again before another confirmation.`, true);
      } finally {
        committing = false;
        close();
        setBusy(false);
        await refresh();
        if (active) refreshButton.focus();
      }
    }, "primary-button");
    actions.append(cancel, confirm);
    dialog.append(actions);
    dialog.addEventListener("cancel", (event) => { event.preventDefault(); close(); });
    document.body.append(dialog);
    dialog.showModal();
    cancel.focus();
  }
  refreshButton.addEventListener("click", refresh);
  window.Integrations = {
    initialize(request) { api = request; },
    activate() { active = true; refresh(); },
    deactivate() { active = false; ++generation; },
  };
})();

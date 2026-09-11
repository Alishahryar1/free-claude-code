(() => {
  "use strict";
  let api, active = false, busy = false, generation = 0;
  const root = document.getElementById("integrationsRoot");
  const message = document.getElementById("integrationMessage");
  const refreshButton = document.getElementById("refreshIntegrations");
  const labels = { setup: "Set up", update: "Update", disconnect: "Disconnect", repair: "Fix" };
  const statuses = { not_configured: "Not configured", configured: "Configured", update_available: "Update available", needs_attention: "Needs attention" };

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
    if (item.manual) badges.append(node("span", "Manual FCC setup detected"));
    card.append(badges, node("p", item.path, "integration-path"));
    if (item.id === "codex") card.append(node("p", "One shared configuration for Codex App, the VS Code extension, and normal Codex CLI use."));
    if (item.id === "claude-login") card.append(node("p", "Apply the shared Claude Code first-run setting when it still asks you to log in."));
    if (item.message) card.append(node("p", item.message, "integration-guidance"));
    for (const missing of item.missing) card.append(node("p", missing, "integration-guidance"));
    if (item.status === "configured" && item.id !== "claude-login") card.append(node("p", "Configuration saved. Restart the client to use it."));
    const actions = node("div", "", "integration-actions");
    for (const action of item.actions) {
      const control = button(labels[action], () => preview(item, action, control), action === "disconnect" ? "secondary-button" : "primary-button");
      actions.append(control);
    }
    const manual = node("a", "Manual setup");
    manual.href = item.documentation_url;
    manual.target = "_blank";
    manual.rel = "noopener noreferrer";
    actions.append(manual);
    card.append(actions);
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
  async function preview(item, action, source) {
    if (busy) return;
    ++generation;
    setBusy(true);
    notice("");
    try {
      const result = await api(`/admin/api/integrations/${item.id}/preview`, {
        method: "POST", body: JSON.stringify({ action }),
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
    const heading = node("h3", `${labels[preview.action]}: ${preview.title}`);
    heading.id = "integrationDialogTitle";
    dialog.setAttribute("aria-labelledby", heading.id);
    const content = node("div", "", "integration-dialog-content");
    dialog.append(heading, content);
    content.append(node("p", "FCC will update this file:"), node("p", preview.path, "integration-path"));
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
          method: "POST", body: JSON.stringify({ action: preview.action, revision: preview.revision }),
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

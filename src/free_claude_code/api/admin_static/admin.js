const state = {
  config: null,
  applying: false,
  restart: null,
  fields: new Map(),
  modelOptions: [],
  modelComboboxes: new Set(),
  authPollers: new Map(),
  authStatuses: new Map(),
  providerChecks: new Map(),
  providerId: null,
  localStatusRequest: null,
  startup: null,
  startupRequest: null,
  startupTimer: null,
  startupAgain: false,
  codeCatalogRetry: false,
  modelOptionsRequest: 0,
  activeView: viewFromLocation(),
};

const MASKED_SECRET = "********";
const NULL_VALUE = "__FCC_NULL__";
const VIEW_GROUPS = [
  {
    id: "providers",
    label: "Providers",
    title: "Providers",
    sections: ["runtime"],
    containerId: "providersSections",
  },
  {
    id: "model_config",
    label: "Model Config",
    title: "Model Config",
    sections: ["models", "reasoning", "web_tools"],
    containerId: "modelConfigSections",
  },
  {
    id: "messaging",
    label: "Messaging",
    title: "Messaging",
    sections: ["messaging", "voice"],
    containerId: "messagingSections",
  },
  {
    id: "integrations",
    label: "Integrations",
    title: "Integrations",
    sections: [],
    containerId: "view-integrations",
  },
  {
    id: "code",
    label: "Code sessions",
    title: "Code sessions",
    sections: [],
    containerId: "codeRoot",
  },
];

function viewFromLocation() {
  return window.location.pathname.split("/")[2] || "providers";
}

const byId = (id) => document.getElementById(id);

function sourceLabel(source) {
  const labels = {
    default: "default",
    managed_env: "",
    process: "process env",
  };
  return Object.prototype.hasOwnProperty.call(labels, source) ? labels[source] : source;
}

function sourceText(field) {
  const parts = [];
  const label = sourceLabel(field.source);
  if (label) {
    parts.push(label);
  }
  if (field.locked) {
    parts.push("locked");
  }
  return parts.join(" ");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = "";
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : "";
    } catch {
      // The status remains useful when an upstream proxy returns a non-JSON page.
    }
    const error = new Error(detail || `${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function startupButton(button, loading) {
  if (button.dataset.operationBusy) return;
  button.disabled = loading;
  button.classList.toggle("startup-busy", loading);
  button.setAttribute("aria-busy", String(loading));
}

function renderStartup() {
  const startup = state.startup?.startup;
  const message = byId("startupMessage");
  if (message) {
    const messaging = startup?.messaging;
    const visible = state.activeView === "messaging" && ["starting", "failed"].includes(messaging?.state);
    message.hidden = !visible;
    message.classList.toggle("startup-spinner", visible && messaging.state === "starting");
    message.classList.toggle("error", messaging?.state === "failed");
    message.textContent = messaging?.state === "failed" ? messaging.message || "Messaging could not start." : "";
    message.setAttribute("aria-label", "Messaging is starting");
  }
  if (!startup) return;
  document.querySelectorAll("[data-startup-provider]").forEach((button) => {
    startupButton(button, startup.providers?.[button.dataset.startupProvider] === "starting");
  });
  document.querySelectorAll("[data-startup-catalog]").forEach((button) => {
    startupButton(button, Object.values(startup.providers || {}).includes("starting"));
  });
  state.config.provider_status.forEach((provider) => {
    renderProviderCheckResult(provider.provider_id);
  });
  renderCodexIntegration();
}

async function refreshStartup() {
  if (!state.config || document.hidden || state.restart) return;
  if (state.startupRequest) { state.startupAgain = true; return; }
  clearTimeout(state.startupTimer);
  state.startupTimer = null;
  const request = { controller: new AbortController(), config: state.config };
  state.startupRequest = request;
  let pending = false;
  try {
    const result = await api("/admin/api/status", { signal: request.controller.signal });
    if (state.startupRequest !== request || state.config !== request.config) return;
    const previous = state.startup;
    const current = result.startup;
    if (!current) return;
    if (previous?.instance_id === result.instance_id && (
      current.generation_id < previous.startup.generation_id ||
      (current.generation_id === previous.startup.generation_id &&
       current.catalog_revision < previous.startup.catalog_revision)
    )) return;
    state.startup = result;
    renderStartup();
    const changed = !previous || previous.instance_id !== result.instance_id ||
      JSON.stringify(previous.startup) !== JSON.stringify(current);
    if (changed) void hydrateModelOptions();
    if (changed || state.codeCatalogRetry) state.codeCatalogRetry = await window.CodeSessions?.refresh(result) === false;
    pending = state.codeCatalogRetry || current.catalog === "starting" || current.catalog_file === "starting" ||
      current.code?.state === "starting" || current.messaging?.state === "starting" ||
      Object.values(current.providers || {}).includes("starting");
  } catch (error) {
    if (error.name !== "AbortError") pending = true;
  } finally {
    if (state.startupRequest === request) {
      state.startupRequest = null;
      if ((pending || state.startupAgain) && !document.hidden) state.startupTimer = setTimeout(refreshStartup, 500);
      state.startupAgain = false;
    }
  }
}

document.addEventListener("visibilitychange", () => {
  clearTimeout(state.startupTimer);
  if (document.hidden) {
    state.startupRequest?.controller.abort();
    state.startupRequest = null;
  } else {
    void refreshStartup();
  }
});
window.addEventListener("pagehide", () => {
  clearTimeout(state.startupTimer);
  state.startupRequest?.controller.abort();
  state.startupRequest = null;
});

async function load({ providersOnly = false } = {}) {
  state.localStatusRequest = null;
  state.startupRequest?.controller.abort();
  state.startupRequest = null;
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  state.config = config;
  state.startup = null;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  if (!providersOnly) renderNav();
  state.providerChecks.clear();
  renderProviders(config.provider_status);
  if (!providersOnly) renderSections(config.sections, config.fields);
  byId("configPath").textContent = config.paths.managed;
  void refreshLocalStatus(config);
  void refreshStartup();
  await Promise.all([
    refreshConnectedAccounts(),
    hydrateModelOptions(),
    providersOnly ? window.CodeSessions.refresh() : window.CodeSessions.initialize(api),
  ]);
  if (state.config !== config) return;
  updateDirtyState();
  showMessage("");
}

function renderNav() {
  const nav = byId("sectionNav");
  nav.innerHTML = "";
  VIEW_GROUPS.forEach((view, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `nav-link${index === 0 ? " active" : ""}`;
    button.dataset.view = view.id;
    button.textContent = view.label;
    if (index === 0) {
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => {
      navigateToView(view.id);
    });
    nav.appendChild(button);
  });
  setActiveView(state.activeView, { scroll: false });
}

function setActiveView(viewId, { scroll = false } = {}) {
  const activeView =
    VIEW_GROUPS.find((view) => view.id === viewId) || VIEW_GROUPS[0];
  state.activeView = activeView.id;
  byId("pageTitle").textContent = activeView.title;
  renderStartup();
  if (activeView.id === "code") state.codeCatalogRetry = true;
  void refreshStartup();
  const sessionActive = activeView.id === "code";
  document.querySelector(".app-shell").classList.toggle("session-active", sessionActive);
  document.querySelector(".main").classList.toggle("session-main", sessionActive);
  document.querySelector(".topbar").hidden = sessionActive;
  document.querySelector(".action-bar").hidden = sessionActive || activeView.id === "integrations";

  document.querySelectorAll(".nav-link").forEach((link) => {
    const selected = link.dataset.view === activeView.id;
    link.classList.toggle("active", selected);
    if (selected) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });

  document.querySelectorAll(".admin-view").forEach((view) => {
    const selected = view.dataset.view === activeView.id;
    view.classList.toggle("active", selected);
    view.hidden = !selected;
  });

  if (scroll) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  if (activeView.id === "code") window.CodeSessions.activate(window.location.pathname);
  else window.CodeSessions.deactivate();
  if (activeView.id === "integrations") {
    refreshClaudeIntegration();
    refreshCodexIntegration();
  }
}

function navigateToView(viewId) {
  const target = viewId === "providers" ? "/admin" : `/admin/${viewId}`;
  if (window.location.pathname + window.location.search !== target) {
    window.history.pushState({}, "", target);
  }
  setActiveView(viewId, { scroll: true });
}

function renderProviders(providerStatus) {
  const container = byId("providerGroups");
  container.replaceChildren();
  [
    ["oauth", "OAuth providers"],
    ["cloud", "Cloud providers"],
    ["local", "Local providers"],
  ].forEach(([kind, label]) => {
    const group = document.createElement("section");
    group.className = "provider-strip";
    group.dataset.providerGroup = kind;
    const heading = document.createElement("h3");
    heading.textContent = label;
    group.appendChild(heading);
    const subgroups = [
      ["configured", kind === "oauth" ? "Connected" : "Configured"],
      ["unconfigured", kind === "oauth" ? "Not connected" : "Not configured"],
    ];
    if (kind === "oauth") subgroups.unshift(["loading", "Checking account status"]);
    subgroups.forEach(([subgroupId, subgroupLabel]) => {
      const subgroup = document.createElement("div");
      subgroup.dataset.providerSubgroup = subgroupId;
      subgroup.hidden = true;
      const title = document.createElement("h4");
      title.textContent = subgroupLabel;
      title.hidden = subgroupId === "loading";
      const grid = document.createElement("div");
      grid.className = "provider-grid";
      grid.id = `providers-${kind}-${subgroupId}`;
      subgroup.append(title, grid);
      group.appendChild(subgroup);
    });
    container.appendChild(group);
  });
  providerStatus.forEach(updateProviderCard);
}

function updateProviderCard(provider) {
  const oauth = provider.kind === "connected_account";
  const status = state.authStatuses.get(provider.provider_id);
  const kind = oauth ? "oauth" : provider.kind === "local" ? "local" : "cloud";
  const configured = oauth ? status?.connected : provider.status === "configured";
  const subgroup = oauth && configured == null ? "loading" : configured ? "configured" : "unconfigured";
  const grid = byId(`providers-${kind}-${subgroup}`);
  let card = document.querySelector(`[data-provider="${provider.provider_id}"]`);
  if (!card) {
    card = document.createElement("article");
    card.className = "provider-card";
    card.dataset.provider = provider.provider_id;
  }
  const focused = card.contains(document.activeElement);
  const focusedLabel = focused ? document.activeElement.textContent : null;
  const title = document.createElement("span");
  title.className = "provider-title";
  const name = document.createElement("strong");
  name.textContent = connectedAccountName(provider);
  title.appendChild(name);
  const meta = document.createElement("span");
  meta.className = "provider-meta";
  meta.hidden = !oauth;
  const result = document.createElement("span");
  result.className = "provider-check-result";
  result.dataset.providerCheckResult = provider.provider_id;
  result.hidden = true;
  const actions = document.createElement("div");
  actions.className = "provider-actions";
  if (oauth) populateConnectedAccountActions(provider, status, actions);
  if (provider.settings_keys?.length) {
    const edit = oauth || configured;
    const settings = authButton(edit ? "Edit" : "Configure", () => openProviderDialog(provider.provider_id), edit ? "secondary-button" : "primary-button");
    settings.dataset.providerSettings = "true";
    settings.setAttribute("aria-haspopup", "dialog");
    settings.setAttribute("aria-controls", "providerDialog");
    actions.appendChild(settings);
  }
  card.replaceChildren(title, meta, result, actions);
  const next = [...grid.children].find((other) => other !== card &&
    connectedAccountName(provider).localeCompare(providerDisplayName(other.dataset.provider), "en", { sensitivity: "base" }) < 0);
  if (card.parentElement !== grid || card.nextElementSibling !== (next || null)) grid.insertBefore(card, next || null);
  document.querySelectorAll("[data-provider-subgroup]").forEach((section) => {
    section.hidden = section.querySelector(".provider-grid").childElementCount === 0;
  });
  if (focused) {
    const buttons = [...actions.querySelectorAll("button:not(:disabled)")];
    (buttons.find((button) => button.textContent === focusedLabel) || buttons[0])?.focus({ preventScroll: true });
  }
  renderProviderCheckResult(provider.provider_id);
}

function openProviderDialog(providerId) {
  state.providerId = providerId;
  const provider = connectedAccountDescriptor(providerId);
  byId("providerDialogTitle").textContent = connectedAccountName(provider);
  byId("providerMessage").textContent = "";
  const fields = byId("providerFields");
  fields.replaceChildren();
  (provider.settings_keys || []).forEach((key) => {
    const field = state.fields.get(key);
    const wrapper = renderField(field);
    const shared = state.config.provider_status.filter((other) =>
      other.provider_id !== providerId && other.settings_keys?.includes(key),
    );
    if (shared.length) {
      const note = document.createElement("div");
      note.className = "field-description";
      note.textContent = `Shared with ${shared.map(connectedAccountName).join(", ")}.`;
      wrapper.appendChild(note);
    }
    fields.appendChild(wrapper);
  });
  renderProviderDialogActions(provider);
  updateDirtyState();
  byId("providerDialog").showModal();
  const missing = provider.missing_configuration_keys?.[0];
  const first = missing ? byId(`field-${missing}`) : fields.querySelector("input:not(:disabled), select:not(:disabled), textarea:not(:disabled)");
  first?.focus();
}

function renderProviderDialogActions(provider) {
  if (state.providerId !== provider.provider_id) return;
  const actions = byId("providerDialogActions");
  actions.replaceChildren();
  const oauth = provider.kind === "connected_account";
  if (!oauth && !provider.missing_configuration_keys.length) {
    const button = authButton(provider.kind === "local" ? "Test" : "Refresh models", (target) => testProvider(provider.provider_id, target), "secondary-button");
    button.dataset.startupProvider = provider.provider_id;
    if (state.providerChecks.get(provider.provider_id)?.status === "checking") {
      button.dataset.operationBusy = "true";
      button.disabled = true;
      button.textContent = "Checking...";
    }
    actions.appendChild(button);
  }
  const check = oauth ? null : providerCheckResult(provider.provider_id);
  const result = byId("providerDialogCheck");
  result.className = `provider-check-result ${check?.status || ""}`;
  result.textContent = check?.message || "";
  result.hidden = !check?.message;
  byId("saveProvider").hidden = !provider.settings_keys?.length;
  renderStartup();
}

function connectedAccountName(provider) {
  return provider.display_name || provider.provider_id;
}

function connectedAccountMeta(provider, status) {
  if (!status) return "Checking account status…";
  const providerName = connectedAccountName(provider);
  if (status.state === "connecting") {
    if (status.mode === "device" && status.user_code && status.verification_url) {
      return `Enter code ${status.user_code} at ${status.verification_url}`;
    }
    return status.message || "Finish signing in, then return to this page.";
  }
  if (status.connected) {
    return providerCheckResult(provider.provider_id)?.message || "Checking models…";
  }
  return status.message || `Connect your ${providerName} account to discover models.`;
}

function populateConnectedAccountActions(provider, status, actions) {
  const providerId = provider.provider_id;
  if (!status) {
    const loading = authButton("Loading…", () => {}, "secondary-button startup-busy");
    loading.disabled = true;
    loading.setAttribute("aria-busy", "true");
    actions.appendChild(loading);
    return;
  }
  if (status.state === "connecting") {
    const target = status.authorization_url || status.verification_url;
    if (target) {
      actions.appendChild(authButton("Open sign-in", () => window.open(target, "_blank", "noopener")));
    }
    if (status.mode === "device" && status.user_code) {
      actions.appendChild(
        authButton("Copy code", () => copyDeviceCode(status.user_code), "secondary-button"),
      );
    }
    actions.appendChild(
      authButton("Cancel sign-in", () => cancelConnectedAccountLogin(providerId), "secondary-button"),
    );
    return;
  }
  if (status.connected) {
    actions.appendChild(authButton("Disconnect", () => disconnectConnectedAccount(providerId), "danger-button"));
    return;
  }
  const modes = Array.isArray(status.supported_login_modes) ? status.supported_login_modes : [];
  const defaultMode = status.default_login_mode;
  if (!["browser", "device"].includes(defaultMode) || !modes.includes(defaultMode)) {
    actions.appendChild(authButton("Retry", () => refreshConnectedAccount(provider)));
    return;
  }
  actions.appendChild(
    authButton(
      "Connect",
      (button) => startConnectedAccountLogin(providerId, defaultMode, button),
    ),
  );
}

function authButton(label, action, className = "primary-button") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", () => action(button));
  return button;
}

async function refreshConnectedAccounts() {
  const providers = (state.config?.provider_status || []).filter(
    (provider) => provider.kind === "connected_account",
  );
  await Promise.all(providers.map(refreshConnectedAccount));
}

async function refreshConnectedAccount(provider) {
  const config = state.config;
  clearConnectedAccountPoll(provider.provider_id);
  updateConnectedAccountCard(provider, null);
  try {
    const status = await api(`/admin/api/providers/${provider.provider_id}/auth`);
    if (config !== state.config) return;
    updateConnectedAccountCard(provider, status);
    if (status.state === "connecting") pollConnectedAccount(provider);
  } catch (error) {
    if (config !== state.config) return;
    updateConnectedAccountCard(provider, {
      state: "error",
      connected: null,
      message: error.message,
    });
  }
}

function updateConnectedAccountCard(provider, status) {
  if (status) void refreshStartup();
  if (JSON.stringify(state.authStatuses.get(provider.provider_id)) === JSON.stringify(status)) return;
  state.authStatuses.set(provider.provider_id, status);
  updateProviderCard(provider);
  renderProviderDialogActions(provider);
}

async function startConnectedAccountLogin(providerId, mode, button) {
  const buttons = button.closest(".provider-actions").querySelectorAll("button");
  buttons.forEach((action) => { action.disabled = true; });
  clearConnectedAccountPoll(providerId);
  const popup = mode === "browser" ? window.open("about:blank", "_blank") : null;
  if (popup) popup.opener = null;
  try {
    const status = await api(`/admin/api/providers/${providerId}/auth/login`, {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    const provider = connectedAccountDescriptor(providerId);
    updateConnectedAccountCard(provider, status);
    const target = status.authorization_url || status.verification_url;
    if (mode === "browser") {
      if (target && popup) {
        popup.location.replace(target);
      } else if (target) {
        window.open(target, "_blank", "noopener");
      } else if (popup) {
        popup.close();
      }
    }
    if (status.state === "connecting") pollConnectedAccount(provider);
    else if (status.connected) await hydrateModelOptions();
  } catch (error) {
    if (popup) popup.close();
    showMessage(error.message, true);
    buttons.forEach((action) => { action.disabled = false; });
  }
}

async function cancelConnectedAccountLogin(providerId) {
  clearConnectedAccountPoll(providerId);
  const provider = connectedAccountDescriptor(providerId);
  try {
    const status = await api(`/admin/api/providers/${providerId}/auth/cancel`, {
      method: "POST",
    });
    updateConnectedAccountCard(provider, status);
  } catch (error) {
    showMessage(error.message, true);
    pollConnectedAccount(provider);
  }
}

async function disconnectConnectedAccount(providerId) {
  const provider = connectedAccountDescriptor(providerId);
  if (!window.confirm(`Disconnect this ${connectedAccountName(provider)} account from FCC?`)) return;
  clearConnectedAccountPoll(providerId);
  try {
    const status = await api(`/admin/api/providers/${providerId}/auth`, { method: "DELETE" });
    updateConnectedAccountCard(provider, status);
    await hydrateModelOptions();
  } catch (error) {
    showMessage(error.message, true);
  }
}

function pollConnectedAccount(provider) {
  const providerId = provider.provider_id;
  clearConnectedAccountPoll(providerId);
  const poller = { timer: null };
  state.authPollers.set(providerId, poller);
  const poll = async () => {
    try {
      const status = await api(`/admin/api/providers/${providerId}/auth`);
      if (state.authPollers.get(providerId) !== poller) return;
      updateConnectedAccountCard(provider, status);
      if (status.state === "connecting") {
        poller.timer = window.setTimeout(poll, 1000);
      } else {
        state.authPollers.delete(providerId);
        if (status.connected) await hydrateModelOptions();
      }
    } catch (error) {
      if (state.authPollers.get(providerId) !== poller) return;
      state.authPollers.delete(providerId);
      showMessage(error.message, true);
    }
  };
  poller.timer = window.setTimeout(poll, 1000);
}

function clearConnectedAccountPoll(providerId) {
  const poller = state.authPollers.get(providerId);
  if (poller?.timer) window.clearTimeout(poller.timer);
  state.authPollers.delete(providerId);
}

function connectedAccountDescriptor(providerId) {
  return state.config.provider_status.find(
    (provider) => provider.provider_id === providerId,
  );
}

async function copyDeviceCode(code) {
  try {
    await navigator.clipboard.writeText(code);
    showMessage("Device code copied.");
  } catch {
    showMessage(`Copy this device code: ${code}`);
  }
}

function modelCountMessage(count) {
  return `${count} model${count === 1 ? "" : "s"} available`;
}

function providerCheckResult(providerId) {
  const check = state.providerChecks.get(providerId);
  // An explicit check takes precedence over background discovery and reachability.
  if (check?.source === "manual") return check;
  const discovery = state.startup?.startup?.providers?.[providerId];
  if (discovery === "starting") return { status: "checking", message: "Checking models…" };
  if (discovery === "failed") {
    return { status: "error", message: "Could not load models. Check the provider's settings and retry." };
  }
  if (discovery === "ready") {
    return { status: "ok", message: modelCountMessage(state.startup.cached_models[providerId]?.length || 0) };
  }
  return check;
}

function updateProviderCheckResult(providerId, status, message, source = "manual") {
  state.providerChecks.set(providerId, { status, message, source });
  renderProviderCheckResult(providerId);
}

function renderProviderCheckResult(providerId) {
  const { status = "", message = "" } = providerCheckResult(providerId) || {};
  const card = document.querySelector(`[data-provider="${providerId}"]`);
  if (!card) return;
  const provider = connectedAccountDescriptor(providerId);
  if (provider.kind === "connected_account") {
    const account = state.authStatuses.get(providerId);
    const meta = card.querySelector(".provider-meta");
    meta.textContent = connectedAccountMeta(provider, account);
    meta.className = "provider-meta";
    if (account?.connected && account.state !== "connecting") {
      meta.classList.add("provider-check-result", status || "checking");
    }
    if (account?.state === "error") meta.classList.add("error");
    return;
  }
  const result = card.querySelector(".provider-check-result");
  result.className = `provider-check-result ${status}`;
  result.textContent = message;
  result.hidden = !message;
  if (state.providerId === providerId) {
    const modalResult = byId("providerDialogCheck");
    modalResult.className = result.className;
    modalResult.textContent = message;
    modalResult.hidden = !message;
  }
}

function renderSections(sections, fields) {
  state.modelComboboxes.clear();
  VIEW_GROUPS.filter((view) => view.sections.length).forEach((view) => {
    byId(view.containerId).innerHTML = "";
  });

  const sectionById = new Map(sections.map((section) => [section.id, section]));
  const bySection = new Map();
  sections.forEach((section) => bySection.set(section.id, []));
  fields.forEach((field) => {
    if (!bySection.has(field.section)) bySection.set(field.section, []);
    bySection.get(field.section).push(field);
  });

  VIEW_GROUPS.forEach((view) => {
    const container = byId(view.containerId);
    view.sections.forEach((sectionId) => {
      const section = sectionById.get(sectionId);
      const sectionFields = bySection.get(sectionId) || [];
      if (!section || sectionFields.length === 0) return;

      const sectionEl = document.createElement("section");
      sectionEl.className = "settings-section";
      sectionEl.id = `section-${section.id}`;

      const heading = document.createElement("div");
      heading.className = "section-heading";
      heading.innerHTML = `<div><h3>${section.label}</h3><p>${section.description}</p></div>`;
      if (section.id === "models") {
        const refreshButton = document.createElement("button");
        refreshButton.type = "button";
        refreshButton.className = "secondary-button";
        refreshButton.textContent = "Refresh models";
        refreshButton.dataset.startupCatalog = "true";
        refreshButton.addEventListener("click", () => refreshModelOptions(refreshButton));
        heading.appendChild(refreshButton);
      }
      sectionEl.appendChild(heading);

      const grid = document.createElement("div");
      grid.className = "field-grid";
      sectionFields.forEach((field) => {
        grid.appendChild(renderField(field));
      });
      sectionEl.appendChild(grid);

      if (sectionFields.some((field) => field.advanced)) {
        const toggle = document.createElement("button");
        toggle.type = "button";
        toggle.className = "ghost-button advanced-toggle";
        toggle.textContent = "Show advanced";
        toggle.addEventListener("click", () => {
          const showing = sectionEl.classList.toggle("show-advanced");
          toggle.textContent = showing ? "Hide advanced" : "Show advanced";
        });
        sectionEl.appendChild(toggle);
      }

      container.appendChild(sectionEl);
    });
  });
}

function renderField(field) {
  const wrapper = document.createElement("div");
  wrapper.className = `field${field.advanced ? " advanced-field" : ""}`;
  wrapper.dataset.key = field.key;

  const label = document.createElement("label");
  label.htmlFor = `field-${field.key}`;
  const labelText = document.createElement("span");
  labelText.textContent = field.label;
  label.appendChild(labelText);

  const source = sourceText(field);
  if (source) {
    const sourceEl = document.createElement("span");
    sourceEl.className = "field-source";
    sourceEl.textContent = source;
    label.appendChild(sourceEl);
  }

  const input = window.FccFormControls.configure(inputForField(field));
  input.id = `field-${field.key}`;
  input.dataset.key = field.key;
  input.dataset.original = comparableValue(field.value);
  input.dataset.secret = field.secret ? "true" : "false";
  input.dataset.configured = field.configured ? "true" : "false";
  input.dataset.nullable = field.nullable ? "true" : "false";
  input.dataset.remove = "false";
  input.dataset.fieldType = field.type;
  input.disabled = field.locked;
  input.addEventListener("input", updateDirtyState);
  input.addEventListener("change", updateDirtyState);
  input.addEventListener("input", () => {
    input.dataset.remove = "false";
    clearCredentialError(input);
  });
  if (field.type === "optional_model") {
    input.addEventListener("blur", () => {
      if (!input.value.trim() || input.value.trim().toLowerCase() === "none") {
        input.value = "None";
        updateDirtyState();
      }
    });
  }

  let control = input;
  if (field.type === "model" || field.type === "optional_model") {
    control = createModelCombobox(input, field).element;
  } else if (field.type === "model_list") {
    const editor = new ModelListEditor(input, field);
    label.htmlFor = editor.inputId;
    control = editor.element;
  }
  wrapper.append(label, control);
  if (field.secret && field.nullable && field.configured && !field.locked) {
    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "ghost-button secret-remove";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", () => {
      const removing = input.dataset.remove !== "true";
      input.dataset.remove = removing ? "true" : "false";
      input.readOnly = removing;
      removeButton.textContent = removing ? "Undo removal" : "Remove";
      clearCredentialError(input);
      updateDirtyState();
    });
    wrapper.appendChild(removeButton);
  }
  if (field.description) {
    const description = document.createElement("div");
    description.className = "field-description";
    description.textContent = field.description;
    wrapper.appendChild(description);
  }
  return wrapper;
}

function inputForField(field) {
  if (field.type === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = String(field.value).toLowerCase() === "true";
    input.dataset.original = input.checked ? "true" : "false";
    return input;
  }

  if (field.type === "select") {
    const select = document.createElement("select");
    field.options.forEach((item) =>
      select.appendChild(option(item.value, item.label)),
    );
    select.value = field.value || field.options[0]?.value || "";
    return select;
  }

  if (field.type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.value = field.value || "";
    return textarea;
  }

  if (field.type === "model" || field.type === "optional_model") {
    const input = document.createElement("input");
    input.type = "text";
    input.value = field.value || (field.type === "optional_model" ? "None" : "");
    input.autocomplete = "off";
    return input;
  }

  if (field.type === "model_list") {
    const input = document.createElement("input");
    input.type = "hidden";
    input.value = field.value || "";
    return input;
  }

  const input = document.createElement("input");
  input.type = field.type === "number" ? "number" : "text";
  if (field.type === "secret") {
    input.setAttribute("autocapitalize", "none");
    input.spellcheck = false;
    input.setAttribute("autocorrect", "off");
    input.placeholder = field.configured
      ? "Configured - enter a new value to replace"
      : "Not configured";
    input.value = "";
    input.autocomplete = "off";
  } else {
    input.value = field.value || "";
  }
  return input;
}

function createModelCombobox(input, field) {
  return new window.FccModelCombobox(input, {
    listboxId: `model-options-${field.key}`,
    label: field.label,
    values: () =>
      field.type === "optional_model"
        ? ["None", ...state.modelOptions]
        : state.modelOptions,
    emptyMessage: () =>
      state.modelOptions.length
        ? "No matching models. You can still enter a custom slug."
        : "No discovered models. Refresh models or enter a custom slug.",
    registry: state.modelComboboxes,
  });
}

class ModelListEditor {
  constructor(input, field) {
    this.input = input;
    this.field = field;
    this.values = input.value
      ? input.value.split(",").map((value) => value.trim()).filter(Boolean)
      : [];
    this.inputId = `field-${field.key}-add`;

    this.element = document.createElement("div");
    this.element.className = "model-list-editor";

    const addRow = document.createElement("div");
    addRow.className = "model-list-add";
    this.addInput = window.FccFormControls.configure(document.createElement("input"));
    this.addInput.id = this.inputId;
    this.addInput.type = "text";
    this.addInput.autocomplete = "off";
    this.addInput.placeholder = "provider/model";
    this.addInput.disabled = field.locked;
    const addCombobox = createModelCombobox(this.addInput, {
      ...field,
      key: `${field.key}-add`,
      label: "fallback model",
      type: "model",
    });

    this.addButton = document.createElement("button");
    this.addButton.type = "button";
    this.addButton.className = "secondary-button";
    this.addButton.textContent = "Add";
    this.addButton.disabled = field.locked;
    this.addButton.addEventListener("click", () => this.add());
    addRow.append(addCombobox.element, this.addButton);

    this.rows = document.createElement("div");
    this.rows.className = "model-list-rows";
    this.element.append(input, addRow, this.rows);
    this.renderRows();
  }

  add() {
    const value = this.addInput.value.trim();
    if (!value) {
      showMessage("Enter a full provider/model fallback.", "error");
      return;
    }
    if (this.values.includes(value)) {
      showMessage("That fallback model is already in the list.", "error");
      return;
    }
    this.values.push(value);
    this.addInput.value = "";
    showMessage("");
    this.sync();
  }

  move(index, offset) {
    const destination = index + offset;
    if (destination < 0 || destination >= this.values.length) return;
    [this.values[index], this.values[destination]] = [
      this.values[destination],
      this.values[index],
    ];
    this.sync();
  }

  remove(index) {
    this.values.splice(index, 1);
    this.sync();
  }

  sync() {
    this.input.value = this.values.join(",");
    this.input.dataset.remove = "false";
    this.input.dispatchEvent(new Event("input", { bubbles: true }));
    this.renderRows();
  }

  renderRows() {
    this.rows.innerHTML = "";
    if (this.values.length === 0) {
      const empty = document.createElement("div");
      empty.className = "model-list-empty";
      empty.textContent = "No fallback models configured.";
      this.rows.appendChild(empty);
      return;
    }

    this.values.forEach((value, index) => {
      const row = document.createElement("div");
      row.className = "model-list-row";

      const model = document.createElement("span");
      model.className = "model-list-value";
      model.textContent = value;

      const up = this.actionButton("Move up", `Move ${value} up`, () =>
        this.move(index, -1),
      );
      up.disabled = this.field.locked || index === 0;
      const down = this.actionButton("Move down", `Move ${value} down`, () =>
        this.move(index, 1),
      );
      down.disabled = this.field.locked || index === this.values.length - 1;
      const remove = this.actionButton("Remove", `Remove ${value}`, () =>
        this.remove(index),
      );
      remove.disabled = this.field.locked;

      row.append(model, up, down, remove);
      this.rows.appendChild(row);
    });
  }

  actionButton(text, label, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button model-list-action";
    button.textContent = text;
    button.setAttribute("aria-label", label);
    button.addEventListener("click", action);
    return button;
  }
}

function option(value, label) {
  const optionEl = document.createElement("option");
  optionEl.value = value;
  optionEl.textContent = label;
  return optionEl;
}

function readFieldValue(input) {
  if (input.type === "checkbox") return input.checked ? "true" : "false";
  if (input.dataset.remove === "true") return null;
  if (
    input.dataset.fieldType === "optional_model" &&
    input.value.trim().toLowerCase() === "none"
  ) {
    return null;
  }
  if (input.dataset.secret === "true" && input.dataset.configured === "true") {
    return input.value ? input.value : MASKED_SECRET;
  }
  if (input.dataset.nullable === "true" && !input.value.trim()) return null;
  return input.value;
}

function comparableValue(value) {
  return value === null ? NULL_VALUE : String(value);
}

function changedValues(root = byId("adminViews")) {
  const values = {};
  root.querySelectorAll("[data-key]").forEach((input) => {
    if (input.disabled || !input.matches("input, select, textarea")) return;
    const value = readFieldValue(input);
    if (comparableValue(value) !== input.dataset.original) {
      values[input.dataset.key] = value;
    }
  });
  return values;
}

function updateDirtyState() {
  const count = Object.keys(changedValues()).length;
  byId("dirtyState").textContent =
    state.restart ? "Changes saved" : count === 0 ? "No changes" : `${count} unsaved change${count === 1 ? "" : "s"}`;
  byId("applyButton").disabled = state.applying || (!state.restart && count === 0);
  byId("saveProvider").disabled = state.applying || !!state.restart || !Object.keys(changedValues(byId("providerFields"))).length;
}

function clearCredentialError(input) {
  byId(`${input.id}-error`)?.remove();
  input.removeAttribute("aria-invalid");
  input.removeAttribute("aria-describedby");
}

function showCredentialErrors(checks) {
  let first = null;
  checks.forEach((check) => {
    const input = byId(`field-${check.key}`);
    if (!input) return;
    clearCredentialError(input);
    if (check.status !== "rejected") return;
    const error = document.createElement("div");
    error.id = `${input.id}-error`;
    error.className = "field-error";
    error.textContent = check.message;
    input.closest(".field").appendChild(error);
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", error.id);
    first ||= input;
  });
  return first;
}

function setApplying(applying) {
  state.applying = applying;
  VIEW_GROUPS.filter((view) => view.id !== "code").forEach((view) => {
    byId(`view-${view.id}`).inert = applying || !!state.restart;
  });
  if (applying) state.modelComboboxes.forEach((combobox) => combobox.close());
  byId("providerDialogBody").inert = applying || !!state.restart;
  byId("closeProviderDialog").disabled = applying;
  byId("cancelProviderDialog").disabled = applying;
  byId("saveProvider").textContent = applying ? "Saving…" : "Save";
  byId("saveProvider").setAttribute("aria-busy", String(applying));
  byId("applyButton").textContent = state.restart
    ? applying ? "Reconnecting…" : "Reconnect"
    : applying ? "Applying…" : "Apply";
  updateDirtyState();
}

async function waitForRestart(restart, target) {
  const deadline = performance.now() + 30_000;
  const statusUrl = new URL("/admin/api/status", target);
  while (performance.now() < deadline) {
    try {
      const response = await fetch(statusUrl, {
        cache: "no-store",
        credentials: "omit",
        signal: AbortSignal.timeout(1500),
      });
      if (response.ok) {
        const status = await response.json();
        if (status.status === "running" && typeof status.instance_id === "string"
          && status.instance_id !== restart.instance_id) return;
      }
    } catch {
      // Closing listeners and unfinished startup are expected during a restart.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("The server has not reconnected yet.");
}

function appendAdminLink(target) {
  const link = document.createElement("a");
  link.href = target.href;
  link.textContent = "Open Admin";
  byId("messageArea").append(document.createElement("br"), link);
}

async function reconnectAfterRestart() {
  clearTimeout(state.startupTimer);
  state.startupTimer = null;
  state.startupRequest?.controller.abort();
  state.startupRequest = null;
  state.startupAgain = false;
  const { restart, warnings, providersOnly } = state.restart;
  const target = new URL(restart.admin_url || "/admin", window.location.href);
  setApplying(true);
  showMessage(["Applied. Reconnecting to the server…", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
  try {
    await waitForRestart(restart, target);
    if (target.origin !== window.location.origin) {
      // Carry only the safe warning text across an address change, never edits.
      target.hash = new URLSearchParams({ "fcc-applied": JSON.stringify(warnings) }).toString();
      window.location.replace(target.href);
      return;
    }
    await load({ providersOnly });
    state.restart = null;
    void refreshStartup();
    showMessage(["Applied", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
  } catch (error) {
    showMessage([`Settings were saved. ${error.message} Use Reconnect to try again.`, ...warnings].join("\n"), "warn");
    appendAdminLink(target);
  } finally {
    setApplying(false);
  }
}

function showRestartNotice() {
  const url = new URL(window.location.href);
  const fragment = new URLSearchParams(url.hash.slice(1));
  const notice = fragment.get("fcc-applied");
  if (notice === null) return;
  fragment.delete("fcc-applied");
  url.hash = fragment.toString();
  window.history.replaceState(window.history.state, "", url);
  try {
    const warnings = JSON.parse(notice);
    if (Array.isArray(warnings) && warnings.every((warning) => typeof warning === "string")) {
      showMessage(["Applied", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
    }
  } catch {
    // A malformed navigation notice must not prevent normal Admin use.
  }
}

async function apply(providerId = null) {
  if (state.applying) return;
  if (state.restart) {
    await reconnectAfterRestart();
    return;
  }
  const values = changedValues(providerId ? byId("providerFields") : byId("adminViews"));
  if (!Object.keys(values).length) return;
  const checkingKeys = Object.keys(values).some((key) => {
    const field = state.fields.get(key);
    return field?.secret && field.section === "providers" && values[key] !== null;
  });
  let rejectedField = null;
  let applied = false;
  setApplying(true);
  showMessage(checkingKeys ? "Checking API keys…" : "Applying…");
  try {
    const result = await api("/admin/api/config/apply", {
      method: "POST",
      body: JSON.stringify({ values }),
    });
    const checks = result.credential_checks || [];
    if (!result.applied) {
      rejectedField = showCredentialErrors(checks);
      showMessage(rejectedField ? "Not applied. Check the highlighted API keys." : result.errors.join("; "), "error");
      return;
    }
    applied = true;
    if (providerId) byId("providerDialog").close();
    const warnings = checks.filter((check) => check.status === "unverified").map((check) =>
      `${state.fields.get(check.key)?.label || check.key}: ${check.message}`
    );
    const restart = result.restart || {};
    if (restart.required && restart.automatic) {
      state.restart = { restart, warnings, providersOnly: !!providerId };
      await reconnectAfterRestart();
      return;
    }
    const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
    await load({ providersOnly: !!providerId });
    const message = pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Applied";
    showMessage([message, ...warnings].join("\n"), warnings.length ? "warn" : "ok");
  } catch (error) {
    showMessage(applied ? `Applied, but could not reload settings: ${error.message}` : `Could not apply settings: ${error.message}`, "error");
  } finally {
    setApplying(false);
    if (rejectedField) {
      if (!providerId) navigateToView("providers");
      rejectedField.closest(".settings-section")?.classList.add("show-advanced");
      rejectedField.scrollIntoView({ block: "center", behavior: "instant" });
      rejectedField.focus();
    } else if (providerId && applied) {
      document.querySelector(`[data-provider="${providerId}"] [data-provider-settings]`)?.focus({ preventScroll: true });
    }
  }
}

async function refreshLocalStatus(config) {
  const request = {
    providerIds: new Set(config.provider_status.filter((provider) =>
      provider.kind === "local" && provider.status === "configured",
    ).map((provider) => provider.provider_id)),
  };
  state.localStatusRequest = request;
  try {
    const result = await api("/admin/api/providers/local-status");
    if (state.localStatusRequest !== request) return;
    result.providers.forEach((provider) => {
      if (!request.providerIds.has(provider.provider_id) || provider.status === "missing_url") return;
      if (provider.status === "reachable") {
        updateProviderCheckResult(
          provider.provider_id,
          "ok",
          `Reachable: ${provider.base_url}`,
          "availability",
        );
        return;
      }
      const detail = provider.message
        ? provider.message
        : provider.status_code
          ? `${provider.base_url} returned HTTP ${provider.status_code}`
          : "The local provider did not respond.";
      updateProviderCheckResult(
        provider.provider_id,
        "error",
        `Unavailable: ${detail}`,
        "availability",
      );
    });
  } catch {
    if (state.localStatusRequest !== request) return;
    request.providerIds.forEach((providerId) => {
      updateProviderCheckResult(
        providerId,
        "error",
        "Availability check failed. Use Test to retry.",
        "availability",
      );
    });
  } finally {
    if (state.localStatusRequest === request) state.localStatusRequest = null;
  }
}

async function testProvider(providerId, button) {
  const config = state.config;
  state.localStatusRequest?.providerIds.delete(providerId);
  const original = button.textContent;
  button.dataset.operationBusy = "true";
  button.disabled = true;
  button.textContent = "Checking...";
  updateProviderCheckResult(providerId, "checking", "Checking...");
  try {
    const result = await api(`/admin/api/providers/${providerId}/test`, {
      method: "POST",
      body: "{}",
    });
    if (config !== state.config) return;
    if (result.ok) {
      updateProviderCheckResult(
        providerId,
        "ok",
        modelCountMessage(result.models.length),
      );
      await hydrateModelOptions();
    } else {
      updateProviderCheckResult(
        providerId,
        "error",
        `Unavailable: ${result.message || "Provider check failed."}`,
      );
    }
  } catch {
    if (config !== state.config) return;
    updateProviderCheckResult(
      providerId,
      "error",
      "Provider check could not be completed.",
    );
  } finally {
    delete button.dataset.operationBusy;
    button.disabled = false;
    button.textContent = original;
    if (config === state.config) renderProviderDialogActions(connectedAccountDescriptor(providerId));
    void refreshStartup();
    renderStartup();
  }
}

async function hydrateModelOptions() {
  try {
    await loadModelOptions();
  } catch {
    // Model fields remain editable when optional catalog hydration is unavailable.
  }
}

async function loadModelOptions(refresh = false) {
  const request = ++state.modelOptionsRequest;
  const config = state.config;
  const result = await api("/admin/api/models" + (refresh ? "/refresh" : ""), {
    method: refresh ? "POST" : "GET",
  });
  if (request === state.modelOptionsRequest && config === state.config) setModelOptions(result.models);
  if (refresh && window.CodeSessions) await window.CodeSessions.refresh();
  return result;
}

async function refreshModelOptions(button) {
  const original = button.textContent;
  button.dataset.operationBusy = "true";
  button.disabled = true;
  button.textContent = "Refreshing";
  try {
    const result = await loadModelOptions(true);
    const failedProviders = result.failed_providers || [];
    if (failedProviders.length) {
      const labels = failedProviders.map(providerDisplayName).join(", ");
      showMessage(
        `${state.modelOptions.length} models available; could not refresh ${labels}`,
        "warn",
      );
    } else {
      showMessage(`${state.modelOptions.length} models available`, "ok");
    }
  } catch (error) {
    showMessage(`Could not refresh models: ${error.message}`, "error");
  } finally {
    delete button.dataset.operationBusy;
    button.disabled = false;
    button.textContent = original;
    void refreshStartup();
    renderStartup();
  }
}

function providerDisplayName(providerId) {
  const provider = state.config?.provider_status?.find(
    (candidate) => candidate.provider_id === providerId,
  );
  return provider?.display_name || providerId;
}

function setModelOptions(models) {
  state.modelOptions = Array.from(
    new Set(models.filter((model) => typeof model === "string" && model.trim())),
  );
  state.modelComboboxes.forEach((combobox) => {
    if (combobox.isOpen) combobox.render(combobox.query);
  });
}

function showMessage(message, kind = "") {
  const area = byId("messageArea");
  area.textContent = message;
  area.className = `message-area ${kind}`.trim();
  if (byId("providerDialog").open) {
    byId("providerMessage").textContent = message;
    byId("providerMessage").className = area.className;
  }
}

byId("applyButton").addEventListener("click", () => apply());
byId("saveProvider").addEventListener("click", () => apply(state.providerId));
byId("closeProviderDialog").addEventListener("click", () => byId("providerDialog").close());
byId("cancelProviderDialog").addEventListener("click", () => byId("providerDialog").close());
byId("providerDialog").addEventListener("cancel", (event) => {
  if (state.applying) event.preventDefault();
});
byId("providerDialog").addEventListener("click", (event) => {
  if (event.target !== event.currentTarget || state.applying) return;
  const rect = event.currentTarget.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.currentTarget.close();
});
byId("providerDialog").addEventListener("close", () => {
  if (byId("providerDialog").open) return;
  const providerId = state.providerId;
  state.providerId = null;
  byId("providerFields").replaceChildren();
  updateDirtyState();
  document.querySelector(`[data-provider="${providerId}"] [data-provider-settings]`)?.focus({ preventScroll: true });
});
document.addEventListener("pointerdown", (event) => {
  state.modelComboboxes.forEach((combobox) => {
    if (combobox.isOpen && !combobox.element.contains(event.target)) combobox.close();
  });
});

window.addEventListener("popstate", () => {
  const viewId = viewFromLocation();
  setActiveView(viewId, { scroll: false });
});

try {
  for (const key of Object.keys(sessionStorage)) {
    if (key.startsWith("fcc.chat.draft.")) sessionStorage.removeItem(key);
  }
} catch {
  console.warn("Chat draft cleanup deferred until the next page load: storage unavailable");
}

const claudeIntegrationDialog = byId("claudeIntegrationDialog");
const claudeIntegration = { connected: null, busy: false, paths: null };
const claudeIntegrationPath = "/admin/api/integrations/claude-vscode";

function integrationMessage(id, message, error = false) {
  const element = byId(id);
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("error", error);
}

function renderClaudeIntegration() {
  const { connected, busy, paths } = claudeIntegration;
  const action = connected ? "Disconnect" : "Connect";
  byId("openClaudeIntegration").textContent = busy ? "Loading…" : connected === null ? "Retry" : action;
  byId("openClaudeIntegration").disabled = busy;
  byId("openClaudeIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmClaudeIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmClaudeIntegration").disabled = busy || connected === null;
  byId("openClaudeIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmClaudeIntegration").className = connected ? "danger-button" : "primary-button";
  byId("claudeIntegrationDescription").textContent = connected
    ? "Remove FCC's VS Code settings. Claude onboarding stays completed."
    : "Will set FCC's URL and token, enable model discovery, skip VS Code login, and complete Claude onboarding.";
  const files = byId("claudeIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = connected ? [paths.vscode_settings] : [paths.vscode_settings, paths.claude_state];
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshClaudeIntegration() {
  if (claudeIntegration.busy) return;
  claudeIntegration.busy = true;
  renderClaudeIntegration();
  integrationMessage("claudeIntegrationMessage", "");
  try {
    const result = await api(claudeIntegrationPath);
    claudeIntegration.connected = result.connected;
    claudeIntegration.paths = result.paths;
  } catch (error) {
    claudeIntegration.connected = null;
    integrationMessage("claudeIntegrationMessage", error.message, true);
  } finally {
    claudeIntegration.busy = false;
    renderClaudeIntegration();
  }
}

byId("openClaudeIntegration").addEventListener("click", () => {
  if (claudeIntegration.connected === null) {
    refreshClaudeIntegration();
    return;
  }
  integrationMessage("claudeIntegrationDialogMessage", "");
  claudeIntegrationDialog.showModal();
});
byId("confirmClaudeIntegration").addEventListener("click", async () => {
  if (claudeIntegration.busy || claudeIntegration.connected === null) return;
  const disconnect = claudeIntegration.connected;
  claudeIntegration.busy = true;
  renderClaudeIntegration();
  integrationMessage("claudeIntegrationDialogMessage", "");
  integrationMessage("claudeIntegrationMessage", "");
  try {
    const result = await api(`${claudeIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    claudeIntegration.connected = result.connected;
    claudeIntegrationDialog.close();
    integrationMessage("claudeIntegrationMessage", disconnect
      ? "Settings removed. Reload VS Code to disconnect."
      : "Settings saved. Reload VS Code to connect.");
  } catch (error) {
    integrationMessage("claudeIntegrationDialogMessage", error.message, true);
    integrationMessage("claudeIntegrationMessage", error.message, true);
  } finally {
    claudeIntegration.busy = false;
    renderClaudeIntegration();
  }
});
byId("closeClaudeIntegration").addEventListener("click", () => claudeIntegrationDialog.close());
claudeIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== claudeIntegrationDialog) return;
  const bounds = claudeIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    claudeIntegrationDialog.close();
  }
});

const codexIntegrationDialog = byId("codexIntegrationDialog");
const codexIntegration = { connected: null, busy: false, paths: null };
const codexIntegrationPath = "/admin/api/integrations/codex";

function renderCodexIntegration() {
  const { connected, paths } = codexIntegration;
  const initializing = connected === false && state.startup?.startup?.catalog_file === "starting";
  const unavailable = connected === false && state.startup?.startup?.catalog_file === "failed";
  const busy = codexIntegration.busy || initializing;
  const catalogError = "Could not prepare the Codex model catalog. Refresh models to retry.";
  if (unavailable) integrationMessage("codexIntegrationMessage", catalogError, true);
  else if (byId("codexIntegrationMessage").textContent === catalogError) integrationMessage("codexIntegrationMessage", "");
  const action = connected ? "Disconnect" : "Connect";
  byId("openCodexIntegration").textContent = busy ? "Loading…" : connected === null ? "Retry" : action;
  byId("openCodexIntegration").disabled = busy;
  byId("openCodexIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmCodexIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmCodexIntegration").disabled = busy || connected === null || unavailable;
  byId("openCodexIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmCodexIntegration").className = connected ? "danger-button" : "primary-button";
  byId("codexIntegrationDescription").textContent = connected
    ? "Remove FCC's Codex configuration. Other settings stay unchanged."
    : "Configure Codex to use FCC. Your selected model stays unchanged.";
  const files = byId("codexIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = [paths.codex_config];
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshCodexIntegration() {
  if (codexIntegration.busy) return;
  codexIntegration.busy = true;
  renderCodexIntegration();
  integrationMessage("codexIntegrationMessage", "");
  try {
    const result = await api(codexIntegrationPath);
    codexIntegration.connected = result.connected;
    codexIntegration.paths = result.paths;
  } catch (error) {
    codexIntegration.connected = null;
    integrationMessage("codexIntegrationMessage", error.message, true);
  } finally {
    codexIntegration.busy = false;
    renderCodexIntegration();
  }
}

byId("openCodexIntegration").addEventListener("click", () => {
  if (codexIntegration.connected === null) {
    refreshCodexIntegration();
    return;
  }
  integrationMessage("codexIntegrationDialogMessage", "");
  codexIntegrationDialog.showModal();
});
byId("confirmCodexIntegration").addEventListener("click", async () => {
  if (codexIntegration.busy || codexIntegration.connected === null) return;
  const disconnect = codexIntegration.connected;
  codexIntegration.busy = true;
  renderCodexIntegration();
  integrationMessage("codexIntegrationDialogMessage", "");
  integrationMessage("codexIntegrationMessage", "");
  try {
    const result = await api(`${codexIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    codexIntegration.connected = result.connected;
    codexIntegration.paths = result.paths;
    codexIntegrationDialog.close();
    integrationMessage("codexIntegrationMessage", disconnect
      ? "Settings removed. Restart Codex to disconnect."
      : "Settings saved. Restart Codex and select an FCC model.");
  } catch (error) {
    integrationMessage("codexIntegrationDialogMessage", error.message, true);
    integrationMessage("codexIntegrationMessage", error.message, true);
  } finally {
    codexIntegration.busy = false;
    renderCodexIntegration();
  }
});
byId("closeCodexIntegration").addEventListener("click", () => codexIntegrationDialog.close());
codexIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== codexIntegrationDialog) return;
  const bounds = codexIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    codexIntegrationDialog.close();
  }
});

const jetBrainsIntegrationDialog = byId("jetBrainsIntegrationDialog");
byId("openJetBrainsIntegration").addEventListener("click", () => jetBrainsIntegrationDialog.showModal());
byId("closeJetBrainsIntegration").addEventListener("click", () => jetBrainsIntegrationDialog.close());
jetBrainsIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== jetBrainsIntegrationDialog) return;
  const bounds = jetBrainsIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    jetBrainsIntegrationDialog.close();
  }
});

// Keep footer clearance exact when messages wrap or a view hides the bar.
new ResizeObserver(([entry]) => {
  document.documentElement.style.setProperty(
    "--action-bar-height",
    `${entry.target.getBoundingClientRect().height}px`,
  );
}).observe(document.querySelector(".action-bar"), { box: "border-box" });

load().then(showRestartNotice).catch((error) => {
  showMessage(error.message, "error");
});

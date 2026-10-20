const adminAssetBase = new URL(".", document.currentScript.src);

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
  customProvider: null,
  configRefreshRequired: false,
  modelLabels: new Map(),
  localStatusRequest: null,
  startup: null,
  startupRequest: null,
  startupTimer: null,
  startupAgain: false,
  configRequest: 0,
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
    sections: ["models", "web_tools"],
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
    const visible = state.activeView === "messaging" && (["starting", "failed"].includes(messaging?.state) || Boolean(messaging?.warning));
    message.hidden = !visible;
    message.classList.toggle("startup-spinner", visible && messaging.state === "starting");
    message.classList.toggle("error", messaging?.state === "failed");
    message.textContent = messaging?.state === "failed" ? messaging.message || "Messaging could not start." : messaging?.warning || "";
    message.setAttribute("aria-label", messaging?.state === "starting" ? "Messaging is starting" : "Messaging status");
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
  renderClaudeIntegration();
  renderVSCodeChatIntegration();
  renderJetBrainsIntegration();
  renderCodexIntegration();
  renderClaudeDesktopIntegration();
  renderDshDesktopIntegration();
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
    refreshIntegrationUpdates(
      previous?.instance_id === result.instance_id ? previous.startup.integrations : null,
      current.integrations,
    );
    const changed = !previous || previous.instance_id !== result.instance_id ||
      JSON.stringify(previous.startup) !== JSON.stringify(current);
    if (changed) void hydrateModelOptions();
    if (changed) void window.CodeSessions?.refresh(result);
    pending = current.catalog === "starting" || current.catalog_file === "starting" ||
      current.code?.state === "starting" || current.messaging?.state === "starting" ||
      Object.values(current.integrations || {}).some((update) => update.state === "starting") ||
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
  const request = ++state.configRequest;
  state.localStatusRequest = null;
  state.startupRequest?.controller.abort();
  state.startupRequest = null;
  showMessage("Loading admin config");
  const config = await api("/admin/api/config");
  if (request !== state.configRequest) return;
  config.custom_providers ||= [];
  config.provider_status.push(...config.custom_providers.map((provider) => ({ ...provider, kind: "custom", status: "configured", settings_keys: [], missing_configuration_keys: [] })));
  state.config = config;
  state.configRefreshRequired = false;
  window.CodeSessions.setProviderNames(config.custom_providers);
  state.startup = null;
  state.fields = new Map(config.fields.map((field) => [field.key, field]));
  if (!providersOnly) renderNav();
  state.providerChecks.clear();
  renderProviders(config.provider_status);
  if (!providersOnly) renderSections(config.sections, config.fields);
  document.querySelectorAll(".model-list-editor").forEach((element) => element.modelListEditor.refreshKeyOptions());
  byId("configPath").textContent = config.paths.managed;
  void refreshLocalStatus(config);
  void refreshStartup();
  void refreshConnectedAccounts();
  void hydrateModelOptions();
  if (providersOnly) void window.CodeSessions.refresh();
  else window.CodeSessions.initialize(api);
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
  if (activeView.id === "code") void window.CodeSessions.refresh();
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
    refreshClaudeOnboarding();
    refreshVSCodeChatIntegration();
    refreshJetBrainsIntegration();
    refreshCodexIntegration();
    refreshClaudeDesktopIntegration();
    refreshDshDesktopIntegration();
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
  const custom = document.createElement("section");
  custom.className = "provider-strip";
  custom.dataset.providerGroup = "custom";
  const heading = document.createElement("h3");
  heading.textContent = "Custom providers";
  const add = authButton("Add provider", () => openCustomProviderDialog(), "primary-button");
  add.id = "addCustomProvider";
  add.disabled = !!state.config.custom_providers_locked;
  const empty = document.createElement("p");
  empty.textContent = "No custom providers yet.";
  empty.hidden = state.config.custom_providers.length > 0;
  const grid = document.createElement("div");
  grid.className = "provider-grid";
  grid.id = "providers-custom";
  const header = document.createElement("div");
  header.className = "strip-header";
  header.append(heading, add);
  custom.append(header, empty, grid);
  container.appendChild(custom);
  providerStatus.forEach(updateProviderCard);
}

function updateProviderCard(provider) {
  const oauth = provider.kind === "connected_account";
  const status = state.authStatuses.get(provider.provider_id);
  const kind = oauth ? "oauth" : provider.kind === "local" ? "local" : "cloud";
  const configured = oauth ? status?.connected : provider.status === "configured";
  const subgroup = oauth && configured == null ? "loading" : configured ? "configured" : "unconfigured";
  const grid = byId(provider.kind === "custom" ? "providers-custom" : `providers-${kind}-${subgroup}`);
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
  const website = document.createElement("a");
  website.href = provider.website_url;
  website.target = "_blank";
  website.rel = "noopener noreferrer";
  const logo = document.createElement("img");
  logo.className = "provider-logo";
  if (provider.logo_filename) logo.src = new URL(`providers/${provider.logo_filename}`, adminAssetBase);
  logo.alt = "";
  logo.width = 20;
  logo.height = 20;
  website.append(name, logo);
  title.appendChild(provider.kind === "custom" ? name : website);
  const meta = document.createElement("span");
  meta.className = "provider-meta";
  meta.hidden = !oauth;
  const result = document.createElement("span");
  result.className = "provider-check-result";
  result.dataset.providerCheckResult = provider.provider_id;
  result.hidden = true;
  const availability = document.createElement("span");
  availability.className = "provider-availability-result";
  availability.dataset.providerAvailabilityResult = provider.provider_id;
  availability.hidden = true;
  const actions = document.createElement("div");
  actions.className = "provider-actions";
  if (oauth) populateConnectedAccountActions(provider, status, actions);
  if (provider.kind === "custom" || provider.settings_keys?.length) {
    const edit = oauth || configured;
    const settings = authButton(edit ? "Edit" : "Configure", () => openProviderDialog(provider.provider_id), edit ? "secondary-button" : "primary-button");
    settings.dataset.providerSettings = "true";
    settings.setAttribute("aria-haspopup", "dialog");
    settings.setAttribute("aria-controls", "providerDialog");
    actions.appendChild(settings);
  }
  card.replaceChildren(title, meta, result, availability, actions);
  const next = [...grid.children].find((other) => other !== card &&
    connectedAccountName(provider).localeCompare(providerDisplayName(other.dataset.provider), "en", { sensitivity: "base" }) < 0);
  if (card.parentElement !== grid || card.nextElementSibling !== (next || null)) grid.insertBefore(card, next || null);
  document.querySelectorAll("[data-provider-subgroup]").forEach((section) => {
    section.hidden = section.querySelector(".provider-grid").childElementCount === 0;
  });
  if (focused) {
    const controls = [...card.querySelectorAll("a[href], button:not(:disabled)")];
    (controls.find((control) => control.textContent === focusedLabel) || actions.querySelector("button:not(:disabled)"))?.focus({ preventScroll: true });
  }
  renderProviderCheckResult(provider.provider_id);
}

async function refreshProviderConfig() {
  if (!state.configRefreshRequired) return true;
  try { await load({ providersOnly: true }); return true; }
  catch (error) { showMessage(`Could not reload settings: ${error.message}`, "error"); return false; }
}

async function openProviderDialog(providerId) {
  if (!await refreshProviderConfig()) return;
  if (connectedAccountDescriptor(providerId)?.kind === "custom") return openCustomProviderDialog(providerId);
  state.customProvider = null;
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
  if (!byId("providerDialog").open) byId("providerDialog").showModal();
  const missing = provider.missing_configuration_keys?.[0];
  const target = missing ? byId(`field-${missing}`) : null;
  const first = target?.dataset.fieldType === "api_key_list"
    ? target.closest(".api-key-editor").querySelector("input:not([type=hidden]):not(:disabled), button:not(:disabled)")
    : target || fields.querySelector("input:not([type=hidden]):not(:disabled), select:not(:disabled), textarea:not(:disabled), button:not(:disabled)");
  first?.focus();
}

const CUSTOM_LABELS = {
  provider_default: "Provider default", openai_effort: "OpenAI effort", limited_effort: "Limited effort (low / medium / high)",
  reasoning_object: "Reasoning object", thinking: "Thinking on/off", chat_template: "Chat-template thinking",
  native_responses: "Native reasoning", messages_manual: "Token-budget thinking", messages_adaptive: "Adaptive thinking",
};

async function openCustomProviderDialog(providerId = null) {
  if (!await refreshProviderConfig()) return;
  const provider = state.config.custom_providers.find((item) => item.provider_id === providerId) || {
    display_name: "", base_url: "", api_keys: { revision: null, entries: [] }, api_format: "openai_chat", reasoning_format: "provider_default", reasoning_history_format: "disabled", model_ids: [],
  };
  state.providerId = providerId || "__custom_new__";
  state.customProvider = provider;
  byId("providerDialogTitle").textContent = providerId ? provider.display_name : "Add provider";
  byId("providerMessage").textContent = "";
  byId("providerDialogCheck").hidden = true;
  renderProviderCheckMessage(byId("providerDialogAvailability"), null);
  const fields = byId("providerFields");
  fields.replaceChildren();
  const formats = (state.config.custom_reasoning_formats || {})[provider.api_format] || ["provider_default"];
  const definitions = [
    ["display_name", "Name", "string", [], ""],
    ["base_url", "Base URL", "string", [], "Include the API path, for example https://gateway.example/v1."],
    ["api_keys", "API keys", "api_key_list", [], "Key 1 is used. Leave the list empty for endpoints that do not require a key."],
    ["api_format", "API format", "select", [["openai_chat", "Chat Completions"], ["openai_responses", "Responses"], ["anthropic_messages", "Anthropic Messages"]], "Select the API exposed by your endpoint."],
    ["reasoning_format", "Reasoning format", "select", formats.map((value) => [value, CUSTOM_LABELS[value]]), "How FCC sends reasoning controls. Provider default leaves computation to the upstream."],
    ["reasoning_history_format", "Reasoning history format", "select", [["disabled", "Text context"], ["reasoning_content", "reasoning_content"], ["reasoning", "reasoning"], ["think_tags", "Think tags"]], "How previous reasoning is sent to a Chat Completions endpoint."],
    ["model_ids", "Model IDs", "textarea", [], "Optional, one upstream model ID per line. Leave empty to discover models automatically."],
  ];
  for (const [key, label, type, options, description] of definitions) {
    fields.appendChild(renderField({ key, label, type, options: options.map(([value, label]) => ({value, label})), description,
      value: key === "model_ids" ? provider.model_ids.join("\n") : provider[key], nullable: key === "api_keys", secret: key === "api_keys",
      configured: key === "api_keys" && !!provider.api_keys?.entries.length, locked: !!state.config.custom_providers_locked, source: state.config.custom_providers_locked ? "process" : "managed_env" }));
  }
  const format = byId("field-api_format");
  const reasoning = byId("field-reasoning_format");
  const history = byId("field-reasoning_history_format");
  history.closest(".field").hidden = format.value !== "openai_chat";
  format.addEventListener("change", () => {
    const allowed = state.config.custom_reasoning_formats[format.value];
    const previous = reasoning.value;
    reasoning.replaceChildren(...allowed.map((value) => option(value, CUSTOM_LABELS[value])));
    reasoning.value = allowed.includes(previous) ? previous : "provider_default";
    history.closest(".field").hidden = format.value !== "openai_chat";
    if (format.value !== "openai_chat") history.value = "disabled";
    updateDirtyState();
  });
  const actions = byId("providerDialogActions");
  actions.replaceChildren();
  if (providerId && !state.config.custom_providers_locked) {
    if (!provider.model_ids.length) actions.appendChild(authButton("Refresh models", (button) => testProvider(providerId, button), "secondary-button"));
    actions.appendChild(authButton("Remove provider", () => apply(providerId, "delete"), "danger-button"));
    const modelConfig = document.createElement("a");
    modelConfig.href = "/admin/model_config";
    modelConfig.textContent = "Model Config";
    modelConfig.addEventListener("click", (event) => { event.preventDefault(); byId("providerDialog").close(); navigateToView("model_config"); });
    actions.appendChild(modelConfig);
  }
  byId("saveProvider").hidden = false;
  updateDirtyState();
  if (!byId("providerDialog").open) byId("providerDialog").showModal();
  byId("field-display_name").focus();
}

function renderProviderDialogActions(provider) {
  if (state.customProvider) return;
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
  renderProviderCheckMessage(byId("providerDialogCheck"), check);
  renderProviderCheckMessage(byId("providerDialogAvailability"), check?.availability);
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
  let result;
  if (discovery === "starting") {
    result = { status: "checking", message: "Checking models…" };
  } else if (discovery === "failed") {
    result = { status: "error", message: state.startup.startup.provider_errors?.[providerId] || "Could not load models. Check the provider's settings and retry." };
  } else if (discovery === "ready") {
    result = { status: "ok", message: modelCountMessage(state.startup.cached_models[providerId]?.length || 0) };
  } else if (discovery === "unavailable") {
    return {
      status: "",
      message: "Local server unavailable",
      availability: check ? { ...check, status: "" } : undefined,
    };
  }
  // Model discovery and endpoint reachability are separate observations.
  return result ? { ...result, availability: check } : check;
}

function updateProviderCheckResult(providerId, status, message, source = "manual") {
  state.providerChecks.set(providerId, { status, message, source });
  renderProviderCheckResult(providerId);
}

function renderProviderCheckMessage(element, check) {
  for (const status of ["ok", "error", "checking"]) {
    element.classList.toggle(status, check?.status === status);
  }
  element.textContent = check?.message || "";
  element.hidden = !check?.message;
}

function renderProviderCheckResult(providerId) {
  const check = providerCheckResult(providerId);
  const { status = "" } = check || {};
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
  renderProviderCheckMessage(card.querySelector(".provider-check-result"), check);
  renderProviderCheckMessage(card.querySelector(".provider-availability-result"), check?.availability);
  if (state.providerId === providerId) {
    renderProviderCheckMessage(byId("providerDialogCheck"), check);
    renderProviderCheckMessage(byId("providerDialogAvailability"), check?.availability);
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
      if (section.id === "models") {
        renderModelRoutes(grid, sectionFields);
      } else {
        sectionFields.forEach((field) => grid.appendChild(renderField(field)));
      }
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

function renderModelRoutes(grid, fields) {
  const fieldsByKey = new Map(fields.map((field) => [field.key, field]));
  const routes = [
    ["Default", "MODEL", "REASONING_POLICY"],
    ["Fable", "MODEL_FABLE", "REASONING_FABLE"],
    ["Opus", "MODEL_OPUS", "REASONING_OPUS"],
    ["Sonnet", "MODEL_SONNET", "REASONING_SONNET"],
    ["Haiku", "MODEL_HAIKU", "REASONING_HAIKU"],
  ];
  for (const [name, modelKey, effortKey] of routes) {
    const row = document.createElement("div");
    row.className = "model-route-row";
    row.setAttribute("role", "group");
    row.setAttribute("aria-label", `${name} model and effort`);
    row.append(renderField(fieldsByKey.get(modelKey)), renderField(fieldsByKey.get(effortKey)));
    grid.appendChild(row);
    fieldsByKey.delete(modelKey);
    fieldsByKey.delete(effortKey);
  }
  fieldsByKey.forEach((field) => grid.appendChild(renderField(field)));
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

  if (field.type === "api_key_list") {
    label.id = `field-${field.key}-label`;
    label.removeAttribute("for");
    const editor = new window.FccApiKeyEditor(field, updateDirtyState);
    wrapper.append(label, editor.element);
    if (field.description) {
      const description = document.createElement("div");
      description.className = "field-description";
      description.textContent = field.description;
      wrapper.appendChild(description);
    }
    return wrapper;
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
    if (state.providerId) credentialEditors().forEach((editor) => editor.clearResults());
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
  if (field.type === "textarea") { const input = document.createElement("textarea"); input.rows = 3; input.value = field.value || ""; return input; }
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
    displayValue: (value) => state.modelLabels.get(value) || value,
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
    this.values = input.value ? JSON.parse(input.value) : [];
    this.inputId = `field-${field.key}-add`;
    this.rowComboboxes = [];
    this.rowKeys = [];
    this.draggedEntry = null;

    this.element = document.createElement("div");
    this.element.className = "model-list-editor";
    this.element.modelListEditor = this;

    const addRow = document.createElement("div");
    addRow.className = "model-list-add";
    const addCombobox = this.modelCombobox("", "add", "New fallback model");
    this.addInput = addCombobox.input;
    this.addEffort = this.effortSelect("inherit", "Fallback reasoning effort");
    this.addEffort.id = `field-${field.key}-effort`;
    this.addKey = this.keySelect("", 1, "Fallback API key");
    this.addProviderId = "";
    const updateAddModel = () => {
      const providerId = this.providerId(this.addInput.value);
      this.updateKeySelect(this.addKey, this.addInput.value, providerId === this.addProviderId ? Number(this.addKey.value) : 1);
      this.addProviderId = providerId;
    };
    this.addInput.addEventListener("input", updateAddModel);
    this.addInput.addEventListener("change", updateAddModel);

    this.addButton = this.actionButton("", "Add fallback", () => this.add());
    this.addButton.className = "secondary-button model-list-action";
    this.addButton.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M12 6v12M6 12h12" stroke-linecap="round"/></svg>';
    addRow.append(addCombobox.element, this.addEffort, this.addKey, this.addButton);

    this.rows = document.createElement("div");
    this.rows.className = "model-list-rows";
    this.rows.addEventListener("dragover", (event) => {
      if (!this.draggedEntry || this.field.locked) return;
      event.preventDefault();
      this.clearDropIndicators();
      const insertion = this.dropInsertion(event);
      if (this.dropDestination(insertion) === this.values.indexOf(this.draggedEntry)) {
        event.dataTransfer.dropEffect = "none";
        return;
      }
      event.dataTransfer.dropEffect = "move";
      const next = this.rows.children[insertion];
      const marker = next || this.rows.lastElementChild;
      marker.dataset.dropPosition = next ? "before" : "after";
    });
    this.rows.addEventListener("dragleave", (event) => {
      if (!this.rows.contains(event.relatedTarget)) this.clearDropIndicators();
    });
    this.rows.addEventListener("drop", (event) => {
      if (!this.draggedEntry || this.field.locked) return;
      event.preventDefault();
      const moved = this.draggedEntry;
      const destination = this.dropDestination(this.dropInsertion(event));
      this.clearDrag();
      this.move(moved, destination);
    });
    this.element.append(input, this.rows, addRow);
    this.renderRows();
  }

  add() {
    const value = this.addInput.value.trim();
    if (!value) {
      showMessage("Enter a full provider/model fallback.", "error");
      return;
    }
    const apiKeyNumber = Number(this.addKey.value);
    if (this.values.some((entry) => entry.model === value && (entry.api_key_number ?? 1) === apiKeyNumber)) {
      showMessage("That model and API key are already in the fallback list.", "error");
      return;
    }
    this.values.push({ model: value, reasoning: this.addEffort.value, api_key_number: apiKeyNumber });
    this.addInput.value = "";
    this.addEffort.value = "inherit";
    this.addProviderId = "";
    this.updateKeySelect(this.addKey, "", 1);
    showMessage("");
    this.sync();
    this.renderRows();
  }

  move(entry, destination) {
    const index = this.values.indexOf(entry);
    if (this.field.locked || destination < 0 || destination >= this.values.length || index === destination) return;
    this.values.splice(index, 1);
    this.values.splice(destination, 0, entry);
    this.sync();
    this.renderRows();
  }

  remove(index) {
    this.values.splice(index, 1);
    this.sync();
    this.renderRows();
  }

  sync() {
    this.input.value = this.values.length ? JSON.stringify(this.values) : "";
    this.input.dataset.remove = "false";
    this.input.dispatchEvent(new Event("input", { bubbles: true }));
  }

  renderRows() {
    this.rowComboboxes.forEach((combobox) => {
      combobox.close();
      state.modelComboboxes.delete(combobox);
    });
    this.rowComboboxes = [];
    this.rowKeys = [];
    this.rows.replaceChildren();

    this.values.forEach((entry, index) => {
      const row = document.createElement("div");
      row.className = "model-list-row";

      const combobox = this.modelCombobox(entry.model, `row-${index}`, `Fallback model ${index + 1}`);
      this.rowComboboxes.push(combobox);
      const effort = this.effortSelect(entry.reasoning, `Reasoning effort for ${entry.model}`);
      effort.addEventListener("change", () => {
        entry.reasoning = effort.value;
        this.sync();
      });
      const key = this.keySelect(entry.model, entry.api_key_number ?? 1, `API key for ${entry.model}`);
      this.rowKeys.push({ entry, select: key });
      key.addEventListener("change", () => {
        entry.api_key_number = Number(key.value);
        this.sync();
      });

      const handle = this.actionButton("⠿", `Move ${entry.model}`);
      handle.classList.add("model-list-drag");
      handle.draggable = !this.field.locked;
      handle.title = "Drag to reorder, or use the up and down arrow keys.";
      handle.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        event.preventDefault();
        this.move(entry, this.values.indexOf(entry) + (event.key === "ArrowUp" ? -1 : 1));
        this.rows.children[this.values.indexOf(entry)].querySelector(".model-list-drag").focus();
      });
      handle.addEventListener("dragstart", (event) => {
        this.draggedEntry = entry;
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", entry.model);
        row.classList.add("dragging");
      });
      handle.addEventListener("dragend", () => this.clearDrag());

      const remove = this.actionButton("", `Remove ${entry.model}`, () => this.remove(index));
      remove.className = "danger-button model-list-action";
      remove.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M3 6h18M9 6V4h6v2M5 6l1 14h12l1-14M10 10v6M14 10v6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      const updateModel = () => {
        const previousProvider = this.providerId(entry.model);
        entry.model = combobox.input.value.trim();
        if (this.providerId(entry.model) !== previousProvider) entry.api_key_number = 1;
        this.updateKeySelect(key, entry.model, entry.api_key_number ?? 1);
        key.setAttribute("aria-label", `API key for ${entry.model}`);
        effort.setAttribute("aria-label", `Reasoning effort for ${entry.model}`);
        handle.setAttribute("aria-label", `Move ${entry.model}`);
        remove.setAttribute("aria-label", `Remove ${entry.model}`);
        this.sync();
      };
      combobox.input.addEventListener("input", updateModel);
      combobox.input.addEventListener("change", updateModel);

      row.append(handle, combobox.element, effort, key, remove);
      this.rows.appendChild(row);
    });
  }

  modelCombobox(value, suffix, label) {
    const input = window.FccFormControls.configure(document.createElement("input"));
    input.id = `field-${this.field.key}-${suffix}`;
    input.type = "text";
    input.autocomplete = "off";
    input.placeholder = "provider/model";
    input.disabled = this.field.locked;
    input.value = value;
    input.setAttribute("aria-label", label);
    return createModelCombobox(input, { ...this.field, key: `${this.field.key}-${suffix}`, label, type: "model" });
  }

  dropInsertion(event) {
    const index = Array.from(this.rows.children).findIndex((row) => {
      const bounds = row.getBoundingClientRect();
      return event.clientY < bounds.top + bounds.height / 2;
    });
    return index === -1 ? this.values.length : index;
  }

  dropDestination(insertion) {
    const source = this.values.indexOf(this.draggedEntry);
    return insertion - (source < insertion ? 1 : 0);
  }

  clearDropIndicators() {
    this.rows.querySelectorAll("[data-drop-position]").forEach((row) => delete row.dataset.dropPosition);
  }

  clearDrag() {
    this.draggedEntry = null;
    this.clearDropIndicators();
    this.rows.querySelectorAll(".dragging").forEach((row) => row.classList.remove("dragging"));
  }

  effortSelect(value, label) {
    const select = window.FccFormControls.configure(document.createElement("select"));
    this.field.options.forEach((item) => select.appendChild(option(item.value, item.label)));
    select.value = value;
    select.disabled = this.field.locked;
    select.setAttribute("aria-label", label);
    return select;
  }

  providerId(model) {
    return model.includes("/") ? model.trim().split("/", 1)[0] : "";
  }

  keyCount(model) {
    const providerId = this.providerId(model);
    const custom = state.config.custom_providers.find((provider) => provider.provider_id === providerId);
    if (custom) return custom.api_keys.entries.length;
    const provider = state.config.provider_status.find((item) => item.provider_id === providerId);
    const credential = provider?.settings_keys.map((key) => state.fields.get(key)).find((field) => field?.type === "api_key_list");
    return credential ? credential.value.entries.length : null;
  }

  keySelect(model, value, label) {
    const select = window.FccFormControls.configure(document.createElement("select"));
    select.setAttribute("aria-label", label);
    this.updateKeySelect(select, model, value);
    return select;
  }

  updateKeySelect(select, model, value) {
    const count = this.keyCount(model);
    select.replaceChildren();
    if (count === null) {
      select.appendChild(option(1, this.providerId(model) ? "Default authentication" : "Key 1"));
    } else {
      for (let number = 1; number <= Math.max(1, count); number++) select.appendChild(option(number, `Key ${number}`));
    }
    if (value > Math.max(1, count ?? 1)) {
      const unavailable = option(value, `Key ${value} (unavailable)`);
      unavailable.disabled = true;
      select.appendChild(unavailable);
    }
    select.value = String(value);
    select.disabled = this.field.locked || count === null || (count === 0 && value === 1);
  }

  refreshKeyOptions() {
    this.updateKeySelect(this.addKey, this.addInput.value, Number(this.addKey.value));
    this.rowKeys.forEach(({ entry, select }) => this.updateKeySelect(select, entry.model, entry.api_key_number ?? 1));
  }

  actionButton(text, label, action = null) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button model-list-action";
    button.textContent = text;
    button.setAttribute("aria-label", label);
    button.title = label;
    button.disabled = this.field.locked;
    if (action) button.addEventListener("click", action);
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
  if (input.dataset.fieldType === "api_key_list") return JSON.parse(input.value);
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
  if (value !== null && typeof value === "object") return JSON.stringify(value);
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
  byId("saveProvider").disabled = state.applying || !!state.restart || state.configRefreshRequired || !Object.keys(changedValues(byId("providerFields"))).length;
}

function clearCredentialError(input) {
  byId(`${input.id}-error`)?.remove();
  input.removeAttribute("aria-invalid");
  input.removeAttribute("aria-describedby");
}

function credentialEditors() {
  return [...byId("providerFields").querySelectorAll(".api-key-editor")].map((element) => element.apiKeyEditor);
}

function showCredentialErrors(checks, rowErrors = []) {
  let first = null;
  [...rowErrors.map((error) => ({ ...error, status: "rejected" })), ...checks].forEach((check) => {
    const editor = credentialEditors().find((editor) => editor.field.key === check.key);
    if (editor) {
      if (check.status === "rejected" || check.status === "unverified") {
        const input = editor.showResult(check.index, check.message, check.status === "rejected");
        if (check.status === "rejected") first ||= input;
      }
      return;
    }
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
  byId("saveProvider").classList.toggle("startup-busy", applying);
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
  const { restart, warnings, providersOnly, providerId, checks } = state.restart;
  const target = new URL(restart.admin_url || "/admin", window.location.href);
  setApplying(true);
  showMessage(["Applied. Reconnecting to the server…", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
  try {
    await waitForRestart(restart, target);
    if (target.origin !== window.location.origin) {
      // Carry only the safe warning text across an address change, never edits.
      target.hash = new URLSearchParams({ "fcc-applied": JSON.stringify(warnings), ...(providerId ? { "fcc-provider": providerId, "fcc-checks": JSON.stringify(checks) } : {}) }).toString();
      window.location.replace(target.href);
      return;
    }
    await load({ providersOnly });
    state.restart = null;
    void refreshStartup();
    showMessage(["Applied", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
    if (providerId) await restoreSavedProvider(providerId, checks);
  } catch (error) {
    state.configRefreshRequired = true;
    showMessage([`Settings were saved. ${error.message} Use Reconnect to try again.`, ...warnings].join("\n"), "warn");
    appendAdminLink(target);
  } finally {
    setApplying(false);
  }
}

async function showRestartNotice() {
  const url = new URL(window.location.href);
  const fragment = new URLSearchParams(url.hash.slice(1));
  const notice = fragment.get("fcc-applied");
  if (notice === null) return;
  fragment.delete("fcc-applied");
  const providerId = fragment.get("fcc-provider");
  const checks = fragment.get("fcc-checks");
  fragment.delete("fcc-provider");
  fragment.delete("fcc-checks");
  url.hash = fragment.toString();
  window.history.replaceState(window.history.state, "", url);
  try {
    const warnings = JSON.parse(notice);
    if (Array.isArray(warnings) && warnings.every((warning) => typeof warning === "string")) {
      showMessage(["Applied", ...warnings].join("\n"), warnings.length ? "warn" : "ok");
      if (providerId && checks) {
        const parsed = JSON.parse(checks);
        if (Array.isArray(parsed) && parsed.every((check) => check && typeof check.key === "string" && Number.isInteger(check.index) && typeof check.message === "string" && check.status === "unverified")) await restoreSavedProvider(providerId, parsed);
      }
    }
  } catch {
    // A malformed navigation notice must not prevent normal Admin use.
  }
}

async function restoreSavedProvider(providerId, checks) {
  if (!connectedAccountDescriptor(providerId)) return;
  await openProviderDialog(providerId);
  showCredentialErrors(checks.map((check) => ({ ...check, message: `Key saved. ${check.message}` })));
  byId("providerMessage").textContent = "Saved. Some keys could not be verified.";
  byId("providerMessage").className = "message-area warn";
}

async function apply(providerId = null, customAction = null) {
  if (state.applying) return;
  if (state.restart) {
    await reconnectAfterRestart();
    return;
  }
  if (providerId && state.configRefreshRequired) return;
  const values = changedValues(providerId ? byId("providerFields") : byId("adminViews"));
  const custom = providerId && state.customProvider;
  const action = customAction || (custom?.provider_id ? "update" : "create");
  if (!Object.keys(values).length && !(custom && action === "delete")) return;
  if (providerId && action !== "delete") {
    let invalid = null;
    for (const editor of credentialEditors()) {
      const input = editor.validate();
      invalid ||= input;
      values[editor.field.key] = editor.payload();
    }
    if (invalid) {
      showMessage("Check the highlighted API key rows.", "error");
      invalid.focus();
      return;
    }
  }
  let mutation = null;
  if (custom) {
    if ("model_ids" in values) values.model_ids = values.model_ids.split(/\r?\n/).map((id) => id.trim()).filter(Boolean);
    mutation = { action, values: action === "delete" ? {} : values };
    if (custom.provider_id) mutation.provider_id = custom.provider_id;
  }
  const checkingKeys = credentialEditors().some((editor) => values[editor.field.key]?.entries.length);
  let rejectedField = null;
  let applied = false;
  setApplying(true);
  showMessage(checkingKeys ? "Checking API keys…" : "Applying…");
  try {
    const result = await api("/admin/api/config/apply", {
      method: "POST",
      body: JSON.stringify(mutation ? { custom_provider: mutation } : { values }),
    });
    const checks = result.credential_checks || [];
    if (!result.applied) {
      rejectedField = showCredentialErrors(checks, result.credential_row_errors || []);
      showMessage(rejectedField ? "Not applied. Check the highlighted API keys." : (result.errors || []).join("; "), "error");
      return;
    }
    applied = true;
    const warningChecks = checks.filter((check) => check.status === "unverified");
    const warnings = warningChecks.map((check) =>
      `${state.fields.get(check.key)?.label || "API keys"}, Key ${check.index + 1}: Key saved. ${check.message}`
    );
    const savedProviderId = warnings.length && providerId && mutation?.action !== "delete" ? result.custom_provider_id || providerId : null;
    if (providerId && !savedProviderId) byId("providerDialog").close();
    const restart = result.restart || {};
    if (restart.required && restart.automatic) {
      if (providerId) byId("providerDialog").close();
      state.restart = { restart, warnings, providersOnly: !!providerId, providerId: savedProviderId, checks: warningChecks };
      await reconnectAfterRestart();
      return;
    }
    const pending = restart.required ? restart.fields || [] : result.pending_fields || [];
    await load({ providersOnly: !!providerId });
    if (savedProviderId) await restoreSavedProvider(savedProviderId, warningChecks);
    const message = pending.length
      ? `Applied. Restart fcc-server to use: ${pending.join(", ")}`
      : "Applied";
    showMessage([message, ...warnings].join("\n"), warnings.length ? "warn" : "ok");
    if (savedProviderId) {
      byId("providerMessage").textContent = `Saved${pending.length ? ". Restart required" : ""}. Some keys could not be verified.`;
      byId("providerMessage").className = "message-area warn";
    }
  } catch (error) {
    if (providerId && (applied || error.status === 409 || !error.status || error.status >= 500)) state.configRefreshRequired = true;
    showMessage(applied ? `Applied, but could not reload settings: ${error.message}` : `Could not apply settings: ${error.message}`, "error");
    if (providerId && state.configRefreshRequired) byId("providerMessage").append(" Close and reopen this modal to load saved settings before saving again.");
  } finally {
    setApplying(false);
    if (rejectedField) {
      if (!providerId) navigateToView("providers");
      rejectedField.closest(".settings-section")?.classList.add("show-advanced");
      rejectedField.scrollIntoView({ block: "center", behavior: "instant" });
      rejectedField.focus();
    } else if (providerId && applied && !byId("providerDialog").open) {
      const focus = mutation && mutation.action !== "update"
        ? byId("addCustomProvider")
        : document.querySelector(`[data-provider="${providerId}"] [data-provider-settings]`);
      focus?.focus({ preventScroll: true });
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
    await Promise.all([...request.providerIds].map(async (providerId) => {
      try {
        const provider = await api(`/admin/api/providers/${providerId}/local-status`);
        if (state.localStatusRequest !== request || !request.providerIds.has(providerId)) return;
        if (provider.status === "missing_url") return;
        const diagnostics = [];
        if (provider.probe_path) diagnostics.push(`GET ${provider.probe_path}`);
        if (typeof provider.latency_ms === "number" && Number.isFinite(provider.latency_ms)) {
          diagnostics.push(`${provider.latency_ms} ms`);
        }
        const suffix = diagnostics.length ? ` (${diagnostics.join(", ")})` : "";
        if (provider.status === "reachable") {
          updateProviderCheckResult(providerId, "ok", `Reachable: ${provider.base_url}${suffix}`, "availability");
          return;
        }
        const detail = provider.message
          ? provider.message
          : provider.status_code
            ? `${provider.base_url} returned HTTP ${provider.status_code}`
            : "The local provider did not respond.";
        updateProviderCheckResult(providerId, "error", `Unavailable: ${detail}${suffix}`, "availability");
      } catch {
        if (state.localStatusRequest !== request || !request.providerIds.has(providerId)) return;
        updateProviderCheckResult(providerId, "error", "Availability check failed. Use Test to retry.", "availability");
      }
    }));
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
      void hydrateModelOptions();
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
  if (request === state.modelOptionsRequest && config === state.config) { state.modelLabels = new Map(Object.entries(result.model_labels || {})); setModelOptions(result.models); }
  if (refresh && window.CodeSessions) void window.CodeSessions.refresh();
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
  const wasCustom = !!state.customProvider;
  state.customProvider = null;
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
const claudeIntegration = { connected: null, busy: false, paths: null, update: null };
const claudeIntegrationPath = "/admin/api/integrations/claude-vscode";
const claudeOnboardingDialog = byId("claudeOnboardingDialog");
const claudeOnboarding = { completed: null, busy: false, paths: null, refreshAfterSave: false };
const claudeOnboardingPath = "/admin/api/integrations/claude-onboarding";

function renderClaudeOnboarding() {
  const { completed, busy, paths, checkRequest } = claudeOnboarding;
  const button = byId("openClaudeOnboarding");
  button.textContent = busy ? checkRequest ? "Loading…" : "Saving…" : completed === null ? "Retry" : completed ? "Completed" : "Complete onboarding";
  button.disabled = busy || completed === true;
  button.setAttribute("aria-busy", String(busy));
  const action = byId("confirmClaudeOnboarding");
  action.textContent = busy ? "Saving…" : "Complete onboarding";
  action.disabled = busy || completed !== false;
  const files = byId("claudeOnboardingFiles");
  files.replaceChildren();
  if (paths) {
    const item = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = paths.claude_state;
    item.appendChild(code);
    files.appendChild(item);
  }
}

async function refreshClaudeOnboarding({ replaceCheck = false } = {}) {
  if (claudeOnboarding.busy && !claudeOnboarding.checkRequest) {
    claudeOnboarding.refreshAfterSave = true;
    return;
  }
  if (replaceCheck && claudeOnboarding.checkRequest) {
    claudeOnboarding.checkRequest = null;
    claudeOnboarding.busy = false;
  }
  const request = beginIntegrationCheck(claudeOnboarding);
  if (!request) return;
  renderClaudeOnboarding();
  try {
    const result = await api(claudeOnboardingPath);
    if (!currentIntegrationCheck(claudeOnboarding, request)) return;
    claudeOnboarding.completed = result.completed;
    claudeOnboarding.paths = result.paths;
    integrationMessage("claudeOnboardingMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(claudeOnboarding, request)) return;
    claudeOnboarding.completed = null;
    integrationMessage("claudeOnboardingMessage", error.message, true);
  } finally {
    finishIntegrationCheck(claudeOnboarding, request, renderClaudeOnboarding);
  }
}

byId("openClaudeOnboarding").addEventListener("click", () => {
  if (claudeOnboarding.completed === null) {
    void refreshClaudeOnboarding();
    return;
  }
  integrationMessage("claudeOnboardingDialogMessage", "");
  claudeOnboardingDialog.showModal();
});
byId("confirmClaudeOnboarding").addEventListener("click", async () => {
  if (byId("confirmClaudeOnboarding").disabled) return;
  claudeOnboarding.busy = true;
  renderClaudeOnboarding();
  integrationMessage("claudeOnboardingDialogMessage", "");
  integrationMessage("claudeOnboardingMessage", "");
  try {
    const result = await api(`${claudeOnboardingPath}/complete`, { method: "POST" });
    claudeOnboarding.completed = result.completed;
    claudeOnboardingDialog.close();
    integrationMessage("claudeOnboardingMessage", "Claude onboarding completed.");
  } catch (error) {
    integrationMessage("claudeOnboardingDialogMessage", error.message, true);
    integrationMessage("claudeOnboardingMessage", error.message, true);
  } finally {
    claudeOnboarding.busy = false;
    renderClaudeOnboarding();
    if (claudeOnboarding.refreshAfterSave) {
      claudeOnboarding.refreshAfterSave = false;
      void refreshClaudeOnboarding();
    }
  }
});
byId("closeClaudeOnboarding").addEventListener("click", () => claudeOnboardingDialog.close());
claudeOnboardingDialog.addEventListener("click", (event) => {
  if (event.target !== claudeOnboardingDialog) return;
  const bounds = claudeOnboardingDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    claudeOnboardingDialog.close();
  }
});

function integrationMessage(id, message, error = false) {
  const element = byId(id);
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("error", error);
}

function integrationUpdating(integration, id) {
  return integration.update?.state === "starting" ||
    state.startup?.startup?.integrations?.[id]?.state === "starting";
}

function beginIntegrationCheck(integration) {
  if (integration.busy && (!integration.checkRequest || integration.checkRequest.config === state.config)) return null;
  const request = { config: state.config };
  integration.checkRequest = request;
  integration.busy = true;
  return request;
}

function currentIntegrationCheck(integration, request) {
  return integration.checkRequest === request && request.config === state.config;
}

function finishIntegrationCheck(integration, request, render) {
  if (integration.checkRequest !== request) return;
  integration.checkRequest = null;
  integration.busy = false;
  render();
  if (request.config === state.config && integration.update?.state === "starting") void refreshStartup();
}

function refreshIntegrationUpdates(previous, current) {
  if (state.activeView !== "integrations") return;
  for (const [id, integration, refresh, messageId, message] of [
    ["vscode-chat", vscodeChatIntegration, refreshVSCodeChatIntegration, "vscodeChatIntegrationMessage", "Models updated in VS Code."],
    ["claude-vscode", claudeIntegration, refreshClaudeIntegration, "claudeIntegrationMessage", "Settings updated. Reload VS Code."],
    ["jetbrains-acp", jetBrainsIntegration, refreshJetBrainsIntegration, "jetBrainsIntegrationMessage", "Settings updated. Reopen JetBrains and start a new chat."],
    ["codex", codexIntegration, refreshCodexIntegration, "codexIntegrationMessage", "Settings updated. Restart Codex."],
    ["claude-desktop", claudeDesktopIntegration, refreshClaudeDesktopIntegration, "claudeDesktopIntegrationMessage", "Settings updated. Reopen Claude Desktop."],
    ["dsh-desktop", dshDesktopIntegration, refreshDshDesktopIntegration, "dshDesktopIntegrationMessage", "DSH Desktop configuration updated."],
  ]) {
    const phase = current?.[id]?.state;
    // Successful actions clear update metadata, not the last observed state.
    const observedPhase = integration.update?.state ?? previous?.[id]?.state;
    // Polling can miss a short update. Reconcile state independently of notices.
    if (phase && phase !== "starting" && (
      previous?.[id]?.state === "starting" || observedPhase !== phase
    )) {
      if (integration.busy) state.startupAgain = true;
      else void refresh(false, { background: true });
    }
    if (previous?.[id]?.state === "starting" && phase === "ready" && current[id].changed === true) {
      integrationMessage(messageId, message);
    }
  }
}

function renderClaudeIntegration() {
  const { connected, paths } = claudeIntegration;
  const busy = claudeIntegration.busy || integrationUpdating(claudeIntegration, "claude-vscode");
  const action = connected ? "Disconnect" : "Connect";
  byId("openClaudeIntegration").textContent = busy ? "Loading…" : connected === null ? "Retry" : action;
  byId("openClaudeIntegration").disabled = busy;
  byId("openClaudeIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmClaudeIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmClaudeIntegration").disabled = busy || connected === null;
  byId("openClaudeIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmClaudeIntegration").className = connected ? "danger-button" : "primary-button";
  byId("claudeIntegrationDescription").textContent = connected
    ? "Remove FCC's VS Code settings."
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

async function refreshClaudeIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("claudeIntegrationMessage", "");
  const request = beginIntegrationCheck(claudeIntegration);
  if (!request) return;
  renderClaudeIntegration();
  try {
    if (retry) await api(`${claudeIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(claudeIntegrationPath);
    if (!currentIntegrationCheck(claudeIntegration, request)) return;
    claudeIntegration.connected = result.connected;
    claudeIntegration.paths = result.paths;
    claudeIntegration.update = result.update;
    if (result.update?.state === "failed") integrationMessage("claudeIntegrationMessage", result.update.message, true);
    else if (byId("claudeIntegrationMessage").classList.contains("error")) integrationMessage("claudeIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(claudeIntegration, request)) return;
    claudeIntegration.connected = null;
    claudeIntegration.update = null;
    integrationMessage("claudeIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(claudeIntegration, request, renderClaudeIntegration);
  }
}

byId("openClaudeIntegration").addEventListener("click", () => {
  if (claudeIntegration.connected === null) {
    refreshClaudeIntegration(claudeIntegration.update?.state === "failed");
    return;
  }
  integrationMessage("claudeIntegrationDialogMessage", "");
  claudeIntegrationDialog.showModal();
});
byId("confirmClaudeIntegration").addEventListener("click", async () => {
  if (byId("confirmClaudeIntegration").disabled) return;
  const disconnect = claudeIntegration.connected;
  claudeIntegration.busy = true;
  renderClaudeIntegration();
  integrationMessage("claudeIntegrationDialogMessage", "");
  integrationMessage("claudeIntegrationMessage", "");
  try {
    const result = await api(`${claudeIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    claudeIntegration.connected = result.connected;
    claudeIntegration.update = null;
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
    if (!disconnect) void refreshClaudeOnboarding({ replaceCheck: true });
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

const vscodeChatIntegrationDialog = byId("vscodeChatIntegrationDialog");
const vscodeChatIntegration = { connected: null, busy: false, paths: null, update: null };
const vscodeChatIntegrationPath = "/admin/api/integrations/vscode-chat";

function renderVSCodeChatIntegration() {
  const { connected, paths } = vscodeChatIntegration;
  const busy = vscodeChatIntegration.busy || integrationUpdating(vscodeChatIntegration, "vscode-chat");
  byId("retryVSCodeChatIntegration").hidden = connected === null || vscodeChatIntegration.update?.state !== "failed";
  byId("retryVSCodeChatIntegration").disabled = busy;
  const action = connected ? "Disconnect" : "Connect";
  byId("openVSCodeChatIntegration").textContent = busy ? "Loading…" : connected === null ? "Retry" : action;
  byId("openVSCodeChatIntegration").disabled = busy;
  byId("openVSCodeChatIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmVSCodeChatIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmVSCodeChatIntegration").disabled = busy || connected === null;
  byId("openVSCodeChatIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmVSCodeChatIntegration").className = connected ? "danger-button" : "primary-button";
  byId("vscodeChatIntegrationDescription").textContent = connected
    ? "Remove FCC's model group from VS Code Chat."
    : "Add FCC models to VS Code Chat. Your selected models stay unchanged.";
  const files = byId("vscodeChatIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = [paths.vscode_models];
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshVSCodeChatIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("vscodeChatIntegrationMessage", "");
  const request = beginIntegrationCheck(vscodeChatIntegration);
  if (!request) return;
  renderVSCodeChatIntegration();
  try {
    if (retry) await api(`${vscodeChatIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(vscodeChatIntegrationPath);
    if (!currentIntegrationCheck(vscodeChatIntegration, request)) return;
    vscodeChatIntegration.connected = result.connected;
    vscodeChatIntegration.paths = result.paths;
    vscodeChatIntegration.update = result.update;
    if (result.update?.state === "failed") integrationMessage("vscodeChatIntegrationMessage", result.update.message, true);
    else if (byId("vscodeChatIntegrationMessage").classList.contains("error")) integrationMessage("vscodeChatIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(vscodeChatIntegration, request)) return;
    vscodeChatIntegration.connected = null;
    vscodeChatIntegration.update = null;
    integrationMessage("vscodeChatIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(vscodeChatIntegration, request, renderVSCodeChatIntegration);
  }
}

byId("openVSCodeChatIntegration").addEventListener("click", () => {
  if (vscodeChatIntegration.connected === null) {
    refreshVSCodeChatIntegration(true);
    return;
  }
  integrationMessage("vscodeChatIntegrationDialogMessage", "");
  vscodeChatIntegrationDialog.showModal();
});
byId("retryVSCodeChatIntegration").addEventListener("click", () => refreshVSCodeChatIntegration(true));
byId("confirmVSCodeChatIntegration").addEventListener("click", async () => {
  if (byId("confirmVSCodeChatIntegration").disabled) return;
  const disconnect = vscodeChatIntegration.connected;
  vscodeChatIntegration.busy = true;
  renderVSCodeChatIntegration();
  integrationMessage("vscodeChatIntegrationDialogMessage", "");
  integrationMessage("vscodeChatIntegrationMessage", "");
  try {
    const result = await api(`${vscodeChatIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    vscodeChatIntegration.connected = result.connected;
    vscodeChatIntegration.update = null;
    vscodeChatIntegrationDialog.close();
    integrationMessage("vscodeChatIntegrationMessage", disconnect
      ? "Settings removed. Reload VS Code to disconnect."
      : "Models saved. Select an FCC model in VS Code Chat. Reload VS Code if needed.");
  } catch (error) {
    integrationMessage("vscodeChatIntegrationDialogMessage", error.message, true);
    integrationMessage("vscodeChatIntegrationMessage", error.message, true);
  } finally {
    vscodeChatIntegration.busy = false;
    renderVSCodeChatIntegration();
  }
});
byId("closeVSCodeChatIntegration").addEventListener("click", () => vscodeChatIntegrationDialog.close());
vscodeChatIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== vscodeChatIntegrationDialog) return;
  const bounds = vscodeChatIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    vscodeChatIntegrationDialog.close();
  }
});

const codexIntegrationDialog = byId("codexIntegrationDialog");
const codexIntegration = { connected: null, busy: false, paths: null, update: null };
const codexIntegrationPath = "/admin/api/integrations/codex";

function renderCodexIntegration() {
  const { connected, paths } = codexIntegration;
  const initializing = connected === false && state.startup?.startup?.catalog_file === "starting";
  const unavailable = connected === false && state.startup?.startup?.catalog_file === "failed";
  const busy = codexIntegration.busy || initializing || integrationUpdating(codexIntegration, "codex");
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

async function refreshCodexIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("codexIntegrationMessage", "");
  const request = beginIntegrationCheck(codexIntegration);
  if (!request) return;
  renderCodexIntegration();
  try {
    if (retry) await api(`${codexIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(codexIntegrationPath);
    if (!currentIntegrationCheck(codexIntegration, request)) return;
    codexIntegration.connected = result.connected;
    codexIntegration.paths = result.paths;
    codexIntegration.update = result.update;
    if (result.update?.state === "failed") integrationMessage("codexIntegrationMessage", result.update.message, true);
    else if (byId("codexIntegrationMessage").classList.contains("error")) integrationMessage("codexIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(codexIntegration, request)) return;
    codexIntegration.connected = null;
    codexIntegration.update = null;
    integrationMessage("codexIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(codexIntegration, request, renderCodexIntegration);
  }
}

byId("openCodexIntegration").addEventListener("click", () => {
  if (codexIntegration.connected === null) {
    refreshCodexIntegration(codexIntegration.update?.state === "failed");
    return;
  }
  integrationMessage("codexIntegrationDialogMessage", "");
  codexIntegrationDialog.showModal();
});
byId("confirmCodexIntegration").addEventListener("click", async () => {
  if (byId("confirmCodexIntegration").disabled) return;
  const disconnect = codexIntegration.connected;
  codexIntegration.busy = true;
  renderCodexIntegration();
  integrationMessage("codexIntegrationDialogMessage", "");
  integrationMessage("codexIntegrationMessage", "");
  try {
    const result = await api(`${codexIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    codexIntegration.connected = result.connected;
    codexIntegration.paths = result.paths;
    codexIntegration.update = null;
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
const jetBrainsIntegration = { connected: null, busy: false, paths: null, update: null };
const jetBrainsIntegrationPath = "/admin/api/integrations/jetbrains-acp";

function renderJetBrainsIntegration() {
  const { connected, paths } = jetBrainsIntegration;
  const busy = jetBrainsIntegration.busy || integrationUpdating(jetBrainsIntegration, "jetbrains-acp");
  const action = connected ? "Disconnect" : "Connect";
  byId("openJetBrainsIntegration").textContent = busy ? "Loading…" : connected === null ? "Retry" : action;
  byId("openJetBrainsIntegration").disabled = busy;
  byId("openJetBrainsIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmJetBrainsIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmJetBrainsIntegration").disabled = busy || connected === null;
  byId("openJetBrainsIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmJetBrainsIntegration").className = connected ? "danger-button" : "primary-button";
  byId("jetBrainsIntegrationDescription").textContent = connected
    ? "Remove the Claude Code (FCC) agent, including its custom settings. Finish any configuration edits first."
    : "Add Claude Code (FCC) using JetBrains' installed Claude Agent. Start that agent once before connecting, and finish any configuration edits.";
  const files = byId("jetBrainsIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = [paths.acp_config];
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshJetBrainsIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("jetBrainsIntegrationMessage", "");
  const request = beginIntegrationCheck(jetBrainsIntegration);
  if (!request) return;
  renderJetBrainsIntegration();
  try {
    if (retry) await api(`${jetBrainsIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(jetBrainsIntegrationPath);
    if (!currentIntegrationCheck(jetBrainsIntegration, request)) return;
    jetBrainsIntegration.connected = result.connected;
    jetBrainsIntegration.paths = result.paths;
    jetBrainsIntegration.update = result.update;
    if (result.update?.state === "failed") integrationMessage("jetBrainsIntegrationMessage", result.update.message, true);
    else if (byId("jetBrainsIntegrationMessage").classList.contains("error")) integrationMessage("jetBrainsIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(jetBrainsIntegration, request)) return;
    jetBrainsIntegration.connected = null;
    jetBrainsIntegration.update = null;
    integrationMessage("jetBrainsIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(jetBrainsIntegration, request, renderJetBrainsIntegration);
  }
}

byId("openJetBrainsIntegration").addEventListener("click", () => {
  if (jetBrainsIntegration.connected === null) {
    refreshJetBrainsIntegration(jetBrainsIntegration.update?.state === "failed");
    return;
  }
  integrationMessage("jetBrainsIntegrationDialogMessage", "");
  jetBrainsIntegrationDialog.showModal();
});
byId("confirmJetBrainsIntegration").addEventListener("click", async () => {
  if (byId("confirmJetBrainsIntegration").disabled) return;
  const disconnect = jetBrainsIntegration.connected;
  jetBrainsIntegration.busy = true;
  renderJetBrainsIntegration();
  integrationMessage("jetBrainsIntegrationDialogMessage", "");
  integrationMessage("jetBrainsIntegrationMessage", "");
  try {
    const result = await api(`${jetBrainsIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    jetBrainsIntegration.connected = result.connected;
    jetBrainsIntegration.update = null;
    jetBrainsIntegrationDialog.close();
    integrationMessage("jetBrainsIntegrationMessage", disconnect
      ? "Agent removed. Reopen JetBrains to disconnect."
      : "Settings saved. Reopen JetBrains, select Claude Code (FCC), and start a new chat.");
  } catch (error) {
    integrationMessage("jetBrainsIntegrationDialogMessage", error.message, true);
    integrationMessage("jetBrainsIntegrationMessage", error.message, true);
  } finally {
    jetBrainsIntegration.busy = false;
    renderJetBrainsIntegration();
  }
});
byId("closeJetBrainsIntegration").addEventListener("click", () => jetBrainsIntegrationDialog.close());
jetBrainsIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== jetBrainsIntegrationDialog) return;
  const bounds = jetBrainsIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    jetBrainsIntegrationDialog.close();
  }
});

const claudeDesktopIntegrationDialog = byId("claudeDesktopIntegrationDialog");
const claudeDesktopIntegration = { connected: null, disconnectPending: false, busy: false, paths: null, update: null };
const claudeDesktopIntegrationPath = "/admin/api/integrations/claude-desktop";

function renderClaudeDesktopIntegration() {
  const { connected, paths, disconnectPending } = claudeDesktopIntegration;
  const busy = claudeDesktopIntegration.busy || integrationUpdating(claudeDesktopIntegration, "claude-desktop");
  const disconnect = disconnectPending || connected;
  const action = disconnectPending ? "Retry disconnect" : connected ? "Disconnect" : "Connect";
  byId("openClaudeDesktopIntegration").textContent = busy ? "Loading…" : connected === null && !disconnectPending ? "Retry" : action;
  byId("openClaudeDesktopIntegration").disabled = busy;
  byId("openClaudeDesktopIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmClaudeDesktopIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmClaudeDesktopIntegration").disabled = busy || (connected === null && !disconnectPending);
  byId("openClaudeDesktopIntegration").className = disconnect && !busy ? "danger-button" : "primary-button";
  byId("confirmClaudeDesktopIntegration").className = disconnect ? "danger-button" : "primary-button";
  byId("claudeDesktopIntegrationDescription").textContent = disconnectPending
    ? "Finish removing FCC's configuration. Fully quit Claude Desktop before retrying, then reopen it."
    : connected
    ? "Remove FCC's configuration and return to normal Claude sign-in. Fully quit Claude Desktop first, then reopen it."
    : "Set FCC as Claude Desktop's gateway. Fully quit Claude Desktop before connecting, then reopen it.";
  const files = byId("claudeDesktopIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = Object.values(paths);
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshClaudeDesktopIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("claudeDesktopIntegrationMessage", "");
  const request = beginIntegrationCheck(claudeDesktopIntegration);
  if (!request) return;
  renderClaudeDesktopIntegration();
  try {
    if (retry) await api(`${claudeDesktopIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(claudeDesktopIntegrationPath);
    if (!currentIntegrationCheck(claudeDesktopIntegration, request)) return;
    claudeDesktopIntegration.connected = result.connected;
    claudeDesktopIntegration.disconnectPending = result.disconnect_pending;
    claudeDesktopIntegration.paths = result.paths;
    claudeDesktopIntegration.update = result.update;
    if (result.update?.state === "failed") integrationMessage("claudeDesktopIntegrationMessage", result.update.message, true);
    else if (byId("claudeDesktopIntegrationMessage").classList.contains("error")) integrationMessage("claudeDesktopIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(claudeDesktopIntegration, request)) return;
    claudeDesktopIntegration.connected = null;
    claudeDesktopIntegration.disconnectPending = false;
    claudeDesktopIntegration.update = null;
    integrationMessage("claudeDesktopIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(claudeDesktopIntegration, request, renderClaudeDesktopIntegration);
  }
}

byId("openClaudeDesktopIntegration").addEventListener("click", () => {
  if (claudeDesktopIntegration.connected === null && !claudeDesktopIntegration.disconnectPending) {
    refreshClaudeDesktopIntegration(claudeDesktopIntegration.update?.state === "failed");
    return;
  }
  integrationMessage("claudeDesktopIntegrationDialogMessage", "");
  claudeDesktopIntegrationDialog.showModal();
});
byId("confirmClaudeDesktopIntegration").addEventListener("click", async () => {
  if (byId("confirmClaudeDesktopIntegration").disabled) return;
  const disconnect = claudeDesktopIntegration.disconnectPending || claudeDesktopIntegration.connected;
  claudeDesktopIntegration.busy = true;
  renderClaudeDesktopIntegration();
  integrationMessage("claudeDesktopIntegrationDialogMessage", "");
  integrationMessage("claudeDesktopIntegrationMessage", "");
  try {
    const result = await api(`${claudeDesktopIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    claudeDesktopIntegration.connected = result.connected;
    claudeDesktopIntegration.disconnectPending = result.disconnect_pending;
    claudeDesktopIntegration.paths = result.paths;
    claudeDesktopIntegration.update = null;
    claudeDesktopIntegrationDialog.close();
    integrationMessage("claudeDesktopIntegrationMessage", disconnect
      ? "Settings removed. Reopen Claude Desktop to disconnect."
      : "Settings saved. Reopen Claude Desktop to connect.");
  } catch (error) {
    if (disconnect) {
      try {
        const result = await api(claudeDesktopIntegrationPath);
        claudeDesktopIntegration.connected = result.connected;
        claudeDesktopIntegration.disconnectPending = result.disconnect_pending;
        claudeDesktopIntegration.paths = result.paths;
        claudeDesktopIntegration.update = result.update;
      } catch {
        // Keep the user's disconnect action available if status is unreadable too.
      }
    }
    integrationMessage("claudeDesktopIntegrationDialogMessage", error.message, true);
    integrationMessage("claudeDesktopIntegrationMessage", error.message, true);
  } finally {
    claudeDesktopIntegration.busy = false;
    renderClaudeDesktopIntegration();
  }
});
byId("closeClaudeDesktopIntegration").addEventListener("click", () => claudeDesktopIntegrationDialog.close());
claudeDesktopIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== claudeDesktopIntegrationDialog) return;
  const bounds = claudeDesktopIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    claudeDesktopIntegrationDialog.close();
  }
});

const dshDesktopIntegrationDialog = byId("dshDesktopIntegrationDialog");
const dshDesktopIntegration = { connected: null, actionRevision: 0, busy: false, paths: null, update: null };
const dshDesktopIntegrationPath = "/admin/api/integrations/dsh-desktop";

function applyDshDesktopStatus(result) {
  dshDesktopIntegration.connected = result.connected;
  dshDesktopIntegration.paths = result.paths ?? dshDesktopIntegration.paths;
  dshDesktopIntegration.update = result.update ?? null;
}

function renderDshDesktopIntegration() {
  const { connected, paths } = dshDesktopIntegration;
  const updating = integrationUpdating(dshDesktopIntegration, "dsh-desktop");
  const busy = dshDesktopIntegration.busy || (updating && !connected);
  byId("retryDshDesktopIntegration").hidden = dshDesktopIntegration.update?.state !== "failed";
  byId("retryDshDesktopIntegration").disabled = busy || updating;
  const action = connected === null ? "Retry" : connected ? "Disconnect" : "Connect";
  byId("openDshDesktopIntegration").textContent = busy ? "Loading…" : action;
  byId("openDshDesktopIntegration").disabled = busy;
  byId("openDshDesktopIntegration").setAttribute("aria-busy", String(busy));
  byId("confirmDshDesktopIntegration").textContent = busy ? "Saving…" : action;
  byId("confirmDshDesktopIntegration").disabled = busy || connected === null;
  byId("openDshDesktopIntegration").className = connected && !busy ? "danger-button" : "primary-button";
  byId("confirmDshDesktopIntegration").className = connected ? "danger-button" : "primary-button";
  byId("dshDesktopIntegrationDescription").textContent = connected
    ? "Remove FCC's Desktop connection. Existing FCC sessions need another model selected to continue."
    : "Install and open DSH Desktop once, then connect FCC as the default for new sessions.";
  const files = byId("dshDesktopIntegrationFiles");
  files.replaceChildren();
  if (paths) {
    const targets = Object.values(paths);
    targets.forEach((path) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      code.textContent = path;
      item.appendChild(code);
      files.appendChild(item);
    });
  }
}

async function refreshDshDesktopIntegration(retry = false, { background = false } = {}) {
  if (!background) integrationMessage("dshDesktopIntegrationMessage", "");
  const request = beginIntegrationCheck(dshDesktopIntegration);
  if (!request) return;
  renderDshDesktopIntegration();
  try {
    if (retry) await api(`${dshDesktopIntegrationPath}/refresh`, { method: "POST" });
    const result = await api(dshDesktopIntegrationPath);
    if (!currentIntegrationCheck(dshDesktopIntegration, request)) return;
    applyDshDesktopStatus(result);
    if (result.update?.state === "failed") integrationMessage("dshDesktopIntegrationMessage", result.update.message, true);
    else if (byId("dshDesktopIntegrationMessage").classList.contains("error")) integrationMessage("dshDesktopIntegrationMessage", "");
  } catch (error) {
    if (!currentIntegrationCheck(dshDesktopIntegration, request)) return;
    integrationMessage("dshDesktopIntegrationMessage", error.message, true);
  } finally {
    finishIntegrationCheck(dshDesktopIntegration, request, renderDshDesktopIntegration);
  }
}

byId("openDshDesktopIntegration").addEventListener("click", () => {
  if (dshDesktopIntegration.connected === null) {
    refreshDshDesktopIntegration(dshDesktopIntegration.update?.state === "failed");
    return;
  }
  integrationMessage("dshDesktopIntegrationDialogMessage", "");
  dshDesktopIntegrationDialog.showModal();
});
byId("confirmDshDesktopIntegration").addEventListener("click", async () => {
  if (byId("confirmDshDesktopIntegration").disabled) return;
  const disconnect = dshDesktopIntegration.connected === true;
  const config = state.config;
  const actionRevision = ++dshDesktopIntegration.actionRevision;
  const current = () => config === state.config && actionRevision === dshDesktopIntegration.actionRevision;
  dshDesktopIntegration.checkRequest = null;
  dshDesktopIntegration.busy = true;
  renderDshDesktopIntegration();
  integrationMessage("dshDesktopIntegrationDialogMessage", "");
  integrationMessage("dshDesktopIntegrationMessage", "");
  try {
    const result = await api(`${dshDesktopIntegrationPath}/${disconnect ? "disconnect" : "connect"}`, { method: "POST" });
    if (!current()) return;
    applyDshDesktopStatus(result);
    dshDesktopIntegrationDialog.close();
    integrationMessage("dshDesktopIntegrationMessage", result.connected
      ? "Connected. New DSH sessions use FCC. Reopen DSH if it does not reload the changes."
      : "DSH Desktop is disconnected. Select another model in existing DSH sessions.");
  } catch (error) {
    if (!current()) return;
    try {
      const result = await api(dshDesktopIntegrationPath);
      if (!current()) return;
      applyDshDesktopStatus(result);
    } catch {
      // Keep the last known status if the follow-up read also fails.
    }
    if (!current()) return;
    integrationMessage("dshDesktopIntegrationDialogMessage", error.message, true);
    integrationMessage("dshDesktopIntegrationMessage", error.message, true);
  } finally {
    if (actionRevision !== dshDesktopIntegration.actionRevision) return;
    dshDesktopIntegration.busy = false;
    renderDshDesktopIntegration();
    if (config !== state.config) void refreshDshDesktopIntegration();
  }
});
byId("retryDshDesktopIntegration").addEventListener("click", () => refreshDshDesktopIntegration(true));
byId("closeDshDesktopIntegration").addEventListener("click", () => dshDesktopIntegrationDialog.close());
dshDesktopIntegrationDialog.addEventListener("click", (event) => {
  if (event.target !== dshDesktopIntegrationDialog) return;
  const bounds = dshDesktopIntegrationDialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) {
    dshDesktopIntegrationDialog.close();
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

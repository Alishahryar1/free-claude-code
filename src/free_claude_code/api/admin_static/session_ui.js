(() => {
  "use strict";
  function node(tag, className = "", text = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return element;
  }
  function button(label, className, action) {
    const element = node("button", className, label);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  }
  function libraryHeader(title, description, create) {
    const header = node("header", "session-library-header"),
      copy = node("div");
    copy.append(node("h2", "", title), node("p", "", description));
    header.append(copy, create);
    return header;
  }
  function search(label, value, onChange) {
    const input = node("input", "session-search");
    input.type = "search";
    input.placeholder = label;
    input.value = value;
    input.setAttribute("aria-label", label);
    let timer;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (input.isConnected) onChange(input.value);
      }, 200);
    });
    return input;
  }
  function card(title, preview, meta, action) {
    const element = button("", "session-card", action);
    element.append(
      node("strong", "", title),
      node("p", "", preview),
      node("span", "", meta),
    );
    return element;
  }
  function shell(header, notice, transcript, composer) {
    const element = node("div", "session-shell");
    element.append(header, notice, transcript, composer);
    return element;
  }
  function header(
    backLabel,
    back,
    titleValue,
    titleLabel,
    onTitle,
    actions,
    controls,
  ) {
    const element = node("header", "session-header"),
      row = node("div", "session-header-row");
    const title = node("input", "session-title");
    title.value = titleValue;
    title.maxLength = 200;
    title.setAttribute("aria-label", titleLabel);
    title.addEventListener("blur", () => {
      if (title.isConnected) onTitle(title);
    });
    title.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.isComposing) title.blur();
    });
    row.append(
      button(backLabel, "session-back-button", back),
      title,
      ...actions,
    );
    element.append(row, controls);
    return element;
  }
  function modelControl(id, value, values, registry, onSelect) {
    const group = node("div", "session-control session-model-control"),
      label = node("label", "", "Model");
    label.htmlFor = id;
    const input = node("input", "session-model-input");
    input.id = id;
    input.type = "text";
    input.autocomplete = "off";
    input.value = value;
    input.setAttribute("aria-label", "Selected model");
    let committed = value;
    const combobox = new window.FccModelCombobox(input, {
      listboxId: `${id}-options`,
      label: "model",
      values,
      registry,
      emptyMessage: () =>
        values().length ? "No matching models." : "No models available.",
      onSelect: (model) => {
        committed = model;
        onSelect(model);
      },
      onClose: () => {
        input.value = committed;
      },
    });
    group.append(label, combobox.element);
    return {
      group,
      input,
      combobox,
      update(model, disabled) {
        committed = model;
        if (!combobox.isOpen) input.value = model;
        input.disabled = disabled;
        combobox.toggle.disabled = disabled;
        if (disabled) combobox.close();
      },
    };
  }
  function selectControl(id, title, options, value, onChange) {
    const group = node("label", "session-control"),
      select = node("select");
    select.id = id;
    const update = (choices, selected) => {
      const signature = JSON.stringify(choices);
      if (select.dataset.options !== signature) {
        select.replaceChildren(
          ...choices.map(([value, label]) => new Option(label, value)),
        );
        select.dataset.options = signature;
      }
      select.value = selected;
    };
    update(options, value);
    select.addEventListener("change", () => onChange(select.value));
    group.append(node("span", "", title), select);
    return { group, select, update };
  }
  function composer(id, draft, placeholder, onInput, onSend, onStop) {
    const element = node("div", "session-composer"),
      textarea = node("textarea");
    textarea.id = `${id}Composer`;
    textarea.rows = 2;
    textarea.value = draft;
    textarea.placeholder = placeholder;
    textarea.setAttribute("aria-label", "Message");
    const actions = node("div", "session-composer-actions"),
      status = node("span", "session-composer-status");
    status.id = `${id}ComposerStatus`;
    status.setAttribute("aria-live", "polite");
    const send = button("Send", "primary-button", onSend),
      stop = button("Stop", "danger-button", onStop);
    send.id = `${id}Send`;
    stop.id = `${id}Stop`;
    stop.hidden = true;
    textarea.addEventListener("input", () => {
      resizeComposer(textarea);
      onInput(textarea.value);
    });
    textarea.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        if (!send.disabled && !send.hidden) onSend();
      }
    });
    actions.append(status, send, stop);
    element.append(textarea, actions);
    return element;
  }
  function message(role, label) {
    const element = node("article", `session-message ${role}-message`);
    element.append(node("div", "session-message-label", label));
    return element;
  }
  function thinking() {
    const element = node("details", "session-thinking");
    element.append(node("summary", "", "Thinking"));
    return element;
  }
  function resizeComposer(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${textarea.scrollHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > textarea.clientHeight ? "auto" : "hidden";
  }
  function nearBottom(scroller) {
    return (
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80
    );
  }
  window.SessionUI = {
    libraryHeader,
    search,
    card,
    shell,
    header,
    modelControl,
    selectControl,
    composer,
    message,
    thinking,
    resizeComposer,
    nearBottom,
  };
})();

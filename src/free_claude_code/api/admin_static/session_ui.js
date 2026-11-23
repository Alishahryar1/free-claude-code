(() => {
  "use strict";
  function node(tag, className = "", text = "") {
    const element = document.createElement(tag);
    if (className) element.className = className;
    element.textContent = text;
    return window.FccFormControls.configure(element);
  }
  function button(label, className, action) {
    const element = node("button", className, label);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  }
  const ICONS = {
    send: ["M12 19V5", "M5 12l7-7 7 7"],
    trash: [
      "M3 6h18",
      "M8 6V4h8v2",
      "M19 6l-1 14H6L5 6",
      "M10 11v6",
      "M14 11v6",
    ],
  };
  function icon(name) {
    const ns = "http://www.w3.org/2000/svg",
      svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2.2");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    for (const d of ICONS[name]) {
      const path = document.createElementNS(ns, "path");
      path.setAttribute("d", d);
      svg.append(path);
    }
    return svg;
  }
  // Icon-only button: the label stays as its accessible name and tooltip.
  function setIcon(element, label, name) {
    element.classList.add("icon-button");
    if (element.dataset.icon !== name) {
      element.replaceChildren(icon(name));
      element.dataset.icon = name;
    }
    if (element.getAttribute("aria-label") !== label) {
      element.setAttribute("aria-label", label);
      element.title = label;
    }
    return element;
  }
  // Rule-based difficulty estimate for a first prompt. Deliberately simple and
  // transparent: every point added comes with a human-readable reason.
  const HARD_WORDS = [
    /\brefactor/i, /\barchitect/i, /\bredesign/i, /\bmigrat/i, /\brewrite\b/i,
    /\bfrom scratch\b/i, /\bwhole (project|codebase|repo)/i, /\bentire (project|codebase|repo)/i,
    /\bperformance\b/i, /\boptimi[sz]/i, /\bconcurren/i, /\brace condition/i,
    /\bsecurity\b/i, /\broot cause\b/i, /\binvestigate\b/i, /\bdebug/i,
    /არქიტექტურ/, /რეფაქტორ/, /მთელი პროექტ/, /ბაგ/, /ოპტიმიზაც/, /დააპროექტ/,
    /გადააკეთე/, /გადაწერე/, /მიგრაც/, /უსაფრთხოებ/,
  ];
  const EASY_WORDS = [
    /\btypo\b/i, /\brename\b/i, /\bexplain\b/i, /\bwhat (is|does)\b/i,
    /\bcomment\b/i, /\bformat\b/i, /\btranslate\b/i, /\bsummari[sz]e\b/i,
    /განმარტ/, /რა არის/, /თარგმნე/, /კომენტარ/, /სახელი გადაარქვი/,
  ];
  function classifyPrompt(text) {
    const value = String(text || "");
    let score = 0;
    const reasons = [];
    const add = (points, reason) => {
      score += points;
      if (points > 0 && reason) reasons.push(reason);
    };
    const length = value.trim().length;
    if (length > 1200) add(4, "long request");
    else if (length > 500) add(3, "detailed request");
    else if (length > 200) add(2, "multi-sentence request");
    else if (length > 120) add(1, "")
    if (/```/.test(value)) add(1, "code block");
    if (
      /Traceback \(most recent call last\)|^\s+at .+\(.+:\d+:\d+\)|\b\w*(Error|Exception):/m.test(
        value,
      )
    )
      add(2, "stack trace");
    const files = new Set(
      value.match(/[\w./\\-]+\.(py|js|jsx|ts|tsx|java|go|rs|cs|cpp|c|h|html|css|scss|json|ya?ml|md|sql|vue|php)\b/gi) || [],
    );
    if (files.size >= 3) add(2, `${files.size} files mentioned`);
    else if (files.size >= 1) add(1, "file mentioned");
    const hard = HARD_WORDS.filter((pattern) => pattern.test(value));
    if (hard.length)
      add(hard.length >= 3 ? 5 : hard.length * 2, "complex task wording");
    else if (length < 80) score -= 1;
    const bullets = value.split("\n").filter((line) => /^\s*(\d+[.)]|[-*•])\s+\S/.test(line));
    if (bullets.length >= 3) add(1, "multiple steps");
    if (EASY_WORDS.some((pattern) => pattern.test(value))) score -= 1;
    const level = score >= 5 ? "hard" : score >= 2 ? "medium" : "easy";
    if (!reasons.length) reasons.push(level === "easy" ? "short, simple request" : "everyday task");
    return { level, score, reasons };
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
    if (back) row.append(button(backLabel, "session-back-button", back));
    row.append(title, ...actions);
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
    let committed = value,
      optionsSignature = JSON.stringify(values());
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
        const signature = JSON.stringify(values());
        if (combobox.isOpen && signature !== optionsSignature) {
          const active = combobox.visibleOptions[combobox.activeIndex]?.dataset.value;
          combobox.render(combobox.query);
          const index = combobox.visibleOptions.findIndex((option) => option.dataset.value === active);
          if (index >= 0) combobox.setActive(index, false);
        }
        optionsSignature = signature;
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
    const actions = node("div", "session-composer-actions");
    const send = setIcon(
        button("Send", "primary-button", onSend),
        "Send",
        "send",
      ),
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
    actions.append(send, stop);
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
    classifyPrompt,
    icon,
    setIcon,
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

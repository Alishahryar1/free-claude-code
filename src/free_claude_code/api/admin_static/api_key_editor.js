/* Ordered masked credentials. Saved references travel with rows, never positions. */
window.FccApiKeyEditor = class {
  constructor(field, onChange) {
    this.field = field;
    this.onChange = onChange;
    this.revision = field.value?.revision ?? null;
    this.entries = (field.value?.entries || []).map((entry) => ({ reference: entry.reference, value: "" }));
    this.dragged = null;
    this.picked = null;
    this.element = document.createElement("div");
    this.element.className = "api-key-editor";
    this.element.setAttribute("role", "group");
    this.element.setAttribute("aria-labelledby", `field-${field.key}-label`);
    this.element.apiKeyEditor = this;
    this.input = document.createElement("input");
    this.input.type = "hidden";
    this.input.id = `field-${field.key}`;
    this.input.dataset.key = field.key;
    this.input.dataset.fieldType = "api_key_list";
    this.input.disabled = field.locked;
    this.input.value = JSON.stringify(this.payload());
    this.input.dataset.original = this.input.value;
    this.rows = document.createElement("div");
    this.rows.className = "api-key-rows";
    this.rows.addEventListener("dragover", (event) => {
      if (!this.dragged || !this.editable()) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });
    this.rows.addEventListener("drop", (event) => {
      if (!this.dragged || !this.editable()) return;
      event.preventDefault();
      const insertion = [...this.rows.children].findIndex((row) => {
        const bounds = row.getBoundingClientRect();
        return event.clientY < bounds.top + bounds.height / 2;
      });
      const source = this.entries.indexOf(this.dragged);
      const destination = insertion < 0 ? this.entries.length : insertion;
      this.move(this.dragged, destination - (source < destination ? 1 : 0));
      this.dragged = null;
    });
    this.empty = document.createElement("div");
    this.empty.className = "field-description";
    this.empty.textContent = "No keys configured.";
    this.add = this.button("Add key", "Add key", () => {
      if (!this.editable()) return;
      this.picked = null;
      this.entries.push({ value: "" });
      this.changed();
      this.render();
      this.rows.lastElementChild.querySelector("input").focus();
    });
    this.add.className = "secondary-button api-key-add";
    this.announcement = document.createElement("div");
    this.announcement.className = "sr-only";
    this.announcement.setAttribute("aria-live", "polite");
    this.element.append(this.input, this.rows, this.empty, this.add, this.announcement);
    this.render();
  }

  editable() { return !this.field.locked && !this.element.closest("[inert]"); }

  payload() {
    return { revision: this.revision, entries: this.entries.map((entry) =>
      entry.value.trim() || entry.reference === undefined ? { value: entry.value.trim() } : { retain: entry.reference },
    ) };
  }

  changed() {
    this.clearResults();
    this.input.value = JSON.stringify(this.payload());
    this.onChange();
  }

  clearResults() {
    this.rows.querySelectorAll(".api-key-result").forEach((result) => { result.textContent = ""; result.hidden = true; });
    this.rows.querySelectorAll("input").forEach((input) => {
      input.removeAttribute("aria-invalid");
      input.removeAttribute("aria-describedby");
    });
  }

  showResult(index, message, rejected = false) {
    const row = this.rows.children[index];
    if (!row) return null;
    const result = row.querySelector(".api-key-result");
    const input = row.querySelector("input");
    result.textContent = message;
    result.hidden = false;
    result.className = `api-key-result ${rejected ? "field-error" : "field-warning"}`;
    input.setAttribute("aria-describedby", result.id);
    if (rejected) input.setAttribute("aria-invalid", "true");
    return input;
  }

  validate() {
    this.clearResults();
    let first = null;
    const seen = new Map();
    this.entries.forEach((entry, index) => {
      const value = entry.value.trim();
      if (!value && entry.reference === undefined) {
        const input = this.showResult(index, "Enter a key or remove this row.", true);
        first ||= input;
      } else if (value) {
        if (seen.has(value)) {
          const previous = seen.get(value);
          const input = this.showResult(previous, `Duplicates Key ${index + 1}.`, true);
          first ||= input;
          this.showResult(index, `Duplicates Key ${previous + 1}.`, true);
        } else seen.set(value, index);
      }
    });
    return first;
  }

  move(entry, destination) {
    const source = this.entries.indexOf(entry);
    if (source === destination || source < 0) return;
    this.entries.splice(source, 1);
    this.entries.splice(destination, 0, entry);
    this.changed();
    this.render();
    this.rows.children[destination].querySelector(".api-key-handle").focus();
    this.announcement.textContent = `Key moved to position ${destination + 1} of ${this.entries.length}.`;
  }

  keyboard(event, entry) {
    if (!this.editable()) return;
    const index = this.entries.indexOf(entry);
    if (event.key === " " || event.key === "Enter") {
      event.preventDefault();
      if (this.picked) {
        this.picked = null;
        this.announcement.textContent = `Key dropped at position ${index + 1}.`;
      } else {
        this.picked = { entry, order: [...this.entries] };
        this.announcement.textContent = `Picked up Key ${index + 1}. Use arrow keys to move, Space to drop, or Escape to cancel.`;
      }
    } else if (this.picked?.entry === entry && ["ArrowUp", "ArrowDown", "Escape"].includes(event.key)) {
      event.preventDefault();
      event.stopPropagation();
      if (event.key === "Escape") {
        this.entries = this.picked.order;
        this.picked = null;
        this.changed();
        this.render();
        this.rows.children[this.entries.indexOf(entry)].querySelector(".api-key-handle").focus();
        this.announcement.textContent = "Move cancelled.";
      } else this.move(entry, Math.max(0, Math.min(this.entries.length - 1, index + (event.key === "ArrowUp" ? -1 : 1))));
    }
  }

  button(text, label, action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button";
    button.textContent = text;
    button.setAttribute("aria-label", label);
    button.disabled = !!this.field.locked;
    if (action) button.addEventListener("click", action);
    return button;
  }

  render() {
    this.rows.replaceChildren();
    this.empty.hidden = !!this.entries.length;
    this.entries.forEach((entry, index) => {
      const row = document.createElement("div");
      row.className = "api-key-row";
      const number = document.createElement("span");
      number.className = "api-key-number";
      number.textContent = index + 1;
      const handle = this.button("", `Move Key ${index + 1}`);
      handle.classList.add("api-key-handle");
      handle.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><circle cx="8" cy="6" r="1.5"/><circle cx="16" cy="6" r="1.5"/><circle cx="8" cy="12" r="1.5"/><circle cx="16" cy="12" r="1.5"/><circle cx="8" cy="18" r="1.5"/><circle cx="16" cy="18" r="1.5"/></svg>';
      handle.draggable = !this.field.locked;
      handle.addEventListener("keydown", (event) => this.keyboard(event, entry));
      handle.addEventListener("dragstart", (event) => {
        if (!this.editable()) { event.preventDefault(); return; }
        this.dragged = entry;
        this.picked = null;
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", "api-key-row");
        row.classList.add("dragging");
      });
      handle.addEventListener("dragend", () => { this.dragged = null; row.classList.remove("dragging"); });
      const input = window.FccFormControls.configure(document.createElement("input"));
      input.id = `field-${this.field.key}-${index + 1}`;
      input.type = "password";
      input.autocomplete = "off";
      input.setAttribute("autocapitalize", "none");
      input.setAttribute("autocorrect", "off");
      input.spellcheck = false;
      input.disabled = !!this.field.locked;
      input.setAttribute("aria-label", `Key ${index + 1}`);
      input.placeholder = entry.reference === undefined ? "Enter API key" : "Configured - enter a new key to replace";
      input.value = entry.value;
      input.addEventListener("input", () => { entry.value = input.value; this.changed(); });
      const remove = this.button("Remove", `Remove Key ${index + 1}`, () => {
        if (!this.editable()) return;
        this.picked = null;
        this.entries.splice(this.entries.indexOf(entry), 1);
        this.changed();
        this.render();
      });
      remove.className = "danger-button api-key-remove";
      const result = document.createElement("div");
      result.id = `${input.id}-result`;
      result.className = "api-key-result";
      result.hidden = true;
      row.append(handle, number, input, remove, result);
      this.rows.appendChild(row);
    });
  }
};

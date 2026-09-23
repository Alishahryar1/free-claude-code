(() => {
  class FccModelCombobox {
    constructor(
      input,
      {
        listboxId,
        label,
        values,
        emptyMessage,
        registry,
        onSelect = null,
        onClose = null,
      },
    ) {
      this.input = input;
      this.getValues = values;
      this.getEmptyMessage = emptyMessage;
      this.registry = registry;
      this.onSelect = onSelect;
      this.onClose = onClose;
      this.activeIndex = -1;
      this.query = "";

      this.element = document.createElement("div");
      this.element.className = "model-combobox";
      this.listbox = document.createElement("div");
      this.listbox.className = "model-combobox-list";
      this.listbox.id = listboxId;
      this.listbox.setAttribute("role", "listbox");
      this.listbox.hidden = true;
      this.toggle = document.createElement("button");
      this.toggle.type = "button";
      this.toggle.className = "model-combobox-toggle";
      this.toggle.disabled = input.disabled;
      this.toggle.setAttribute("aria-label", `Show ${label} options`);

      input.setAttribute("role", "combobox");
      input.setAttribute("aria-autocomplete", "list");
      input.setAttribute("aria-haspopup", "listbox");
      for (const control of [input, this.toggle]) {
        control.setAttribute("aria-controls", this.listbox.id);
        control.setAttribute("aria-expanded", "false");
      }

      input.addEventListener("click", () => this.open());
      input.addEventListener("input", () => this.open(input.value));
      input.addEventListener("keydown", (event) => this.handleKeydown(event));
      this.toggle.addEventListener("mousedown", (event) => event.preventDefault());
      this.toggle.addEventListener("click", () => {
        if (this.isOpen) this.close();
        else this.open();
        input.focus();
      });
      this.listbox.addEventListener("mousedown", (event) => event.preventDefault());
      this.listbox.addEventListener("mousemove", (event) => {
        const optionEl = event.target.closest('[role="option"]');
        if (optionEl) this.setActive(this.visibleOptions.indexOf(optionEl));
      });
      this.listbox.addEventListener("click", (event) => {
        const optionEl = event.target.closest('[role="option"]');
        if (optionEl) this.select(optionEl.dataset.value);
      });

      this.element.append(input, this.toggle, this.listbox);
      registry.add(this);
    }

    get isOpen() {
      return this.element.classList.contains("open");
    }

    get visibleOptions() {
      return Array.from(this.listbox.querySelectorAll('[role="option"]'));
    }

    open(query = this.input.value) {
      if (this.input.disabled) return;
      this.registry.forEach((combobox) => {
        if (combobox !== this && combobox.isOpen) combobox.close();
      });
      this.render(query);
      this.element.classList.add("open");
      this.listbox.hidden = false;
      this.setExpanded(true);
    }

    close() {
      if (!this.isOpen) return;
      this.element.classList.remove("open");
      this.listbox.hidden = true;
      this.activeIndex = -1;
      this.input.removeAttribute("aria-activedescendant");
      this.setExpanded(false);
      if (this.onClose) this.onClose();
    }

    setExpanded(expanded) {
      for (const control of [this.input, this.toggle]) {
        control.setAttribute("aria-expanded", String(expanded));
      }
    }

    render(query) {
      this.query = query;
      const normalizedQuery = query.trim().toLocaleLowerCase();
      const allValues = this.getValues();
      const values = normalizedQuery
        ? allValues.filter((value) =>
            value.toLocaleLowerCase().includes(normalizedQuery),
          )
        : allValues;
      this.listbox.replaceChildren();

      if (values.length === 0) {
        const empty = document.createElement("div");
        empty.className = "model-combobox-empty";
        empty.textContent = this.getEmptyMessage();
        this.listbox.appendChild(empty);
        this.activeIndex = -1;
        this.input.removeAttribute("aria-activedescendant");
        return;
      }

      values.forEach((value, index) => {
        const optionEl = document.createElement("div");
        optionEl.className = "model-combobox-option";
        optionEl.id = `${this.listbox.id}-option-${index}`;
        optionEl.dataset.value = value;
        optionEl.setAttribute("role", "option");
        optionEl.textContent = value;
        this.listbox.appendChild(optionEl);
      });
      const selectedIndex = values.indexOf(this.input.value);
      this.setActive(selectedIndex >= 0 ? selectedIndex : 0, false);
    }

    setActive(index, scroll = true) {
      const options = this.visibleOptions;
      if (options.length === 0) return;
      this.activeIndex = Math.max(0, Math.min(index, options.length - 1));
      options.forEach((optionEl, optionIndex) => {
        const active = optionIndex === this.activeIndex;
        optionEl.classList.toggle("active", active);
        optionEl.setAttribute("aria-selected", String(active));
      });
      const activeOption = options[this.activeIndex];
      this.input.setAttribute("aria-activedescendant", activeOption.id);
      if (scroll) activeOption.scrollIntoView({ block: "nearest" });
    }

    move(offset) {
      const count = this.visibleOptions.length;
      if (count) this.setActive((this.activeIndex + offset + count) % count);
    }

    select(value) {
      this.input.value = value;
      if (this.onSelect) this.onSelect(value);
      this.input.dispatchEvent(new Event("change", { bubbles: true }));
      this.close();
      this.input.focus();
    }

    handleKeydown(event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (this.isOpen) {
          this.move(event.key === "ArrowDown" ? 1 : -1);
        } else {
          this.open();
          if (event.key === "ArrowUp") {
            this.setActive(this.visibleOptions.length - 1);
          }
        }
      } else if (this.isOpen && (event.key === "Home" || event.key === "End")) {
        event.preventDefault();
        this.setActive(event.key === "Home" ? 0 : this.visibleOptions.length - 1);
      } else if (this.isOpen && event.key === "Enter") {
        const active = this.visibleOptions[this.activeIndex];
        if (active) {
          event.preventDefault();
          this.select(active.dataset.value);
        }
      } else if (this.isOpen && event.key === "Escape") {
        event.preventDefault();
        this.close();
      } else if (this.isOpen && event.key === "Tab") {
        this.close();
      }
    }
  }

  window.FccModelCombobox = FccModelCombobox;
})();

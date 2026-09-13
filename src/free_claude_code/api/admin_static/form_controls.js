(() => {
  "use strict";
  window.FccFormControls = {
    configure(node) {
      if (node.matches('input:not([type="hidden"]), textarea, form')) {
        node.autocomplete = "off";
        // Advisory opt-outs for 1Password, Bitwarden, and Dashlane.
        node.setAttribute("data-1p-ignore", "");
        node.setAttribute("data-bwignore", "");
        node.setAttribute("data-form-type", "other");
      }
      return node;
    },
  };
})();

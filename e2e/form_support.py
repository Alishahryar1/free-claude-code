"""Browser contracts for FCC's advisory autofill opt-outs."""


def assert_autofill_opt_out(page):
    failures = page.locator(
        'form, textarea, input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"])'
    ).evaluate_all("""nodes => nodes.flatMap(node => {
      const valid = node.autocomplete === 'off' && node.hasAttribute('data-1p-ignore') &&
        node.hasAttribute('data-bwignore') && node.getAttribute('data-form-type') === 'other';
      return valid ? [] : [{id: node.id, tag: node.tagName, type: node.type}];
    })""")
    assert failures == [], failures

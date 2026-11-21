"""Browser contracts for FCC's advisory autofill opt-outs."""

from playwright.sync_api import Page


def assert_autofill_opt_out(page: Page):
    failures = page.locator(
        'form, textarea, input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"])'
    ).evaluate_all("""nodes => nodes.flatMap(node => {
      const valid = node.autocomplete === 'off' && node.hasAttribute('data-1p-ignore') &&
        node.hasAttribute('data-bwignore') && node.getAttribute('data-form-type') === 'other';
      return valid ? [] : [{id: node.id, tag: node.tagName, type: node.type}];
    })""")
    assert failures == [], failures

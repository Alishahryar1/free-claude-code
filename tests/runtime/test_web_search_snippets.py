"""DDG Lite snippets remain attached to their result, including duplicate links."""

from free_claude_code.runtime.web_tools.parsers import SearchResultParser


def test_snippets_follow_their_result_links():
    parser = SearchResultParser()
    parser.feed("""
      <tr><td><a class='result-link' href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa'>First</a></td></tr>
      <tr><td class='result-snippet'>The <b>first</b> indexed snippet.</td></tr>
      <tr><td><a class='result-link' href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb'>Second</a></td></tr>
      <tr><td class='result-snippet'>Second snippet.</td></tr>
      <tr><td><a class='result-link' href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa'>Duplicate first</a></td></tr>
      <tr><td class='result-snippet'>Updated first snippet.</td></tr>
    """)
    assert len(parser.results) == 2
    assert parser.results[0].snippet == "Updated first snippet."
    assert parser.results[1].snippet == "Second snippet."

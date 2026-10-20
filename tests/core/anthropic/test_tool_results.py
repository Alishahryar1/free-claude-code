from free_claude_code.core.anthropic.tool_results import (
    ToolResultImage,
    ToolResultText,
    decompose_tool_result_content,
)


def test_decompose_tool_result_preserves_mixed_part_order() -> None:
    result = decompose_tool_result_content(
        [
            {"type": "text", "text": "before"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8=",
                },
            },
            {"type": "text", "text": "after"},
        ]
    )

    assert result.has_images is True
    assert result.parts == (
        ToolResultText("before"),
        ToolResultImage("data:image/png;base64,aGVsbG8="),
        ToolResultText("after"),
    )


def test_decompose_tool_result_preserves_unknown_siblings_as_text() -> None:
    result = decompose_tool_result_content(
        [
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://images.example.test/result.png",
                },
            },
            {"type": "document", "title": "café"},
            7,
        ]
    )

    assert result.parts == (
        ToolResultImage("https://images.example.test/result.png"),
        ToolResultText('{"type": "document", "title": "café"}'),
        ToolResultText("7"),
    )


def test_decompose_text_only_result_does_not_claim_an_image() -> None:
    result = decompose_tool_result_content(
        [{"type": "text", "text": "first"}, {"answer": 42}]
    )

    assert result.has_images is False
    assert result.parts == (
        ToolResultText("first"),
        ToolResultText('{"answer": 42}'),
    )


def test_decompose_single_image_dictionary() -> None:
    result = decompose_tool_result_content(
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://images.example.test/single.png",
            },
        }
    )

    assert result.has_images is True
    assert result.parts == (ToolResultImage("https://images.example.test/single.png"),)

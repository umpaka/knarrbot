"""Markdown-to-Telegram-HTML converter.

Converts common Markdown formatting to Telegram-compatible HTML.
Telegram supports: <b>, <i>, <code>, <pre>, <a>, <s>, <u>.
Falls back to plain text if conversion produces invalid HTML.
"""

import re
import html
import logging

log = logging.getLogger("telegram-format")


def markdown_to_telegram_html(text: str) -> str:
    """Convert Markdown text to Telegram-safe HTML.

    Handles: bold, italic, code, code blocks, strikethrough.
    Strips: headers (#), images (![](...)), horizontal rules (---).
    Preserves: line breaks, plain text.

    Returns HTML string suitable for Telegram's parse_mode="HTML".
    """
    if not text:
        return text

    # Protect code blocks first (don't process markdown inside them)
    code_blocks: list[str] = []

    def _save_code_block(m):
        code_blocks.append(m.group(2) or m.group(3))
        return f"\x00CODEBLOCK{len(code_blocks) - 1}\x00"

    # ```lang\ncode\n``` or ```code```
    text = re.sub(r"```(\w*)\n(.*?)```|```(.*?)```", _save_code_block, text, flags=re.DOTALL)

    inline_codes: list[str] = []

    def _save_inline_code(m):
        inline_codes.append(m.group(1))
        return f"\x00INLINECODE{len(inline_codes) - 1}\x00"

    # `inline code`
    text = re.sub(r"`([^`\n]+)`", _save_inline_code, text)

    # HTML-escape the remaining text (must happen after extracting code blocks)
    text = html.escape(text, quote=False)

    # --- Markdown → HTML conversions ---

    # Bold+Italic: ***text*** (must be handled BEFORE separate bold/italic)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"___(.+?)___", r"<b><i>\1</i></b>", text)

    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # Italic: *text* or _text_ (but not inside words like file_name)
    text = re.sub(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

    # Strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)

    # Links: [text](url) → <a href="url">text</a>
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', text)

    # Strip markdown headers (# Header → Header)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Strip images: ![alt](url)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", text)

    # Strip horizontal rules
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Blockquotes: > text → <blockquote>text</blockquote>
    # Collect consecutive > lines into a single blockquote
    def _convert_blockquotes(t):
        lines = t.split("\n")
        result = []
        in_quote = False
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("&gt; ") or stripped == "&gt;":
                # Remove the &gt; prefix (HTML-escaped >)
                content = stripped[5:] if stripped.startswith("&gt; ") else ""
                if not in_quote:
                    result.append("<blockquote>")
                    in_quote = True
                result.append(content)
            else:
                if in_quote:
                    result.append("</blockquote>")
                    in_quote = False
                result.append(line)
        if in_quote:
            result.append("</blockquote>")
        return "\n".join(result)

    text = _convert_blockquotes(text)

    # Bullet lists: * item or - item → • item
    text = re.sub(r"^[\*\-]\s+", "• ", text, flags=re.MULTILINE)

    # Numbered lists: 1. item → 1. item (leave as-is, just clean up)

    # --- Restore code blocks ---
    for i, code in enumerate(inline_codes):
        escaped = html.escape(code, quote=False)
        text = text.replace(f"\x00INLINECODE{i}\x00", f"<code>{escaped}</code>")

    for i, code in enumerate(code_blocks):
        escaped = html.escape(code, quote=False)
        text = text.replace(f"\x00CODEBLOCK{i}\x00", f"<pre>{escaped}</pre>")

    return text.strip()


def safe_html_reply(text: str) -> tuple:
    """Convert text to HTML with fallback.

    Returns (formatted_text, parse_mode).
    If HTML conversion looks safe, returns (html, "HTML").
    Otherwise returns (original, "") for plain text.
    """
    try:
        html_text = markdown_to_telegram_html(text)
        # Quick validation: check that tags are balanced
        # Use regex with word boundary to avoid prefix matching (e.g. <s vs <strong>)
        for tag in ("b", "i", "code", "pre", "s", "u", "a", "blockquote"):
            opens = len(re.findall(rf"<{tag}[\s>]", html_text))
            closes = html_text.count(f"</{tag}>")
            if opens != closes:
                log.warning("Unbalanced <%s> tags (%d open, %d close), falling back to plain text", tag, opens, closes)
                return text, ""
        return html_text, "HTML"
    except Exception:
        log.warning("HTML conversion failed, falling back to plain text", exc_info=True)
        return text, ""

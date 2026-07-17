from __future__ import annotations

# Languages written without inter-word spaces. Canonical source shared by smart_split (layout /
# joiner) and breakpoints (phrase atoms) so the set lives in exactly one place instead of being
# hand-synced across modules (the two import the other's module, so neither can own it without a
# circular import). The aligner supports a smaller subset, but every consumer agrees that
# Cantonese (yue) uses the same no-space writing policy as Chinese.
LANGUAGES_WITHOUT_SPACES = {"zh", "yue", "ja", "th", "lo", "my"}

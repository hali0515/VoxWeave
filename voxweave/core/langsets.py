from __future__ import annotations

# Languages written without inter-word spaces, as the segmentation core sees them. layout (tokens,
# joiner, line budgets), breakpoints (phrase atoms), providers, subunit, unit_repair and the align
# seed/distribution code all read this one set; it sits in a dependency-free leaf module so any of
# them can import it without pulling in the engine. It is not the only such set:
# realign.NO_SPACE_LANGS ({"zh", "ja", "yue"}) is a second, narrower one -- the languages the
# aligners emit per-character units for -- used by realign, the aligners and the pipeline's word
# joiner, so th/lo/my are no-space here but space-joined there. Both treat Cantonese (yue) like
# Chinese.
LANGUAGES_WITHOUT_SPACES = {"zh", "yue", "ja", "th", "lo", "my"}

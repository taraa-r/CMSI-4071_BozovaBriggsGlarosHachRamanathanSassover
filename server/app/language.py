"""English check before text inference (SRS 5.3.3.71), using lingua-language-detector."""

from lingua import Language, LanguageDetectorBuilder

from .config import ENGLISH_MIN_CONFIDENCE

# Common web languages the detector chooses between. Text in other languages is matched to the
# closest of these, which is still "not English" in almost all cases.
LANGUAGES = [
    Language.ENGLISH, Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.ITALIAN,
    Language.PORTUGUESE, Language.DUTCH, Language.RUSSIAN, Language.POLISH, Language.TURKISH,
    Language.ARABIC, Language.HINDI, Language.CHINESE, Language.JAPANESE, Language.KOREAN,
    Language.VIETNAMESE, Language.INDONESIAN, Language.TAGALOG,
]


class LanguageChecker:
    def __init__(self):
        self.detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()

    def is_english(self, text: str) -> bool:
        confidences = self.detector.compute_language_confidence_values(text)
        if not confidences:
            return False
        top = confidences[0]
        return top.language == Language.ENGLISH and top.value >= ENGLISH_MIN_CONFIDENCE

"""
guardrails/abstention.py
─────────────────────────
Language-specific abstention messages.

Design rationale:
- When any guardrail triggers (off-topic query, unsafe query, low retrieval
  confidence, or NLI failure), the system must gracefully decline to answer.
- The decline message MUST be in the language of the query to maintain UX.
- These messages are pre-translated by native speakers/high-quality models.
"""

from __future__ import annotations

ABSTENTION_MESSAGES = {
    "hi": "मुझे इस प्रश्न का उत्तर अपने स्रोतों में नहीं मिला।",
    "ta": "எனது ஆதாரங்களில் இந்த தகவலை என்னால் கண்டுபிடிக்க முடியவில்லை.",
    "te": "నా మూలాలలో ఈ సమాచారం నాకు కనుగొనబడలేదు.",
    "bn": "আমার উৎসগুলিতে এই তথ্য পাওয়া যায়নি।",
    "mr": "माझ्या स्रोतांमध्ये ही माहिती मला सापडली नाही.",
    "gu": "મારા સ્ત્રોતોમાં આ માહિતી મળી શકી નથી.",
    "kn": "ನನ್ನ ಮೂಲಗಳಲ್ಲಿ ಈ ಮಾಹಿತಿ ನನಗೆ ಸಿಗಲಿಲ್ಲ.",
    "ml": "എന്റെ ഉറവിടങ്ങളിൽ ഈ വിവരം എനിക്ക് കണ്ടെത്താനായില്ല.",
    "pa": "ਮੇਰੇ ਸਰੋਤਾਂ ਵਿੱਚ ਇਹ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।",
    "or": "ମୋ ଉତ୍ସଗୁଡ଼ିକରେ ଏହି ତଥ୍ୟ ମିଳିଲା ନାହିଁ।",
    "as": "মোৰ উৎসবোৰত এই তথ্য পোৱা নগ'ল।",
    "ur": "مجھے اپنے ذرائع میں یہ معلومات نہیں مل سکیں۔",
    "en": "I couldn't find this information in my sources.",
}


def get_abstention_message(language: str) -> str:
    """Return the correct abstention message for the given ISO code."""
    # Fallback to English if language is missing or unknown
    return ABSTENTION_MESSAGES.get(language, ABSTENTION_MESSAGES["en"])

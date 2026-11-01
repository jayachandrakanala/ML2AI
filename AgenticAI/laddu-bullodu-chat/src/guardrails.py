import re

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except Exception as e:
    print(f"[GUARDRAILS] Presidio not available ({e}). Falling back to regex-based guardrails.")
    PRESIDIO_AVAILABLE = False


def sanitize_text(text: str) -> str:
    """Remove supported PII using Presidio when available and regex fallbacks.

    Person names are deliberately excluded from Presidio detection to avoid
    redacting public figures. Empty input passes through unchanged; regex rules
    still run after Presidio so domain-specific IDs and common contacts are
    covered even when the NLP analyzer is unavailable.
    """
    if not text:
        return text

    sanitized = text

    # Layer 1: NLP-based entity detection via Microsoft Presidio 
    # (Excluded 'PERSON' to prevent public figures/journalists in web search from becoming <PERSON>)
    if PRESIDIO_AVAILABLE:
        try:
            results = analyzer.analyze(
                text=sanitized,
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"],
                language="en"
            )
            anonymized_result = anonymizer.anonymize(text=sanitized, analyzer_results=results)
            sanitized = anonymized_result.text
        except Exception as e:
            print(f"[GUARDRAILS ERROR] Presidio analysis failed: {e}")

    # Layer 2: Domain-specific Regex Safeguards for True Private PII
    # Redact 5 to 8 digit numeric ID numbers (e.g., School ID 77277)
    sanitized = re.sub(r'\b\d{5,8}\b', '[REDACTED_ID]', sanitized)

    # Redact standard email addresses
    sanitized = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', sanitized)

    # Redact phone numbers
    sanitized = re.sub(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', sanitized)

    return sanitized
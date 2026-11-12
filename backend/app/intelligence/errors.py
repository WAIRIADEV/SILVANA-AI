
class ProviderError(Exception):
    """Base exception for normalized AI provider failures."""


class ProviderUnavailable(ProviderError):
    """The upstream provider is temporarily unavailable."""


class ProviderAuthenticationError(ProviderError):
    """The upstream provider rejected its credentials or permissions."""


class ProviderTimeout(ProviderError):
    """The upstream provider did not respond within the expected time."""


class ProviderRequestError(ProviderError):
    """The upstream provider rejected or failed to process a request."""


def normalize_status_error(
    status_code: int,
    message: str = "The upstream provider request failed.",
) -> ProviderError:
    """Convert an upstream HTTP status into a SILVANA exception."""

    if status_code in (408, 504):
        return ProviderTimeout(
            "The AI provider request timed out."
        )

    if status_code == 429:
        return ProviderUnavailable(
            "The AI provider is temporarily rate limited."
        )

    if status_code in (401, 403):
        return ProviderAuthenticationError(
            "The AI provider rejected its credentials or permissions."
        )

    if status_code >= 500:
        return ProviderUnavailable(
            "The AI provider is temporarily unavailable."
        )

    return ProviderRequestError(
        "The AI provider could not process the request."
    )

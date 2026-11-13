
const API_BASE_URL = '/api/v1'

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `API request failed: ${response.status}`

    try {
      const body = await response.json()
      if (typeof body.detail === 'string') {
        message = body.detail
      }
    } catch {
      // Keep the original HTTP error message.
    }

    throw new Error(message)
  }

  return response.json() as Promise<T>
}

export async function apiGet<T>(endpoint: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  })

  return handleResponse<T>(response)
}

export async function apiPost<TRequest, TResponse>(
  endpoint: string,
  body: TRequest,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })

  return handleResponse<TResponse>(response)
}

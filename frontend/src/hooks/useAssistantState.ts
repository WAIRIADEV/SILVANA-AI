
import { useCallback, useEffect, useRef, useState } from 'react'
import type { AssistantState } from '../types/assistant'

const MINIMUM_STATE_DURATION: Record<AssistantState, number> = {
  idle: 0,
  listening: 0,
  thinking: 350,
  executing: 450,
  verifying: 450,
  completed: 800,
  error: 1000,
}

function useAssistantState() {
  const [state, setState] = useState<AssistantState>('idle')

  const stateRef = useRef<AssistantState>('idle')
  const enteredAtRef = useRef(0)
  const queueRef = useRef<AssistantState[]>([])
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)
  const processQueueRef = useRef<() => void>(() => {})

  const commitState = useCallback((next: AssistantState) => {
    if (!mountedRef.current) return

    stateRef.current = next
    enteredAtRef.current = Date.now()
    setState(next)
  }, [])

  const processQueue = useCallback(() => {
    if (!mountedRef.current || timerRef.current !== null) return

    const current = stateRef.current
    const minimumDuration = MINIMUM_STATE_DURATION[current]
    const elapsed = Date.now() - enteredAtRef.current
    const remaining = Math.max(0, minimumDuration - elapsed)

    if (remaining > 0) {
      timerRef.current = setTimeout(() => {
        timerRef.current = null
        processQueueRef.current()
      }, remaining)

      return
    }

    const next = queueRef.current.shift()

    if (next !== undefined) {
      if (next !== current) {
        commitState(next)
      }

      processQueueRef.current()
      return
    }

    if (current === 'completed' || current === 'error') {
      commitState('idle')
    }
  }, [commitState])

  useEffect(() => {
    processQueueRef.current = processQueue
  }, [processQueue])

  const transitionTo = useCallback(
    (next: AssistantState) => {
      if (!mountedRef.current) return

      // Explicit reset immediately returns to IDLE.
      if (next === 'idle') {
        if (timerRef.current !== null) {
          clearTimeout(timerRef.current)
          timerRef.current = null
        }

        queueRef.current = []
        commitState('idle')
        return
      }

      // A new request takes priority over an older sequence.
      if (next === 'thinking') {
        if (timerRef.current !== null) {
          clearTimeout(timerRef.current)
          timerRef.current = null
        }

        queueRef.current = []
        commitState('thinking')
        processQueueRef.current()
        return
      }

      // Typing can interrupt an old terminal display.
      if (
        next === 'listening' &&
        (stateRef.current === 'completed' ||
          stateRef.current === 'error')
      ) {
        if (timerRef.current !== null) {
          clearTimeout(timerRef.current)
          timerRef.current = null
        }

        queueRef.current = []
        commitState('listening')
        return
      }

      const lastQueued =
        queueRef.current[queueRef.current.length - 1]

      if (
        next === lastQueued ||
        (next === stateRef.current && queueRef.current.length === 0)
      ) {
        return
      }

      queueRef.current.push(next)
      processQueueRef.current()
    },
    [commitState],
  )

  useEffect(() => {
    mountedRef.current = true

    return () => {
      mountedRef.current = false

      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }

      queueRef.current = []
    }
  }, [])

  return {
    state,
    transitionTo,
  }
}

export default useAssistantState

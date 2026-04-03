import { beforeEach, describe, expect, it } from '@jest/globals'

import {
  appendGoogleOAuthState,
  consumeGoogleOAuthState,
  isGoogleEmailAllowed,
} from '@/lib/server/auth'

describe('Google auth helpers', () => {
  beforeEach(() => {
    delete process.env.GOOGLE_AUTH_ALLOWED_EMAILS
    delete process.env.AUTH_USER_EMAIL
  })

  describe('isGoogleEmailAllowed', () => {
    it('supports wildcard allowlist values with extra formatting', () => {
      process.env.GOOGLE_AUTH_ALLOWED_EMAILS = ' "*" '

      expect(isGoogleEmailAllowed('anyone@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('another@workspace.dev')).toBe(true)
    })

    it('supports comma, semicolon, and whitespace-separated allowlist values', () => {
      process.env.GOOGLE_AUTH_ALLOWED_EMAILS =
        'first@example.com;\nsecond@example.com third@example.com'

      expect(isGoogleEmailAllowed('first@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('second@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('third@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('other@example.com')).toBe(false)
    })

    it('supports domain-style allowlist entries', () => {
      process.env.GOOGLE_AUTH_ALLOWED_EMAILS = '@example.com, partner.org'

      expect(isGoogleEmailAllowed('owner@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('dev@partner.org')).toBe(true)
      expect(isGoogleEmailAllowed('intruder@outside.org')).toBe(false)
    })

    it('falls back to AUTH_USER_EMAIL when allowlist is unset', () => {
      process.env.AUTH_USER_EMAIL = 'admin@example.com'

      expect(isGoogleEmailAllowed('admin@example.com')).toBe(true)
      expect(isGoogleEmailAllowed('guest@example.com')).toBe(false)
    })
  })

  describe('OAuth state cookie helpers', () => {
    it('supports multiple pending states and consumes them individually', () => {
      let cookieValue = appendGoogleOAuthState(undefined, 'state-a')
      cookieValue = appendGoogleOAuthState(cookieValue, 'state-b')

      const miss = consumeGoogleOAuthState(cookieValue, 'state-missing')
      expect(miss.matched).toBe(false)
      expect(miss.nextCookieValue).toBeTruthy()

      const consumeA = consumeGoogleOAuthState(miss.nextCookieValue ?? undefined, 'state-a')
      expect(consumeA.matched).toBe(true)
      expect(consumeA.nextCookieValue).toBeTruthy()

      const consumeB = consumeGoogleOAuthState(consumeA.nextCookieValue ?? undefined, 'state-b')
      expect(consumeB.matched).toBe(true)
      expect(consumeB.nextCookieValue).toBeNull()
    })

    it('accepts legacy single-state cookie values', () => {
      const result = consumeGoogleOAuthState('legacy-state', 'legacy-state')

      expect(result.matched).toBe(true)
      expect(result.nextCookieValue).toBeNull()
    })
  })
})

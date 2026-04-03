import { createHmac, timingSafeEqual } from "node:crypto";

import type { NextRequest } from "next/server";

export const AUTH_COOKIE_NAME = "qml_session";
export const SESSION_TTL_SECONDS = 60 * 60 * 12;
export const GOOGLE_OAUTH_STATE_COOKIE_NAME = "qml_google_oauth_state";
export const GOOGLE_OAUTH_STATE_TTL_SECONDS = 60 * 10;
const MAX_GOOGLE_OAUTH_STATES = 5;

function getAuthSecret(): string {
  const secret = process.env.AUTH_SECRET;
  if (!secret || secret.length < 16) {
    throw new Error("AUTH_SECRET must be set and at least 16 characters long.");
  }
  return secret;
}

function getAllowedCredentials(): { email: string; password: string } {
  const email = process.env.AUTH_USER_EMAIL;
  const password = process.env.AUTH_USER_PASSWORD;

  if (!email || !password) {
    throw new Error("AUTH_USER_EMAIL and AUTH_USER_PASSWORD must be configured.");
  }

  return { email, password };
}

function normalizeAllowListEntry(value: string): string {
  return value.trim().toLowerCase().replace(/^['"]+|['"]+$/g, "");
}

function normalizeEmail(email: string): string {
  return normalizeAllowListEntry(email);
}

function getExplicitGoogleAllowedEmails(): string[] {
  const raw = process.env.GOOGLE_AUTH_ALLOWED_EMAILS;
  if (!raw) {
    return [];
  }

  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (Array.isArray(parsed)) {
      return parsed
        .filter((value): value is string => typeof value === "string")
        .map((value) => normalizeAllowListEntry(value))
        .filter((value) => value.length > 0);
    }
  } catch {
    // Fallback to delimiter-based parsing.
  }

  return trimmed
    .split(/[\s,;]+/)
    .map((value) => normalizeAllowListEntry(value))
    .filter((value) => value.length > 0);
}

function normalizeGoogleOAuthState(state: string): string {
  return state.trim();
}

function normalizeGoogleOAuthStateList(states: string[]): string[] {
  const uniqueStates = new Set<string>();
  const normalized: string[] = [];

  for (const state of states) {
    const normalizedState = normalizeGoogleOAuthState(state);
    if (!normalizedState || uniqueStates.has(normalizedState)) {
      continue;
    }
    uniqueStates.add(normalizedState);
    normalized.push(normalizedState);
  }

  if (normalized.length <= MAX_GOOGLE_OAUTH_STATES) {
    return normalized;
  }

  return normalized.slice(normalized.length - MAX_GOOGLE_OAUTH_STATES);
}

function parseGoogleOAuthStateList(rawCookieValue: string | undefined): string[] {
  if (!rawCookieValue) {
    return [];
  }

  const parseJsonStateList = (rawValue: string): string[] | null => {
    try {
      const parsed = JSON.parse(rawValue) as unknown;
      if (Array.isArray(parsed)) {
        return parsed
          .filter((value): value is string => typeof value === "string")
          .map((value) => normalizeGoogleOAuthState(value));
      }
      if (typeof parsed === "string") {
        return [normalizeGoogleOAuthState(parsed)];
      }
    } catch {
      // Not a JSON payload.
    }

    return null;
  };

  const directJson = parseJsonStateList(rawCookieValue);
  if (directJson) {
    return normalizeGoogleOAuthStateList(directJson);
  }

  try {
    const decoded = Buffer.from(rawCookieValue, "base64url").toString("utf8");
    const decodedJson = parseJsonStateList(decoded);
    if (decodedJson) {
      return normalizeGoogleOAuthStateList(decodedJson);
    }
  } catch {
    // Not base64url; fallback to legacy format below.
  }

  return normalizeGoogleOAuthStateList([rawCookieValue]);
}

function serializeGoogleOAuthStateList(states: string[]): string | null {
  const normalized = normalizeGoogleOAuthStateList(states);
  if (normalized.length === 0) {
    return null;
  }

  const encoded = Buffer.from(JSON.stringify(normalized), "utf8").toString("base64url");
  return encoded;
}

function googleAllowListEntryMatchesEmail(entry: string, normalizedEmail: string): boolean {
  if (entry === "*") {
    return true;
  }

  if (entry.includes("@")) {
    if (entry.startsWith("@")) {
      return normalizedEmail.endsWith(entry);
    }
    return entry === normalizedEmail;
  }

  const domain = normalizedEmail.split("@")[1];
  if (!domain) {
    return false;
  }

  if (entry.startsWith("*.")) {
    const targetDomain = entry.slice(2);
    return domain === targetDomain || domain.endsWith(`.${targetDomain}`);
  }

  return domain === entry;
}

export function appendGoogleOAuthState(
  existingCookieValue: string | undefined,
  newState: string
): string {
  const normalizedState = normalizeGoogleOAuthState(newState);
  const existingStates = parseGoogleOAuthStateList(existingCookieValue);
  const nextStates = normalizedState
    ? [...existingStates, normalizedState]
    : existingStates;
  return serializeGoogleOAuthStateList(nextStates) ?? "";
}

export function consumeGoogleOAuthState(
  existingCookieValue: string | undefined,
  state: string | null | undefined
): {
  matched: boolean;
  nextCookieValue: string | null;
} {
  const normalizedState = state ? normalizeGoogleOAuthState(state) : "";
  const states = parseGoogleOAuthStateList(existingCookieValue);

  if (!normalizedState) {
    return {
      matched: false,
      nextCookieValue: serializeGoogleOAuthStateList(states),
    };
  }

  const stateIndex = states.lastIndexOf(normalizedState);
  if (stateIndex === -1) {
    return {
      matched: false,
      nextCookieValue: serializeGoogleOAuthStateList(states),
    };
  }

  const remainingStates = states.filter((_, index) => index !== stateIndex);
  return {
    matched: true,
    nextCookieValue: serializeGoogleOAuthStateList(remainingStates),
  };
}

function sign(value: string): string {
  return createHmac("sha256", getAuthSecret()).update(value).digest("hex");
}

function encodePayload(payload: string): string {
  return Buffer.from(payload, "utf8").toString("base64url");
}

function decodePayload(payload: string): string {
  return Buffer.from(payload, "base64url").toString("utf8");
}

export function verifyCredentials(email: string, password: string): boolean {
  const allowed = getAllowedCredentials();
  return email === allowed.email && password === allowed.password;
}

export function getGoogleOAuthCredentials():
  | { clientId: string; clientSecret: string }
  | null {
  const clientId = process.env.GOOGLE_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    return null;
  }

  return { clientId, clientSecret };
}

export function getGoogleRedirectUri(origin: string): string {
  const configured = process.env.GOOGLE_REDIRECT_URI;
  if (configured) {
    return configured;
  }

  return `${origin.replace(/\/$/, "")}/api/auth/google/callback`;
}

export function isGoogleEmailAllowed(email: string): boolean {
  const normalizedEmail = normalizeEmail(email);
  if (!normalizedEmail) {
    return false;
  }

  const explicitAllowList = getExplicitGoogleAllowedEmails();
  if (explicitAllowList.length > 0) {
    return explicitAllowList.some((entry) =>
      googleAllowListEntryMatchesEmail(entry, normalizedEmail)
    );
  }

  const fallbackEmail = process.env.AUTH_USER_EMAIL;
  if (!fallbackEmail) {
    return false;
  }

  return normalizeEmail(fallbackEmail) === normalizedEmail;
}

export function createSessionToken(email: string): string {
  const expiresAt = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const payload = JSON.stringify({ email, exp: expiresAt });
  const encoded = encodePayload(payload);
  const signature = sign(encoded);
  return `${encoded}.${signature}`;
}

export function verifySessionToken(token: string | undefined): { email: string } | null {
  if (!token) {
    return null;
  }

  const [encoded, signature] = token.split(".");
  if (!encoded || !signature) {
    return null;
  }

  const expectedSig = sign(encoded);
  const provided = Buffer.from(signature, "hex");
  const expected = Buffer.from(expectedSig, "hex");

  if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) {
    return null;
  }

  const parsed = JSON.parse(decodePayload(encoded)) as { email?: string; exp?: number };

  if (!parsed.email || !parsed.exp) {
    return null;
  }

  if (parsed.exp < Math.floor(Date.now() / 1000)) {
    return null;
  }

  return { email: parsed.email };
}

export function getRequestUser(request: NextRequest): { email: string } | null {
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value;
  return verifySessionToken(token);
}

import { createHmac, timingSafeEqual } from "node:crypto";

import type { NextRequest } from "next/server";

export const AUTH_COOKIE_NAME = "qml_session";
export const SESSION_TTL_SECONDS = 60 * 60 * 12;
export const GOOGLE_OAUTH_STATE_COOKIE_NAME = "qml_google_oauth_state";
export const GOOGLE_OAUTH_STATE_TTL_SECONDS = 60 * 10;

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

function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

function getExplicitGoogleAllowedEmails(): string[] {
  const raw = process.env.GOOGLE_AUTH_ALLOWED_EMAILS;
  if (!raw) {
    return [];
  }

  return raw
    .split(",")
    .map((value) => normalizeEmail(value))
    .filter((value) => value.length > 0);
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
    return explicitAllowList.includes(normalizedEmail);
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

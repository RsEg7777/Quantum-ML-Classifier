import { NextRequest, NextResponse } from "next/server";

import {
  AUTH_COOKIE_NAME,
  GOOGLE_OAUTH_STATE_COOKIE_NAME,
  SESSION_TTL_SECONDS,
  createSessionToken,
  getGoogleOAuthCredentials,
  getGoogleRedirectUri,
  isGoogleEmailAllowed,
} from "@/lib/server/auth";

type GoogleTokenResponse = {
  access_token?: string;
};

type GoogleUserInfoResponse = {
  email?: string;
  email_verified?: boolean;
};

function buildLoginRedirect(request: NextRequest, errorCode: string): NextResponse {
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("error", errorCode);

  const response = NextResponse.redirect(loginUrl);
  response.cookies.set({
    name: GOOGLE_OAUTH_STATE_COOKIE_NAME,
    value: "",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });

  return response;
}

export async function GET(request: NextRequest) {
  const callbackError = request.nextUrl.searchParams.get("error");
  if (callbackError) {
    return buildLoginRedirect(request, "google_access_denied");
  }

  const code = request.nextUrl.searchParams.get("code");
  const state = request.nextUrl.searchParams.get("state");
  const expectedState = request.cookies.get(GOOGLE_OAUTH_STATE_COOKIE_NAME)?.value;

  if (!code || !state || !expectedState || state !== expectedState) {
    return buildLoginRedirect(request, "google_state_mismatch");
  }

  const credentials = getGoogleOAuthCredentials();
  if (!credentials) {
    return buildLoginRedirect(request, "google_not_configured");
  }

  const redirectUri = getGoogleRedirectUri(request.nextUrl.origin);
  const exchangeBody = new URLSearchParams({
    code,
    client_id: credentials.clientId,
    client_secret: credentials.clientSecret,
    redirect_uri: redirectUri,
    grant_type: "authorization_code",
  });

  const tokenResponse = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: exchangeBody,
    cache: "no-store",
  });

  if (!tokenResponse.ok) {
    return buildLoginRedirect(request, "google_token_exchange_failed");
  }

  const tokenPayload = (await tokenResponse.json()) as GoogleTokenResponse;
  if (!tokenPayload.access_token) {
    return buildLoginRedirect(request, "google_token_missing");
  }

  const userInfoResponse = await fetch("https://openidconnect.googleapis.com/v1/userinfo", {
    headers: {
      Authorization: `Bearer ${tokenPayload.access_token}`,
    },
    cache: "no-store",
  });

  if (!userInfoResponse.ok) {
    return buildLoginRedirect(request, "google_profile_failed");
  }

  const userInfo = (await userInfoResponse.json()) as GoogleUserInfoResponse;
  if (!userInfo.email || userInfo.email_verified !== true) {
    return buildLoginRedirect(request, "google_email_unverified");
  }

  if (!isGoogleEmailAllowed(userInfo.email)) {
    return buildLoginRedirect(request, "google_email_not_allowed");
  }

  const token = createSessionToken(userInfo.email);
  const dashboardUrl = new URL("/dashboard", request.url);
  const response = NextResponse.redirect(dashboardUrl);

  response.cookies.set({
    name: AUTH_COOKIE_NAME,
    value: token,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_TTL_SECONDS,
  });

  response.cookies.set({
    name: GOOGLE_OAUTH_STATE_COOKIE_NAME,
    value: "",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
  });

  return response;
}

import { NextRequest, NextResponse } from "next/server";

import {
  AUTH_COOKIE_NAME,
  GOOGLE_OAUTH_STATE_COOKIE_NAME,
  GOOGLE_OAUTH_STATE_TTL_SECONDS,
  SESSION_TTL_SECONDS,
  consumeGoogleOAuthState,
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

function setGoogleStateCookie(response: NextResponse, nextCookieValue: string | null): void {
  if (!nextCookieValue) {
    response.cookies.set({
      name: GOOGLE_OAUTH_STATE_COOKIE_NAME,
      value: "",
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 0,
    });
    return;
  }

  response.cookies.set({
    name: GOOGLE_OAUTH_STATE_COOKIE_NAME,
    value: nextCookieValue,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: GOOGLE_OAUTH_STATE_TTL_SECONDS,
  });
}

function buildLoginRedirect(
  request: NextRequest,
  errorCode: string,
  nextStateCookieValue: string | null = null
): NextResponse {
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("error", errorCode);

  const response = NextResponse.redirect(loginUrl);
  setGoogleStateCookie(response, nextStateCookieValue);

  return response;
}

export async function GET(request: NextRequest) {
  const state = request.nextUrl.searchParams.get("state");
  const existingStateCookie = request.cookies.get(GOOGLE_OAUTH_STATE_COOKIE_NAME)?.value;
  const stateValidation = consumeGoogleOAuthState(existingStateCookie, state);

  const callbackError = request.nextUrl.searchParams.get("error");
  if (callbackError) {
    return buildLoginRedirect(
      request,
      "google_access_denied",
      stateValidation.nextCookieValue
    );
  }

  const code = request.nextUrl.searchParams.get("code");

  if (!code || !state || !stateValidation.matched) {
    return buildLoginRedirect(
      request,
      "google_state_mismatch",
      stateValidation.nextCookieValue
    );
  }

  const credentials = getGoogleOAuthCredentials();
  if (!credentials) {
    return buildLoginRedirect(
      request,
      "google_not_configured",
      stateValidation.nextCookieValue
    );
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
    return buildLoginRedirect(
      request,
      "google_token_exchange_failed",
      stateValidation.nextCookieValue
    );
  }

  const tokenPayload = (await tokenResponse.json()) as GoogleTokenResponse;
  if (!tokenPayload.access_token) {
    return buildLoginRedirect(
      request,
      "google_token_missing",
      stateValidation.nextCookieValue
    );
  }

  const userInfoResponse = await fetch("https://openidconnect.googleapis.com/v1/userinfo", {
    headers: {
      Authorization: `Bearer ${tokenPayload.access_token}`,
    },
    cache: "no-store",
  });

  if (!userInfoResponse.ok) {
    return buildLoginRedirect(
      request,
      "google_profile_failed",
      stateValidation.nextCookieValue
    );
  }

  const userInfo = (await userInfoResponse.json()) as GoogleUserInfoResponse;
  if (!userInfo.email || userInfo.email_verified !== true) {
    return buildLoginRedirect(
      request,
      "google_email_unverified",
      stateValidation.nextCookieValue
    );
  }

  if (!isGoogleEmailAllowed(userInfo.email)) {
    return buildLoginRedirect(
      request,
      "google_email_not_allowed",
      stateValidation.nextCookieValue
    );
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

  setGoogleStateCookie(response, stateValidation.nextCookieValue);

  return response;
}

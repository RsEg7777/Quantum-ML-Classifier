import { randomBytes } from "node:crypto";

import { NextRequest, NextResponse } from "next/server";

import {
  GOOGLE_OAUTH_STATE_COOKIE_NAME,
  GOOGLE_OAUTH_STATE_TTL_SECONDS,
  getGoogleOAuthCredentials,
  getGoogleRedirectUri,
} from "@/lib/server/auth";

function redirectToLogin(request: NextRequest, errorCode: string): NextResponse {
  const loginUrl = new URL("/login", request.url);
  loginUrl.searchParams.set("error", errorCode);
  return NextResponse.redirect(loginUrl);
}

export async function GET(request: NextRequest) {
  const credentials = getGoogleOAuthCredentials();
  if (!credentials) {
    return redirectToLogin(request, "google_not_configured");
  }

  const oauthState = randomBytes(24).toString("base64url");
  const redirectUri = getGoogleRedirectUri(request.nextUrl.origin);

  const googleAuthUrl = new URL("https://accounts.google.com/o/oauth2/v2/auth");
  googleAuthUrl.searchParams.set("client_id", credentials.clientId);
  googleAuthUrl.searchParams.set("redirect_uri", redirectUri);
  googleAuthUrl.searchParams.set("response_type", "code");
  googleAuthUrl.searchParams.set("scope", "openid email profile");
  googleAuthUrl.searchParams.set("state", oauthState);
  googleAuthUrl.searchParams.set("prompt", "select_account");
  googleAuthUrl.searchParams.set("include_granted_scopes", "true");

  const response = NextResponse.redirect(googleAuthUrl);
  response.cookies.set({
    name: GOOGLE_OAUTH_STATE_COOKIE_NAME,
    value: oauthState,
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: GOOGLE_OAUTH_STATE_TTL_SECONDS,
  });

  return response;
}

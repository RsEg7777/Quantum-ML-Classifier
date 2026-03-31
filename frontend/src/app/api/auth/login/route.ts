import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";

import { AUTH_COOKIE_NAME, createSessionToken, verifyCredentials } from "@/lib/server/auth";

const LoginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

export async function POST(request: NextRequest) {
  try {
    const payload = LoginSchema.parse(await request.json());

    if (!verifyCredentials(payload.email, payload.password)) {
      return NextResponse.json({ error: "Invalid credentials" }, { status: 401 });
    }

    const token = createSessionToken(payload.email);
    const response = NextResponse.json({ authenticated: true, email: payload.email });

    response.cookies.set({
      name: AUTH_COOKIE_NAME,
      value: token,
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 12,
    });

    return response;
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }
}

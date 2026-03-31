import { NextRequest, NextResponse } from "next/server";

import { getRequestUser } from "@/lib/server/auth";

export async function GET(request: NextRequest) {
  const user = getRequestUser(request);
  if (!user) {
    return NextResponse.json({ authenticated: false });
  }

  return NextResponse.json({
    authenticated: true,
    email: user.email,
  });
}

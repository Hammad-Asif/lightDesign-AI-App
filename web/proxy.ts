import { NextResponse, type NextRequest } from "next/server";

// Optimistic check only: no cookie means straight to sign-in. The backend verifies the cookie on
// every API call, and the pages send you back here when it answers 401.
const COOKIE = "roofline_session";
const PUBLIC = ["/login", "/signup"];

export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  const signedIn = request.cookies.has(COOKIE);
  if (!signedIn && !PUBLIC.includes(pathname)) {
    const url = new URL("/login", request.url);
    if (pathname !== "/") url.searchParams.set("next", pathname + search);
    return NextResponse.redirect(url);
  }
  return NextResponse.next();
}

export const config = {
  // pages only: /api is proxied straight to the backend (large uploads never pass through here)
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"],
};

"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";
import { api, post } from "@/lib/api";
import type { User } from "@/lib/types";

export default function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const params = useSearchParams();
  const next = safeNext(params.get("next"));
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [signupAllowed, setSignupAllowed] = useState(true);

  useEffect(() => {
    // already signed in (a valid cookie): skip the form
    api<{ user: User | null; signup_allowed: boolean }>("/api/auth/me")
      .then((r) => { setSignupAllowed(r.signup_allowed); if (r.user) router.replace(next); })
      .catch(() => {});
  }, [router, next]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true); setError("");
    try {
      await post(mode === "login" ? "/api/auth/login" : "/api/auth/signup", mode === "login" ? { email, password } : { email, name, password });
      router.replace(next);
      router.refresh();
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  const isLogin = mode === "login";
  return (
    <main className="min-h-screen grid place-items-center px-4 py-10">
      <div className="w-full max-w-sm">
        <div className="mb-6">
          <p className="text-lg font-semibold tracking-tight">Roofline</p>
          <p className="text-sm text-muted">Roof lines from one photo of a house.</p>
        </div>
        <form onSubmit={submit} className="rounded-lg border border-rule bg-panel p-6 space-y-4">
          <h1 className="text-base font-semibold">{isLogin ? "Sign in" : "Create an account"}</h1>
          {!isLogin && (
            <label className="block text-sm space-y-1">
              <span>Name</span>
              <input className="field w-full" value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" />
            </label>
          )}
          <label className="block text-sm space-y-1">
            <span>Email</span>
            <input className="field w-full" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" autoFocus />
          </label>
          <label className="block text-sm space-y-1">
            <span>Password</span>
            <input className="field w-full" type="password" required minLength={isLogin ? 1 : 8} value={password} onChange={(e) => setPassword(e.target.value)}
              autoComplete={isLogin ? "current-password" : "new-password"} />
            {!isLogin && <span className="block text-xs text-muted">At least 8 characters.</span>}
          </label>
          {error && <p role="alert" className="text-sm rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2">{error}</p>}
          <button type="submit" disabled={busy}
            className="w-full rounded-md bg-ink text-panel py-2 text-sm font-medium hover:bg-black disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-draft">
            {busy ? (isLogin ? "Signing in…" : "Creating account…") : isLogin ? "Sign in" : "Create account"}
          </button>
        </form>
        <p className="mt-4 text-sm text-muted text-center">
          {isLogin
            ? signupAllowed && <>No account yet? <Link className="text-draft font-medium hover:underline" href={`/signup${next !== "/" ? `?next=${encodeURIComponent(next)}` : ""}`}>Create one</Link></>
            : <>Already have an account? <Link className="text-draft font-medium hover:underline" href={`/login${next !== "/" ? `?next=${encodeURIComponent(next)}` : ""}`}>Sign in</Link></>}
        </p>
      </div>
    </main>
  );
}

/** Only same-site paths, so ?next= cannot bounce a user to another site after sign-in. */
function safeNext(value: string | null): string {
  return value && value.startsWith("/") && !value.startsWith("//") ? value : "/";
}

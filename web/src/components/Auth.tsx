import { useState, type FormEvent } from "react";

import * as api from "../api";
import type { User } from "../types";

interface Props {
  onSignedIn: (user: User) => void;
}

export function Auth({ onSignedIn }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const registering = mode === "register";

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(
        registering
          ? await api.register({ email, name, password })
          : await api.login({ email, password }),
      );
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="panel centered" onSubmit={submit}>
      <h1>{registering ? "Create your account" : "Sign in"}</h1>
      <p className="muted">
        {registering
          ? "Set up an account to start creating lighting mock-ups."
          : "Sign in to your lighting mock-ups."}
      </p>

      {registering && (
        <label className="field">
          <span>Name</span>
          <input value={name} autoComplete="name" required onChange={(event) => setName(event.target.value)} />
        </label>
      )}

      <label className="field">
        <span>Email</span>
        <input
          type="email"
          value={email}
          autoComplete="email"
          required
          onChange={(event) => setEmail(event.target.value)}
        />
      </label>

      <label className="field">
        <span>Password</span>
        <input
          type="password"
          value={password}
          autoComplete={registering ? "new-password" : "current-password"}
          minLength={registering ? 8 : undefined}
          required
          onChange={(event) => setPassword(event.target.value)}
        />
      </label>
      {registering && <p className="muted small">At least 8 characters.</p>}

      {error && <p className="form-error">{error}</p>}

      <button className="primary" type="submit" disabled={busy}>
        {busy ? "Please wait…" : registering ? "Create account" : "Sign in"}
      </button>
      <button
        type="button"
        className="link"
        onClick={() => {
          setMode(registering ? "login" : "register");
          setError(null);
        }}
      >
        {registering ? "Already have an account? Sign in" : "New here? Create an account"}
      </button>
    </form>
  );
}

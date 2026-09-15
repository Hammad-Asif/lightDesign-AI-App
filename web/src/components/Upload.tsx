import { useState } from "react";

interface Props {
  busy: boolean;
  onSubmit: (file: File, prompt: string) => void;
}

export function Upload({ busy, onSubmit }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");

  return (
    <div className="panel centered">
      <h1>New mock-up</h1>
      <p className="muted">Upload the customer photo and describe the lighting.</p>

      <label className="dropzone">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <span>{file ? file.name : "Choose a photo (JPG, PNG or WEBP)"}</span>
      </label>

      <textarea
        rows={4}
        value={prompt}
        placeholder="e.g. C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges"
        onChange={(event) => setPrompt(event.target.value)}
      />
      <p className="muted small">
        The instruction is stored now and interpreted in Milestone 2.
      </p>

      <button className="primary" disabled={!file || busy} onClick={() => file && onSubmit(file, prompt)}>
        {busy ? "Uploading…" : "Continue"}
      </button>
    </div>
  );
}

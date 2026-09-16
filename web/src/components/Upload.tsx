import { useState } from "react";

interface Props {
  busy: boolean;
  onSubmit: (file: File, prompt: string) => void;
}

export function Upload({ busy, onSubmit }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");

  return (
    <div className="panel">
      <h1>New mock-up</h1>
      <p className="lede">Start with a daytime photo of the house, taken before any lights go up.</p>

      <label className="dropzone">
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <span>{file ? file.name : "Choose a photo (JPG, PNG or WEBP)"}</span>
      </label>

      <label className="field">
        <span>Where do the lights go?</span>
        <textarea
          rows={4}
          value={prompt}
          placeholder="C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges"
          onChange={(event) => setPrompt(event.target.value)}
        />
        <span>Only the areas you mention get masked. You can change any of them afterwards.</span>
      </label>

      <button className="primary" disabled={!file || busy} onClick={() => file && onSubmit(file, prompt)}>
        {busy ? "Uploading…" : "Continue"}
      </button>
    </div>
  );
}

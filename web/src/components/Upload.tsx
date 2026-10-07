import { useState } from "react";

interface Props {
  busy: boolean;
  onSubmit: (file: File, prompt: string, name: string) => void;
  // Back to the projects list without uploading.
  onCancel?: () => void;
}

export function Upload({ busy, onSubmit, onCancel }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [name, setName] = useState("");

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
        <span>Project name</span>
        <input
          value={name}
          maxLength={200}
          placeholder="The customer or the address, e.g. 14 Oak Lane"
          onChange={(event) => setName(event.target.value)}
        />
        <span>How this mock-up is found again on your projects list. Optional.</span>
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

      <button className="primary" disabled={!file || busy} onClick={() => file && onSubmit(file, prompt, name.trim())}>
        {busy ? "Uploading…" : "Continue"}
      </button>
      {onCancel && (
        <button className="link" disabled={busy} onClick={onCancel}>
          Back to your projects
        </button>
      )}
    </div>
  );
}

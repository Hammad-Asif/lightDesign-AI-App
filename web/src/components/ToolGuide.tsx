import { useEffect, type ReactNode } from "react";

import type { Tool } from "../types";

// A short how-to card per tool: what it is for, the gesture shown on a plain house outline, and
// the steps. Shown the first time a tool is picked (remembered per browser) and from the dock's
// "?" any time. Instructional only: line drawings in the app's own colours, nothing decorative.

const SEEN_KEY = "ld-tool-guides-seen";
const LOOP = "3.2s";

export function guideSeen(tool: Tool): boolean {
  try {
    return (JSON.parse(localStorage.getItem(SEEN_KEY) ?? "[]") as string[]).includes(tool);
  } catch {
    return true; // no storage: never pop up uninvited
  }
}

export function markGuideSeen(tool: Tool): void {
  try {
    const seen = new Set(JSON.parse(localStorage.getItem(SEEN_KEY) ?? "[]") as string[]);
    seen.add(tool);
    localStorage.setItem(SEEN_KEY, JSON.stringify([...seen]));
  } catch {
    // storage unavailable: the guide simply shows again next time
  }
}

const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

// An animation that is left out when motion is reduced: every element is drawn in its finished
// state, and the animation only plays the way there.
function Animate(props: Record<string, string>) {
  if (still) return null;
  return <animate dur={LOOP} repeatCount="indefinite" {...props} />;
}

function Move({ path }: { path: string }) {
  if (still) return null;
  return <animateTransform attributeName="transform" type="translate" dur={LOOP} repeatCount="indefinite" values={path} />;
}

// The pointer, its tip at the group's origin.
function Cursor({ path, at }: { path: string; at: string }) {
  return (
    <g transform={`translate(${at})`}>
      <Move path={path} />
      <path d="M0 0 L0 15 L4 11 L7 18 L10 17 L7 10 L12 10 Z" fill="var(--text)" stroke="var(--surface)" strokeWidth="1.2" />
    </g>
  );
}

// The same plain house in every scene.
function House({ children }: { children?: ReactNode }) {
  return (
    <svg viewBox="0 0 360 200" className="guide-scene" aria-hidden="true">
      <g fill="var(--surface-2)" stroke="var(--muted)" strokeWidth="1.6" strokeLinejoin="round">
        <rect x="70" y="95" width="220" height="93" />
        <polygon points="55,95 180,35 305,95" />
        <rect x="95" y="115" width="50" height="36" />
        <path d="M120 115 V151 M95 133 H145" fill="none" />
        <rect x="165" y="130" width="30" height="58" />
        <rect x="215" y="135" width="60" height="53" />
        <path d="M215 148 H275 M215 161 H275 M215 174 H275" fill="none" />
        <rect x="30" y="166" width="72" height="22" rx="10" />
      </g>
      <line x1="10" y1="188" x2="350" y2="188" stroke="var(--muted)" strokeWidth="1.6" />
      {children}
    </svg>
  );
}

const ACCENT = "var(--accent)";
const AMBER = "#f59e0b";

interface Guide {
  title: string;
  line: string;
  steps: string[];
  scene: ReactNode;
}

const GUIDES: Partial<Record<Tool, Guide>> = {
  select: {
    title: "Select",
    line: "Pick a line or an area to see and change it.",
    steps: ["Click a line or an area on the photo.", "Its details open in the list on the right.", "Press S to come back to this tool."],
    scene: (
      <House>
        <line x1="55" y1="95" x2="305" y2="95" stroke={ACCENT} strokeWidth="4" opacity="1">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.45;0.5;1" />
        </line>
        <Cursor at="150,95" path="300,170; 150,95; 150,95" />
      </House>
    ),
  },
  brush: {
    title: "Brush",
    line: "Paint more of the selected area.",
    steps: ["Select an area first, such as a hedge.", "Drag over the photo to add to it.", "Change the brush size under the tools, or with [ and ]."],
    scene: (
      <House>
        <path d="M34 177 H100" stroke={ACCENT} strokeWidth="18" strokeLinecap="round" opacity="0.55"
              strokeDasharray="70" strokeDashoffset="0">
          <Animate attributeName="stroke-dashoffset" values="70;70;0;0" keyTimes="0;0.2;0.75;1" />
        </path>
        <Cursor at="100,177" path="34,177; 34,177; 100,177; 100,177" />
      </House>
    ),
  },
  erase: {
    title: "Erase",
    line: "Take away part of the selected area.",
    steps: ["Select an area first.", "Drag over the part that should not be there.", "Change the size under the tools, or with [ and ]."],
    scene: (
      <House>
        <rect x="30" y="166" width="72" height="22" rx="10" fill={ACCENT} opacity="0.55" />
        <path d="M60 177 H100" stroke="var(--surface-2)" strokeWidth="20" strokeLinecap="round"
              strokeDasharray="40" strokeDashoffset="0">
          <Animate attributeName="stroke-dashoffset" values="40;40;0;0" keyTimes="0;0.2;0.75;1" />
        </path>
        <Cursor at="100,177" path="60,177; 60,177; 100,177; 100,177" />
      </House>
    ),
  },
  vertex: {
    title: "Points",
    line: "Reshape a line by moving its points.",
    steps: ["Select a line, then drag one of its points.", "Click on the line to add a point there.", "Alt-click a point to remove it."],
    scene: (
      <House>
        <line x1="55" y1="95" x2="180" y2="35" stroke={ACCENT} strokeWidth="3.5">
          <Animate attributeName="y2" values="52;52;35;35" keyTimes="0;0.2;0.7;1" />
        </line>
        <circle cx="180" cy="35" r="5" fill="#fff" stroke={ACCENT} strokeWidth="2">
          <Animate attributeName="cy" values="52;52;35;35" keyTimes="0;0.2;0.7;1" />
        </circle>
        <Cursor at="180,35" path="180,52; 180,52; 180,35; 180,35" />
      </House>
    ),
  },
  move: {
    title: "Move / extend",
    line: "Slide a whole line, or lengthen it along its own direction.",
    steps: ["Select a line.", "Drag the line itself to move it.", "Drag an end to make it longer or shorter; it keeps its angle."],
    scene: (
      <House>
        <line x1="70" y1="95" x2="290" y2="95" stroke={ACCENT} strokeWidth="4">
          <Animate attributeName="x2" values="230;230;290;290" keyTimes="0;0.2;0.7;1" />
        </line>
        <Cursor at="290,95" path="230,95; 230,95; 290,95; 290,95" />
      </House>
    ),
  },
  box: {
    title: "Box",
    line: "Find an object by drawing a box around it.",
    steps: ["Drag a box around the object, such as a hedge.", "It is found and outlined inside the box.", "Correct the result with Brush and Erase."],
    scene: (
      <House>
        <rect x="24" y="158" width="84" height="36" fill="none" stroke={ACCENT} strokeWidth="2" strokeDasharray="6 4">
          <Animate attributeName="width" values="0;84;84;84" keyTimes="0;0.45;0.5;1" />
          <Animate attributeName="height" values="0;36;36;36" keyTimes="0;0.45;0.5;1" />
        </rect>
        <rect x="30" y="166" width="72" height="22" rx="10" fill={ACCENT} opacity="0.45">
          <Animate attributeName="opacity" values="0;0;0.45;0.45" keyTimes="0;0.5;0.6;1" />
        </rect>
        <Cursor at="108,194" path="24,158; 108,194; 108,194; 108,194" />
      </House>
    ),
  },
  click: {
    title: "Click object",
    line: "Click an object and it is outlined for you.",
    steps: ["Click once on the object, such as a hedge.", "It is found and outlined.", "Alt-click an outline to remove it."],
    scene: (
      <House>
        <rect x="30" y="166" width="72" height="22" rx="10" fill={ACCENT} opacity="0.45" stroke={ACCENT} strokeWidth="2">
          <Animate attributeName="opacity" values="0;0;0.45;0.45" keyTimes="0;0.45;0.55;1" />
        </rect>
        <Cursor at="66,177" path="200,60; 66,177; 66,177; 66,177" />
      </House>
    ),
  },
  find: {
    title: "Find in box",
    line: "Find roof lines that detection missed.",
    steps: ["Drag a box where a roof edge has no line.", "The lines inside the box are found and added.", "Nothing already on the photo changes."],
    scene: (
      <House>
        <rect x="170" y="30" width="140" height="72" fill="none" stroke="#22d3ee" strokeWidth="2" strokeDasharray="6 4">
          <Animate attributeName="width" values="0;140;140;140" keyTimes="0;0.4;0.45;1" />
          <Animate attributeName="height" values="0;72;72;72" keyTimes="0;0.4;0.45;1" />
        </rect>
        <line x1="180" y1="35" x2="305" y2="95" stroke={ACCENT} strokeWidth="4">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.5;0.6;1" />
        </line>
        <Cursor at="310,102" path="170,30; 310,102; 310,102; 310,102" />
      </House>
    ),
  },
  measure: {
    title: "Measure",
    line: "Give the photo real scale with one length you know.",
    steps: ["Tap both ends of something you know the size of, such as a 16 ft garage door. Or pick an eave.",
            "Type its real length and save.", "Every line on the photo then shows its real length."],
    scene: (
      <House>
        <line x1="215" y1="130" x2="275" y2="130" stroke={AMBER} strokeWidth="3">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.55;0.6;1" />
        </line>
        <circle cx="215" cy="130" r="4.5" fill="#fff" stroke={AMBER} strokeWidth="2">
          <Animate attributeName="opacity" values="0;1;1;1" keyTimes="0;0.25;0.3;1" />
        </circle>
        <circle cx="275" cy="130" r="4.5" fill="#fff" stroke={AMBER} strokeWidth="2">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.55;0.6;1" />
        </circle>
        <g>
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.65;0.72;1" />
          <rect x="226" y="108" width="38" height="16" rx="8" fill={AMBER} />
          <text x="245" y="119.5" textAnchor="middle" fontSize="10" fontWeight="600" fill="#1f2937">16 ft</text>
        </g>
        <Cursor at="275,130" path="320,80; 215,130; 215,130; 275,130; 275,130" />
      </House>
    ),
  },
  wreath: {
    title: "Wreath",
    line: "Place a wreath where it goes, at its real size.",
    steps: ["Measure the house first, so the wreath can be sized.", "Tap where the wreath hangs, such as the front door.", "Change its size (24, 36 or 48 in) in the list."],
    scene: (
      <House>
        <circle cx="180" cy="150" r="10" fill="none" stroke="#2f6b3a" strokeWidth="5">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.45;0.52;1" />
        </circle>
        <Cursor at="180,150" path="300,60; 180,150; 180,150; 180,150" />
      </House>
    ),
  },
  house: {
    title: "House",
    line: "Choose which roofs belong to the house being quoted.",
    steps: ["Tap a roof to add it to the house, or tap again to take it out.", "Green roofs are on the house; red ones are left out.", "The roof lines are redrawn to match."],
    scene: (
      <House>
        <polygon points="55,95 180,35 305,95" fill="rgba(57, 255, 20, 0.25)" stroke="#2f9e44" strokeWidth="2">
          <Animate attributeName="opacity" values="0;0;1;1" keyTimes="0;0.45;0.52;1" />
        </polygon>
        <Cursor at="180,70" path="320,170; 180,70; 180,70; 180,70" />
      </House>
    ),
  },
};

export function hasGuide(tool: Tool): boolean {
  return Boolean(GUIDES[tool]);
}

export function ToolGuide({ tool, onClose }: { tool: Tool; onClose: () => void }) {
  const guide = GUIDES[tool];
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  if (!guide) return null;
  return (
    <div className="guide-backdrop" onClick={onClose}>
      <div className="guide-card" role="dialog" aria-modal="true" aria-labelledby="guide-title" onClick={(event) => event.stopPropagation()}>
        <h2 id="guide-title">{guide.title}</h2>
        <p className="muted">{guide.line}</p>
        <div className="guide-stage">{guide.scene}</div>
        <ol className="guide-steps">
          {guide.steps.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <button className="primary" autoFocus onClick={onClose}>
          Got it
        </button>
      </div>
    </div>
  );
}

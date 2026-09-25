import { Suspense } from "react";
import DesignEditor from "@/components/DesignEditor";

export const metadata = { title: "Editor · Roofline" };

export default function Page() {
  return (
    <Suspense>
      <DesignEditor />
    </Suspense>
  );
}

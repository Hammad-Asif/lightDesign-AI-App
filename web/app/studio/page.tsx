import { redirect } from "next/navigation";

// The detection studio is hidden from the UI (owner's request, 2026-09-26): lines are detected and edited in the
// mock-up editor (/new, /design). The studio's code is kept in components/Studio.tsx; to bring it back, restore:
//
//   import { Suspense } from "react";
//   import Studio from "@/components/Studio";
//   export const metadata = { title: "Studio · Roofline" };
//   export default function Page() {
//     return (
//       <Suspense>
//         <Studio />
//       </Suspense>
//     );
//   }

export default function Page() {
  redirect("/");
}

import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./AppShell";
import { HistoryPage } from "../pages/HistoryPage";
import { WorkspacePage } from "../pages/WorkspacePage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<HistoryPage />} />
        <Route path="workspace" element={<WorkspacePage />} />
        <Route path="history" element={<Navigate to="/" replace />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

import { type Dispatch, type ReactNode, type SetStateAction, useState } from "react";
import { Link, Outlet } from "react-router-dom";

export type AppShellHeaderContent = {
  title: ReactNode;
  actions: ReactNode;
};

export type AppShellOutletContext = {
  setHeaderContent: Dispatch<SetStateAction<AppShellHeaderContent>>;
};

export const EMPTY_APP_SHELL_HEADER_CONTENT: AppShellHeaderContent = {
  title: null,
  actions: null
};

export function AppShell() {
  const [headerContent, setHeaderContent] = useState<AppShellHeaderContent>(EMPTY_APP_SHELL_HEADER_CONTENT);

  return (
    <div className="app-shell">
      <header className="global-header">
        <div className="global-brand">
          <Link className="brand-mark" to="/">
            AutoVisionLab
          </Link>
        </div>
        <div className="global-header-title">{headerContent.title}</div>
        <div className="global-header-actions">{headerContent.actions}</div>
      </header>
      <main className="app-main">
        <Outlet context={{ setHeaderContent }} />
      </main>
    </div>
  );
}

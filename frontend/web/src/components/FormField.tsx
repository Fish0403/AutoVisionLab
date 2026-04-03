import type { ReactNode } from "react";

interface FormFieldProps {
  label: string;
  hint?: string;
  labelAccessory?: ReactNode;
  children: ReactNode;
}

export function FormField({ label, hint, labelAccessory, children }: FormFieldProps) {
  return (
    <div className="form-field">
      <span className="field-label-row">
        <span className="field-label">{label}</span>
        {labelAccessory}
      </span>
      {children}
      {hint ? <span className="field-hint">{hint}</span> : null}
    </div>
  );
}

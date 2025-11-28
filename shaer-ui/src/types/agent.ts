export interface AgentStep {
  step: number;
  agent: string;
  tool: string;
  summary: string;
}

export interface LibraryItem {
  poem_id: string;
  title: string;
  poet_name?: string;
  poem_meter?: string;
  poem_era?: string;
  poem_theme?: string;
}
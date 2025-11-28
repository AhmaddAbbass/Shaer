import type { PoemSpec, PoemVersion } from './poem';
import type { AgentStep, LibraryItem } from './agent';

export interface ChatResponse {
  reply: string;
  mode?: 'generate' | 'fix' | 'search' | 'explain' | 'other';
  poem_spec?: PoemSpec | null;
  poem_version?: PoemVersion | null;
  agent_trace?: AgentStep[];
  library_context?: LibraryItem[];
}
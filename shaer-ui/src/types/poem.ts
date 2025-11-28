export interface PoemSpec {
  poem_title?: string;
  poem_meter?: string;
  poem_theme?: string;
  poem_era?: string;
  poet_name?: string;
  poem_description?: string;
  num_verses?: number;
}

export interface PoemVersion {
  spec: PoemSpec;
  verses: string[];
}
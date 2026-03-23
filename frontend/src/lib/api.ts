const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

export interface ZoningRegulation {
  id: number;
  municipality: string;
  province: string;
  zone_code: string;
  zone_name: string;
  zone_category: string;
  min_lot_size_sqm: number | null;
  max_building_height_m: number | null;
  max_stories: number | null;
  max_lot_coverage_pct: number | null;
  min_front_setback_m: number | null;
  min_rear_setback_m: number | null;
  min_side_setback_m: number | null;
  parking_spaces_per_unit: number | null;
  permitted_dwelling_types: string[];
  max_units_per_lot: number | null;
  density_units_per_hectare: number | null;
  restrictiveness_score: number | null;
  confidence_score: number;
  needs_review: boolean;
  source_url: string;
  source_document: string;
  extraction_date: string;
  additional_regulations: Record<string, unknown> | null;
  raw_text_excerpt: string;
}

export interface Municipality {
  id: number;
  name: string;
  province: string;
  population: number | null;
  latitude: number;
  longitude: number;
  data_sources: string[];
  last_updated: string;
}

export interface ZoneStats {
  municipality: string;
  zone_count: number;
  avg_restrictiveness: number | null;
  needs_review_count: number;
}

export interface ChatSource {
  municipality: string;
  zone_code?: string;
  source_url?: string;
  source_document?: string;
  type?: string;
  sql?: string;
}

export interface ChatResponse {
  answer: string;
  sources: ChatSource[];
}

export type ChatStreamEvent =
  | { type: "tool_start"; name: string; description: string; detail?: string }
  | { type: "tool_end"; name: string; summary: string; preview?: string }
  | { type: "token"; content: string }
  | { type: "done"; sources: ChatSource[] }
  | { type: "error"; message: string };

export interface PipelineEvent {
  step: string;
  timestamp: string;
  message?: string;
  detail?: string;
  [key: string]: unknown;
}

export interface PipelineJob {
  job_id: string;
  municipality: string;
  province: string;
  status: "running" | "completed" | "failed";
  current_step: string;
  step_data: Record<string, unknown>;
  started_at: string;
  completed_at: string | null;
  regulations_found: number;
  policies_found: number;
  error: string | null;
  events: PipelineEvent[];
}

export interface ScrapeLog {
  id: number;
  municipality: string;
  url: string;
  status: string;
  documents_found: number;
  error_message: string | null;
  scraped_at: string;
}

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getMunicipalities: () => fetchJSON<Municipality[]>("/municipalities"),

  getZones: (params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return fetchJSON<ZoningRegulation[]>(`/zones${qs}`);
  },

  getZoneStats: () => fetchJSON<ZoneStats[]>("/zones/stats"),

  getReviewQueue: () => fetchJSON<ZoningRegulation[]>("/review"),

  chat: async (
    question: string,
    municipality?: string,
    history?: { role: string; content: string }[],
  ): Promise<ChatResponse> => {
    const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, municipality, history: history || [] }),
    });
    if (!res.ok) throw new Error(`Chat error: ${res.status}`);
    return res.json();
  },

  chatStream: async function* (
    question: string,
    municipality?: string,
    history?: { role: string; content: string }[],
  ): AsyncGenerator<ChatStreamEvent> {
    const res = await fetch(`${API_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, municipality, history: history || [] }),
    });
    if (!res.ok) throw new Error(`Chat stream error: ${res.status}`);

    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith("data: ")) {
          try {
            const event = JSON.parse(trimmed.slice(6)) as ChatStreamEvent;
            yield event;
          } catch {
            // skip malformed lines
          }
        }
      }
    }
  },

  getExportUrl: (format: string) => `${API_URL}/export?format=${format}`,

  runPipeline: async (municipality: string, province: string = "Ontario") => {
    const res = await fetch(`${API_URL}/pipeline/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ municipality, province }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: "Pipeline error" }));
      throw new Error(data.detail || `Pipeline error: ${res.status}`);
    }
    return res.json();
  },

  getPipelineStatus: () => fetchJSON<PipelineJob[]>("/pipeline/status"),

  getPipelineJobStatus: (jobId: string) =>
    fetchJSON<PipelineJob>(`/pipeline/status/${jobId}`),

  getPipelineLogs: () => fetchJSON<ScrapeLog[]>("/pipeline/logs"),

  deleteMunicipality: async (name: string) => {
    const res = await fetch(`${API_URL}/municipalities/${encodeURIComponent(name)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: "Delete error" }));
      throw new Error(data.detail || `Delete error: ${res.status}`);
    }
    return res.json();
  },
};

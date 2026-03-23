"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import Link from "next/link";
import { api, ZoneStats, Municipality, PipelineJob, PipelineEvent, ScrapeLog } from "@/lib/api";
import {
  Building2,
  Layers,
  TrendingUp,
  Sparkles,
  Send,
  Plus,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Activity,
  Globe,
  FileText,
  Database,
  Brain,
  MapPin,
  ChevronDown,
  ChevronUp,
  Search,
  Trash2,
  AlertTriangle,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useChatContext } from "@/components/AppShell";

const SUGGESTED_QUESTIONS = [
  "Can I build a laneway house? What are the rules?",
  "Which zones have the tallest building height limits?",
  "Are Airbnb-style short-term rentals allowed?",
  "What does the official plan say about affordable housing?",
  "Compare density limits across residential zones",
];

const PROVINCES = [
  "Ontario", "British Columbia", "Alberta", "Quebec",
  "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
];

const STEP_META: Record<string, { label: string; icon: typeof Search; color: string }> = {
  queued:                     { label: "Queued",                      icon: Clock,        color: "text-muted-foreground" },
  started:                    { label: "Starting pipeline",           icon: Activity,     color: "text-blue-500" },
  discovering:                { label: "Searching for sources",       icon: Search,       color: "text-blue-500" },
  scraping:                   { label: "Scraping pages",              icon: Globe,        color: "text-blue-500" },
  scraping_done:              { label: "Scraping complete",           icon: CheckCircle2, color: "text-green-500" },
  extracting:                 { label: "Extracting regulations",      icon: FileText,     color: "text-amber-500" },
  extracting_regulations_done:{ label: "Regulations extracted",       icon: CheckCircle2, color: "text-green-500" },
  extracting_policies:        { label: "Extracting policies",         icon: FileText,     color: "text-amber-500" },
  extracting_policies_done:   { label: "Policies extracted",          icon: CheckCircle2, color: "text-green-500" },
  geocoding:                  { label: "Geocoding",                   icon: MapPin,       color: "text-violet-500" },
  storing:                    { label: "Saving to database",          icon: Database,     color: "text-indigo-500" },
  embedding:                  { label: "Building AI knowledge base",  icon: Brain,        color: "text-pink-500" },
  embedding_pdfs:             { label: "Processing PDFs",             icon: FileText,     color: "text-pink-500" },
  embedding_images:           { label: "Processing images",           icon: Brain,        color: "text-pink-500" },
  embedding_storing:          { label: "Storing embeddings",          icon: Database,     color: "text-pink-500" },
  embedding_done:             { label: "Embeddings complete",         icon: CheckCircle2, color: "text-green-500" },
  embedding_warning:          { label: "Embedding warning",           icon: AlertTriangle,color: "text-amber-500" },
  completed:                  { label: "Completed",                   icon: CheckCircle2, color: "text-green-600" },
  failed:                     { label: "Failed",                      icon: XCircle,      color: "text-destructive" },
};

const MAIN_STEPS = [
  "discovering", "scraping", "extracting", "geocoding", "storing", "embedding", "completed",
];

// ---------------------------------------------------------------------------
// Pipeline sub-components
// ---------------------------------------------------------------------------

function elapsedStr(startedAt: string, completedAt: string | null): string {
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const sec = Math.floor((end - start) / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const s = sec % 60;
  return `${min}m ${s}s`;
}

function StepProgressBar({ currentStep }: { currentStep: string }) {
  const currentMainIdx = MAIN_STEPS.findIndex(
    (s) => currentStep === s || currentStep.startsWith(s),
  );
  return (
    <div className="flex items-center gap-1 mt-3">
      {MAIN_STEPS.map((step, i) => {
        const meta = STEP_META[step];
        const isDone = i < currentMainIdx || currentStep === "completed";
        const isActive = i === currentMainIdx && currentStep !== "completed";
        return (
          <div key={step} className="flex-1 flex flex-col items-center gap-1">
            <div
              className={`h-1.5 w-full rounded-full transition-all duration-500 ${
                isDone ? "bg-green-500" : isActive ? "bg-primary animate-pulse" : "bg-muted"
              }`}
            />
            <span className={`text-[9px] leading-none ${isDone ? "text-green-600" : isActive ? "text-primary font-medium" : "text-muted-foreground/50"}`}>
              {meta?.label.split(" ")[0] || step}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function EventLog({ events }: { events: PipelineEvent[] }) {
  const displayEvents = useMemo(() => events.filter((e) => e.message), [events]);
  if (displayEvents.length === 0) return null;

  return (
    <div className="mt-3 space-y-1 max-h-52 overflow-y-auto">
      {displayEvents.map((event, i) => {
        const meta = STEP_META[event.step];
        const Icon = meta?.icon || Activity;
        const colorClass = meta?.color || "text-muted-foreground";
        const isLatest = i === displayEvents.length - 1;
        return (
          <div key={i} className={`flex items-start gap-2 px-2 py-1.5 rounded-md text-xs transition-colors ${isLatest ? "bg-muted/60" : ""}`}>
            <Icon className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${colorClass} ${isLatest && event.step !== "completed" && event.step !== "failed" ? "animate-pulse" : ""}`} />
            <div className="flex-1 min-w-0">
              <p className={`leading-snug ${isLatest ? "text-foreground font-medium" : "text-muted-foreground"}`}>{event.message}</p>
              {event.detail && <p className="text-[10px] text-muted-foreground/70 mt-0.5">{event.detail as string}</p>}
              {(() => {
                const urls = event.urls;
                if (!urls || !Array.isArray(urls)) return null;
                const list = urls as string[];
                return (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {list.slice(0, 5).map((url, j) => {
                      try { return <span key={j} className="text-[10px] bg-muted px-1.5 py-0.5 rounded font-mono truncate max-w-[200px] inline-block">{new URL(url).hostname}</span>; }
                      catch { return <span key={j} className="text-[10px] bg-muted px-1.5 py-0.5 rounded font-mono truncate max-w-[200px] inline-block">{url}</span>; }
                    })}
                    {list.length > 5 && <span className="text-[10px] text-muted-foreground">+{list.length - 5} more</span>}
                  </div>
                );
              })()}
            </div>
            <span className="text-[9px] text-muted-foreground/50 flex-shrink-0 tabular-nums mt-0.5">
              {new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function JobCard({ job }: { job: PipelineJob }) {
  const [expanded, setExpanded] = useState(job.status === "running");
  const elapsed = elapsedStr(job.started_at, job.completed_at);
  const events = job.events || [];
  const latestMessage = [...events].reverse().find((e) => e.message)?.message;

  const jobStats = useMemo(() => {
    const s: { label: string; value: string | number }[] = [];
    const data = job.step_data || {};
    if (data.urls_scraped) s.push({ label: "Pages scraped", value: data.urls_scraped as number });
    if (data.chars_collected) s.push({ label: "Text collected", value: `${((data.chars_collected as number) / 1000).toFixed(0)}K chars` });
    if (job.regulations_found) s.push({ label: "Regulations", value: job.regulations_found });
    if (job.policies_found) s.push({ label: "Policies", value: job.policies_found });
    const embEvt = events.find((e) => e.step === "embedding_done");
    if (embEvt?.chunks_stored) s.push({ label: "AI chunks", value: embEvt.chunks_stored as number });
    const pdfEvt = events.find((e) => e.step === "embedding_pdfs");
    if (pdfEvt?.pdfs) s.push({ label: "PDFs", value: pdfEvt.pdfs as number });
    const imgEvt = events.find((e) => e.step === "embedding_images");
    if (imgEvt?.images) s.push({ label: "Images", value: imgEvt.images as number });
    return s;
  }, [job, events]);

  return (
    <div className={`rounded-lg border transition-colors ${
      job.status === "running" ? "border-primary/30 bg-primary/5"
        : job.status === "completed" ? "border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950/30"
        : "border-destructive/30 bg-destructive/5"
    }`}>
      <button onClick={() => setExpanded((v) => !v)} className="w-full flex items-center gap-3 px-4 py-3 text-left">
        {job.status === "running" ? <Loader2 className="w-5 h-5 text-primary animate-spin flex-shrink-0" />
          : job.status === "completed" ? <CheckCircle2 className="w-5 h-5 text-green-600 flex-shrink-0" />
          : <XCircle className="w-5 h-5 text-destructive flex-shrink-0" />}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{job.municipality}</span>
            <Badge variant="secondary" className="text-[10px]">{job.province}</Badge>
          </div>
          {latestMessage && <p className="text-xs text-muted-foreground mt-0.5 truncate">{latestMessage}</p>}
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-muted-foreground tabular-nums">{elapsed}</span>
          {expanded ? <ChevronUp className="w-4 h-4 text-muted-foreground" /> : <ChevronDown className="w-4 h-4 text-muted-foreground" />}
        </div>
      </button>
      {job.status === "running" && <div className="px-4 pb-2"><StepProgressBar currentStep={job.current_step} /></div>}
      {expanded && (
        <div className="px-4 pb-4 border-t border-border/50 mt-1 pt-3 animate-fade-in">
          {jobStats.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-3">
              {jobStats.map((s) => (
                <div key={s.label} className="bg-background/60 rounded-md px-3 py-2 border border-border/50">
                  <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{s.label}</p>
                  <p className="text-sm font-semibold tabular-nums mt-0.5">{s.value}</p>
                </div>
              ))}
            </div>
          )}
          <EventLog events={events} />
          {job.error && (
            <div className="mt-3 p-2 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-xs text-destructive font-medium">{job.error}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function HomePage() {
  const [query, setQuery] = useState("");
  const { sendQuestion } = useChatContext();

  const [municipality, setMunicipality] = useState("");
  const [province, setProvince] = useState("Ontario");
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState<PipelineJob[]>([]);
  const [municipalities, setMunicipalities] = useState<Municipality[]>([]);
  const [zoneStats, setZoneStats] = useState<ZoneStats[]>([]);
  const [logs, setLogs] = useState<ScrapeLog[]>([]);
  const [error, setError] = useState("");
  const [loadingMunis, setLoadingMunis] = useState(true);

  const fetchStatus = useCallback(async () => {
    try {
      const [jobData, logData] = await Promise.all([
        api.getPipelineStatus(),
        api.getPipelineLogs(),
      ]);
      setJobs(jobData);
      setLogs(logData);
    } catch (err) {
      console.error("Failed to fetch pipeline status:", err);
    }
  }, []);

  const fetchMunicipalities = useCallback(async () => {
    try {
      const [muniData, statsData] = await Promise.all([
        api.getMunicipalities(),
        api.getZoneStats(),
      ]);
      setMunicipalities(muniData);
      setZoneStats(statsData);
    } catch (err) {
      console.error("Failed to fetch municipalities:", err);
    } finally {
      setLoadingMunis(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchMunicipalities();
  }, [fetchStatus, fetchMunicipalities]);

  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === "running");
    if (!hasRunning) return;
    const interval = setInterval(() => {
      fetchStatus();
      fetchMunicipalities();
    }, 2000);
    return () => clearInterval(interval);
  }, [jobs, fetchStatus, fetchMunicipalities]);

  const totalZones = zoneStats.reduce((sum, s) => sum + s.zone_count, 0);
  const avgRestrict =
    zoneStats.length > 0
      ? zoneStats.reduce((sum, s) => sum + (s.avg_restrictiveness || 0), 0) / zoneStats.length
      : 0;

  function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (query.trim()) {
      sendQuestion(query.trim());
      setQuery("");
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!municipality.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      await api.runPipeline(municipality.trim(), province);
      setMunicipality("");
      await fetchStatus();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start pipeline");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleRescrape(name: string, prov: string) {
    try {
      await api.runPipeline(name, prov);
      await fetchStatus();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start pipeline");
    }
  }

  async function handleDelete(name: string) {
    if (!confirm(`Delete all data for ${name}? This cannot be undone.`)) return;
    try {
      await api.deleteMunicipality(name);
      await fetchMunicipalities();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    }
  }

  function getStats(muniName: string): ZoneStats | undefined {
    return zoneStats.find((s) => s.municipality.toLowerCase() === muniName.toLowerCase());
  }

  const runningJobs = jobs.filter((j) => j.status === "running");
  const recentJobs = jobs.filter((j) => j.status !== "running");

  return (
    <div className="flex flex-col min-h-full">
      {/* Hero — AI input */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-background to-accent/10" />
        <div className="relative max-w-3xl mx-auto px-4 pt-12 pb-8 sm:pt-16 sm:pb-10 text-center">
          <div className="animate-fade-in-up">
            <div className="inline-flex items-center gap-2 text-sm text-muted-foreground mb-3 bg-muted/50 px-3 py-1 rounded-full border border-border">
              <Sparkles className="w-3.5 h-3.5 text-primary" />
              AI-powered zoning intelligence
            </div>
            <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold tracking-tight text-foreground leading-tight">
              Ask anything about{" "}
              <span className="text-primary">Canadian zoning</span>
            </h1>
            <p className="mt-2 text-sm sm:text-base text-muted-foreground max-w-lg mx-auto leading-relaxed">
              Query zoning bylaws, compare regulations, and explore land use data
              — powered by an agentic AI.
            </p>
          </div>

          <form onSubmit={handleAsk} className="mt-6 animate-fade-in-up" style={{ animationDelay: "150ms" }}>
            <div className="relative max-w-xl mx-auto">
              <div className="flex items-center gap-2 bg-background border border-border rounded-2xl px-4 py-2 shadow-lg focus-within:ring-2 focus-within:ring-primary/30 focus-within:border-primary/50 transition-all">
                <Sparkles className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask about zoning in any Canadian city..."
                  className="flex-1 bg-transparent border-0 outline-none text-sm py-1.5 placeholder:text-muted-foreground/60"
                />
                <button
                  type="submit"
                  disabled={!query.trim()}
                  className="w-8 h-8 rounded-xl bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-30 hover:opacity-90 transition-opacity flex-shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </form>

          <div className="mt-4 flex flex-wrap justify-center gap-2 max-w-xl mx-auto animate-fade-in-up" style={{ animationDelay: "300ms" }}>
            {SUGGESTED_QUESTIONS.map((q, i) => (
              <button
                key={i}
                onClick={() => sendQuestion(q)}
                className="text-[12px] text-muted-foreground bg-muted/50 border border-border px-3 py-1.5 rounded-full hover:bg-accent hover:text-accent-foreground transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Dashboard content */}
      <div className="max-w-5xl mx-auto px-4 pb-10 w-full space-y-6">
        {/* Stats bar */}
        {(loadingMunis || zoneStats.length > 0) && (
          <section className="animate-fade-in-up" style={{ animationDelay: "400ms" }}>
            <div className="grid grid-cols-3 gap-3">
              {loadingMunis ? (
                Array.from({ length: 3 }).map((_, i) => (
                  <Card key={i}><CardContent className="p-4"><Skeleton className="h-4 w-20 mb-2" /><Skeleton className="h-8 w-16" /></CardContent></Card>
                ))
              ) : (
                <>
                  <Card className="card-hover"><CardContent className="p-4">
                    <div className="flex items-center gap-2 text-muted-foreground mb-1"><Layers className="w-4 h-4 text-blue-600" /><span className="text-xs font-medium">Municipalities</span></div>
                    <p className="text-2xl font-semibold tracking-tight">{municipalities.length}</p>
                  </CardContent></Card>
                  <Card className="card-hover"><CardContent className="p-4">
                    <div className="flex items-center gap-2 text-muted-foreground mb-1"><Building2 className="w-4 h-4 text-green-600" /><span className="text-xs font-medium">Zone Types</span></div>
                    <p className="text-2xl font-semibold tracking-tight">{totalZones}</p>
                  </CardContent></Card>
                  <Card className="card-hover"><CardContent className="p-4">
                    <div className="flex items-center gap-2 text-muted-foreground mb-1"><TrendingUp className="w-4 h-4 text-orange-600" /><span className="text-xs font-medium">Avg Restrictiveness</span></div>
                    <p className="text-2xl font-semibold tracking-tight">{avgRestrict > 0 ? `${avgRestrict.toFixed(0)}/100` : "N/A"}</p>
                  </CardContent></Card>
                </>
              )}
            </div>
          </section>
        )}

        {/* Add Municipality */}
        <Card className="animate-fade-in-up">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Plus className="w-4 h-4" />
              Add Municipality
            </CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-2">
              <Input
                value={municipality}
                onChange={(e) => setMunicipality(e.target.value)}
                placeholder="e.g. London, Kitchener, Markham..."
                className="flex-1 h-9"
                disabled={submitting}
              />
              <Select value={province} onValueChange={(v) => v && setProvince(v)} disabled={submitting}>
                <SelectTrigger className="w-48 h-9"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {PROVINCES.map((p) => <SelectItem key={p} value={p}>{p}</SelectItem>)}
                </SelectContent>
              </Select>
              <Button type="submit" disabled={submitting || !municipality.trim()} className="h-9 gap-1.5">
                {submitting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
                Scrape
              </Button>
            </form>
            {error && <p className="mt-2 text-sm text-destructive bg-destructive/10 px-3 py-2 rounded-md">{error}</p>}
          </CardContent>
        </Card>

        {/* Active Jobs */}
        {runningJobs.length > 0 && (
          <Card className="animate-fade-in-up">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                Active Pipeline
                <Badge variant="secondary" className="text-[10px]">{runningJobs.length} running</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {runningJobs.map((job) => <JobCard key={job.job_id} job={job} />)}
            </CardContent>
          </Card>
        )}

        {/* Recent Jobs */}
        {recentJobs.length > 0 && (
          <Card className="animate-fade-in-up">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Recent Jobs
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {recentJobs.map((job) => <JobCard key={job.job_id} job={job} />)}
            </CardContent>
          </Card>
        )}

        {/* Municipalities Table */}
        <Card className="animate-fade-in-up">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Tracked Municipalities</CardTitle>
              {municipalities.length > 0 && (
                <Link href="/explore" className="text-xs text-primary hover:underline flex items-center gap-1">
                  Explore data <ArrowRight className="w-3 h-3" />
                </Link>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {loadingMunis ? (
              <div className="space-y-2">
                {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-12 w-full" />)}
              </div>
            ) : municipalities.length === 0 ? (
              <div className="text-center py-6">
                <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mx-auto mb-3">
                  <MapPin className="w-5 h-5 text-muted-foreground" />
                </div>
                <p className="text-sm text-muted-foreground">No municipalities yet. Add one above to get started.</p>
              </div>
            ) : (
              <div className="rounded-md border border-border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50 hover:bg-muted/50">
                      <TableHead>Municipality</TableHead>
                      <TableHead>Province</TableHead>
                      <TableHead className="text-right">Zones</TableHead>
                      <TableHead className="text-right">Restrictiveness</TableHead>
                      <TableHead>Last Updated</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {municipalities.map((m) => {
                      const s = getStats(m.name);
                      return (
                        <TableRow key={m.id}>
                          <TableCell>
                            <button
                              onClick={() => sendQuestion(`What are the main zoning categories in ${m.name}?`)}
                              className="font-medium text-primary hover:underline"
                            >
                              {m.name}
                            </button>
                          </TableCell>
                          <TableCell className="text-muted-foreground">{m.province}</TableCell>
                          <TableCell className="text-right tabular-nums">{s?.zone_count || 0}</TableCell>
                          <TableCell className="text-right tabular-nums">
                            {s?.avg_restrictiveness != null ? `${s.avg_restrictiveness.toFixed(0)}/100` : "N/A"}
                          </TableCell>
                          <TableCell className="text-muted-foreground text-sm">
                            {new Date(m.last_updated).toLocaleDateString()}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Button variant="ghost" size="sm" className="h-7 text-xs gap-1" onClick={() => handleRescrape(m.name, m.province)}>
                                <RefreshCw className="w-3 h-3" /> Re-scrape
                              </Button>
                              <Button variant="ghost" size="sm" className="h-7 text-xs gap-1 text-destructive hover:text-destructive" onClick={() => handleDelete(m.name)}>
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Scrape History */}
        {logs.length > 0 && (
          <Card className="animate-fade-in-up">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Clock className="w-4 h-4" />
                Scrape History
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              {logs.slice(0, 20).map((log) => (
                <div key={log.id} className="flex items-center justify-between py-1.5 px-3 rounded-md hover:bg-muted/50 transition-colors">
                  <div className="flex items-center gap-2.5">
                    {log.status === "success" ? <CheckCircle2 className="w-3.5 h-3.5 text-green-500" /> : <XCircle className="w-3.5 h-3.5 text-destructive" />}
                    <span className="text-sm font-medium">{log.municipality}</span>
                    <Badge variant="secondary" className="text-[10px] font-normal">{log.documents_found} docs</Badge>
                  </div>
                  <span className="text-xs text-muted-foreground">{log.scraped_at ? new Date(log.scraped_at).toLocaleString() : ""}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

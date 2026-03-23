"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, ZoningRegulation, ZoneStats } from "@/lib/api";
import { MapView } from "@/components/MapView";
import { ZoneTable } from "@/components/ZoneTable";
import { Map, TableIcon, Download, Search } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

const CATEGORIES = [
  { value: "all", label: "All categories" },
  { value: "residential", label: "Residential" },
  { value: "commercial", label: "Commercial" },
  { value: "industrial", label: "Industrial" },
  { value: "mixed_use", label: "Mixed Use" },
  { value: "institutional", label: "Institutional" },
  { value: "open_space", label: "Open Space" },
  { value: "agricultural", label: "Agricultural" },
  { value: "other", label: "Other" },
];

const LEGEND_ITEMS = [
  { color: "bg-green-500", label: "Low (0-25)" },
  { color: "bg-yellow-500", label: "Medium (25-50)" },
  { color: "bg-orange-500", label: "High (50-75)" },
  { color: "bg-red-500", label: "Very High (75-100)" },
];

export default function ExplorePage() {
  return (
    <Suspense fallback={<div className="p-8"><Skeleton className="h-96 w-full" /></div>}>
      <ExploreContent />
    </Suspense>
  );
}

function ExploreContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";

  const [zones, setZones] = useState<ZoningRegulation[]>([]);
  const [stats, setStats] = useState<ZoneStats[]>([]);
  const [municipality, setMunicipality] = useState(initialQuery);
  const [category, setCategory] = useState("all");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, [municipality, category]);

  async function loadData() {
    setLoading(true);
    try {
      const params: Record<string, string> = {};
      if (municipality) params.municipality = municipality;
      if (category && category !== "all") params.category = category;

      const [zonesData, statsData] = await Promise.all([
        api.getZones(params),
        api.getZoneStats(),
      ]);
      setZones(zonesData);
      setStats(statsData);
    } catch (err) {
      console.error("Failed to load data:", err);
    } finally {
      setLoading(false);
    }
  }

  const totalZones = stats.reduce((sum, s) => sum + s.zone_count, 0);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="border-b border-border bg-background px-4 py-3 animate-fade-in-up">
        <div className="flex flex-col sm:flex-row gap-2 items-start sm:items-center">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              value={municipality}
              onChange={(e) => setMunicipality(e.target.value)}
              placeholder="Filter by municipality..."
              className="pl-8 h-9 text-sm"
            />
          </div>

          <Select value={category} onValueChange={(v) => v && setCategory(v)}>
            <SelectTrigger className="w-44 h-9 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CATEGORIES.map((c) => (
                <SelectItem key={c.value} value={c.value}>
                  {c.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="flex items-center gap-2 ml-auto">
            <Badge variant="secondary" className="text-xs font-normal">
              {loading ? "..." : `${totalZones} zones`}
            </Badge>

            <DropdownMenu>
              <DropdownMenuTrigger className="inline-flex items-center justify-center gap-1.5 rounded-md border border-input bg-background px-3 h-9 text-sm font-medium shadow-xs hover:bg-accent hover:text-accent-foreground transition-colors">
                <Download className="w-3.5 h-3.5" />
                Export
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem>
                  <a href={api.getExportUrl("csv")} className="w-full">Export as CSV</a>
                </DropdownMenuItem>
                <DropdownMenuItem>
                  <a href={api.getExportUrl("json")} className="w-full">Export as JSON</a>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <Tabs defaultValue="table" className="h-full flex flex-col">
          <div className="border-b border-border px-4">
            <TabsList className="h-10">
              <TabsTrigger value="table" className="gap-1.5 text-sm">
                <TableIcon className="w-3.5 h-3.5" />
                Table
              </TabsTrigger>
              <TabsTrigger value="map" className="gap-1.5 text-sm">
                <Map className="w-3.5 h-3.5" />
                Map
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="table" className="flex-1 overflow-y-auto mt-0 p-4 animate-fade-in">
            {loading ? (
              <div className="space-y-2 stagger-children">
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-12 w-full" />
                ))}
              </div>
            ) : (
              <div className="animate-fade-in">
                <ZoneTable zones={zones} />
              </div>
            )}
          </TabsContent>

          <TabsContent value="map" className="flex-1 mt-0 p-4">
            <div className="flex gap-3 mb-3 text-xs text-muted-foreground">
              {LEGEND_ITEMS.map((item) => (
                <div key={item.label} className="flex items-center gap-1.5">
                  <div className={`w-2.5 h-2.5 rounded-full ${item.color}`} />
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
            <div className="rounded-lg overflow-hidden border border-border">
              <MapView />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

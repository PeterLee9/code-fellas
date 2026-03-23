"use client";

import { useEffect, useState } from "react";
import { api, Municipality, ZoneStats } from "@/lib/api";
import dynamic from "next/dynamic";

interface EnrichedMunicipality extends Municipality {
  stats?: ZoneStats;
}

const LeafletMapDynamic = dynamic(
  () => import("./LeafletMap").then((mod) => mod.LeafletMap),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-[600px] text-gray-500">
        Loading map...
      </div>
    ),
  }
);

export function MapView() {
  const [municipalities, setMunicipalities] = useState<EnrichedMunicipality[]>(
    []
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getMunicipalities(), api.getZoneStats()])
      .then(([m, s]) => {
        const enriched = m.map((muni) => ({
          ...muni,
          stats: s.find((stat) => stat.municipality === muni.name),
        }));
        setMunicipalities(enriched);
      })
      .catch((err) => {
        console.error("Failed to load map data:", err);
        setError("Failed to load map data. Please try refreshing the page.");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[600px] text-gray-500">
        Loading map data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[600px] text-red-500">
        {error}
      </div>
    );
  }

  return <LeafletMapDynamic municipalities={municipalities} />;
}

"use client";

import { MapContainer, TileLayer, Marker, Popup, CircleMarker } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Municipality, ZoneStats } from "@/lib/api";

// Fix default marker icon
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

interface EnrichedMunicipality extends Municipality {
  stats?: ZoneStats;
}

function getRestrictivenessColor(score: number | null | undefined): string {
  if (score == null) return "#6B7280";
  if (score > 75) return "#EF4444";
  if (score > 50) return "#F97316";
  if (score > 25) return "#EAB308";
  return "#22C55E";
}

export function LeafletMap({
  municipalities,
}: {
  municipalities: EnrichedMunicipality[];
}) {
  const center: [number, number] = [43.7, -79.4]; // Ontario center
  const zoom = 7;

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      className="h-[600px] w-full rounded-xl"
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {municipalities.map((m) => {
        const score = m.stats?.avg_restrictiveness;
        const color = getRestrictivenessColor(score);
        const radius = Math.max(10, (m.stats?.zone_count || 1) * 1.5);

        return (
          <CircleMarker
            key={m.id}
            center={[m.latitude, m.longitude]}
            radius={radius}
            pathOptions={{
              color: color,
              fillColor: color,
              fillOpacity: 0.6,
              weight: 2,
            }}
          >
            <Popup>
              <div className="text-sm">
                <h3 className="font-bold text-base">{m.name}, {m.province}</h3>
                {m.population && (
                  <p className="text-gray-600">Pop: {m.population.toLocaleString()}</p>
                )}
                {m.stats && (
                  <>
                    <p className="mt-1">
                      <span className="font-medium">Zones:</span> {m.stats.zone_count}
                    </p>
                    <p>
                      <span className="font-medium">Avg Restrictiveness:</span>{" "}
                      <span style={{ color }}>{score?.toFixed(1) ?? "N/A"}/100</span>
                    </p>
                    {m.stats.needs_review_count > 0 && (
                      <p className="text-amber-600">
                        {m.stats.needs_review_count} zones need review
                      </p>
                    )}
                  </>
                )}
              </div>
            </Popup>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

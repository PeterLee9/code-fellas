"use client";

import { ZoningRegulation } from "@/lib/api";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const CATEGORY_VARIANT: Record<string, string> = {
  residential: "bg-green-50 text-green-700 border-green-200",
  commercial: "bg-blue-50 text-blue-700 border-blue-200",
  industrial: "bg-purple-50 text-purple-700 border-purple-200",
  mixed_use: "bg-orange-50 text-orange-700 border-orange-200",
  institutional: "bg-pink-50 text-pink-700 border-pink-200",
  open_space: "bg-emerald-50 text-emerald-700 border-emerald-200",
  agricultural: "bg-yellow-50 text-yellow-700 border-yellow-200",
  other: "bg-muted text-muted-foreground border-border",
};

export function ZoneTable({ zones }: { zones: ZoningRegulation[] }) {
  if (zones.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        <p className="text-sm">No zoning regulations found matching your filters.</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50 hover:bg-muted/50">
              <TableHead className="font-medium">Municipality</TableHead>
              <TableHead className="font-medium">Zone</TableHead>
              <TableHead className="font-medium">Category</TableHead>
              <TableHead className="text-right font-medium">Lot Size</TableHead>
              <TableHead className="text-right font-medium">Height</TableHead>
              <TableHead className="text-right font-medium">Stories</TableHead>
              <TableHead className="text-right font-medium">Coverage</TableHead>
              <TableHead className="text-right font-medium">Parking</TableHead>
              <TableHead className="text-right font-medium">Restrictiveness</TableHead>
              <TableHead className="text-center font-medium">Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {zones.map((z) => (
              <TableRow key={z.id}>
                <TableCell className="font-medium">{z.municipality}</TableCell>
                <TableCell>
                  <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono">
                    {z.zone_code}
                  </code>
                  <span className="ml-1.5 text-muted-foreground text-xs">{z.zone_name}</span>
                </TableCell>
                <TableCell>
                  <Badge
                    variant="outline"
                    className={`text-[11px] font-medium ${
                      CATEGORY_VARIANT[z.zone_category] || CATEGORY_VARIANT.other
                    }`}
                  >
                    {z.zone_category.replace("_", " ")}
                  </Badge>
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {z.min_lot_size_sqm?.toFixed(0) ?? "—"}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {z.max_building_height_m?.toFixed(1) ?? "—"}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {z.max_stories ?? "—"}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {z.max_lot_coverage_pct != null ? `${z.max_lot_coverage_pct.toFixed(0)}%` : "—"}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {z.parking_spaces_per_unit?.toFixed(1) ?? "—"}
                </TableCell>
                <TableCell className="text-right">
                  {z.restrictiveness_score != null ? (
                    <RestrictivenessBar score={z.restrictiveness_score} />
                  ) : (
                    <span className="text-muted-foreground text-sm">—</span>
                  )}
                </TableCell>
                <TableCell className="text-center">
                  {z.needs_review ? (
                    <AlertTriangle className="w-4 h-4 text-amber-500 mx-auto" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-green-500 mx-auto" />
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      <div className="px-4 py-2.5 border-t border-border text-xs text-muted-foreground bg-muted/30">
        Showing {zones.length} zone{zones.length !== 1 ? "s" : ""}
      </div>
    </div>
  );
}

function RestrictivenessBar({ score }: { score: number }) {
  const color =
    score > 75
      ? "bg-red-500"
      : score > 50
        ? "bg-orange-500"
        : score > 25
          ? "bg-yellow-500"
          : "bg-green-500";

  return (
    <div className="flex items-center gap-2 justify-end">
      <div className="w-14 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs tabular-nums w-7 text-muted-foreground">{score.toFixed(0)}</span>
    </div>
  );
}
